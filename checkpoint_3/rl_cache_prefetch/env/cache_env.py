"""
Gymnasium environment for RL cache prefetching.
One episode = one workload trace (50 queries).
Each step = one query arriving.
"""

from __future__ import annotations

import json
import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Any

from .cache_simulator import CacheSimulator
from .tier_config import TierConfig
from .reward import compute_reward, compute_baseline_latency


class CacheEnv(gym.Env):
    """
    RL Environment for KV-Cache prefetching decisions.

    Observation: [query_embedding(384) | cache_stats(3) | candidate_recency(16)]
    Action:      MultiBinary(16) — which L3 candidates to prefetch
    Reward:      α·time_saved − β·bytes_migrated − γ·unused_prefetches
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_path: str | Path,
        config: Optional[TierConfig] = None,
        query_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        trace_path : str | Path
            Path to a trace CSV with columns:
            query_id, query_text, input_tokens, chunk_ids_needed
        config : TierConfig, optional
            Tier configuration. Defaults if not provided.
        query_embeddings : np.ndarray, optional
            Pre-computed embeddings of shape (n_queries, 384).
            If None, uses zero embeddings (for testing).
        """
        super().__init__()
        self.cfg = config or TierConfig()
        self.sim = CacheSimulator(self.cfg)

        # Load trace
        self.trace_path = Path(trace_path)
        self.trace_df = pd.read_csv(self.trace_path)
        self.n_queries = len(self.trace_df)

        # Parse chunk_ids_needed from string lists in CSV
        self.chunk_lists: List[List[int]] = []
        for _, row in self.trace_df.iterrows():
            raw = row["chunk_ids_needed"]
            if isinstance(raw, str):
                chunks = json.loads(raw)
            else:
                chunks = [raw]
            self.chunk_lists.append(chunks)

        # Pre-computed embeddings or zeros
        if query_embeddings is not None:
            assert query_embeddings.shape[0] == self.n_queries
            self.embeddings = query_embeddings
        else:
            self.embeddings = np.zeros(
                (self.n_queries, self.cfg.embed_dim), dtype=np.float32
            )

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiBinary(self.cfg.max_candidate_chunks)

        # Episode state
        self.current_step = 0
        self.candidate_ids: List[int] = []

        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_hits = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        self.episode_prefetches = 0
        self.episode_useful_prefetches = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.candidate_ids = []
        self.episode_rewards = []
        self.episode_hits = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        self.episode_prefetches = 0
        self.episode_useful_prefetches = 0

        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step:
        1. Decode action → prefetch selected L3 candidates
        2. Process the query (access needed chunks)
        3. Compute reward
        4. Advance to next query
        """
        if self.current_step >= self.n_queries:
            obs = self._build_obs()
            return obs, 0.0, True, False, {}

        # Current query's needed chunks
        needed_chunks = self.chunk_lists[self.current_step]
        needed_set = set(needed_chunks)

        # ── 1. Snapshot pre-prefetch tier locations (for baseline) ──
        pre_tiers = {}
        for cid in needed_chunks:
            pre_tiers[cid] = self.sim.chunk_in_cache(cid)

        baseline_latency_ms = compute_baseline_latency(
            needed_chunks, pre_tiers, self.cfg
        )

        # ── 2. Execute prefetch actions ──
        prefetched_ids = []
        total_prefetch_cost_ms = 0.0
        for i, do_prefetch in enumerate(action):
            if do_prefetch and i < len(self.candidate_ids):
                cid = self.candidate_ids[i]
                cost = self.sim.prefetch(cid)
                if cost > 0:
                    prefetched_ids.append(cid)
                    total_prefetch_cost_ms += cost

        self.episode_prefetches += len(prefetched_ids)
        useful = set(prefetched_ids) & needed_set
        self.episode_useful_prefetches += len(useful)

        # ── 3. Process query (access needed chunks) ──
        access_latency_ms, tier_counts = self.sim.access_chunks(needed_chunks)

        for tier, count in tier_counts.items():
            self.episode_hits[tier] += count

        # ── 4. Compute reward ──
        reward = compute_reward(
            prefetched_chunk_ids=prefetched_ids,
            actually_accessed_chunk_ids=needed_set,
            prefetch_cost_ms=total_prefetch_cost_ms,
            access_latency_ms=access_latency_ms,
            baseline_latency_ms=baseline_latency_ms,
            config=self.cfg,
        )
        self.episode_rewards.append(reward)

        # ── 5. Advance ──
        self.current_step += 1
        terminated = self.current_step >= self.n_queries
        truncated = False

        obs = self._build_obs()

        info = {
            "step": self.current_step,
            "reward": reward,
            "access_latency_ms": access_latency_ms,
            "baseline_latency_ms": baseline_latency_ms,
            "prefetch_cost_ms": total_prefetch_cost_ms,
            "tier_counts": tier_counts,
            "n_prefetched": len(prefetched_ids),
            "n_useful": len(useful),
        }

        if terminated:
            total_accesses = sum(self.episode_hits.values())
            hit_rate = (
                (self.episode_hits["L1"] + self.episode_hits["L2"])
                / total_accesses * 100
                if total_accesses > 0 else 0
            )
            prefetch_accuracy = (
                self.episode_useful_prefetches / self.episode_prefetches * 100
                if self.episode_prefetches > 0 else 0
            )
            info["episode_summary"] = {
                "total_reward": sum(self.episode_rewards),
                "hit_rate_pct": round(hit_rate, 2),
                "total_prefetches": self.episode_prefetches,
                "useful_prefetches": self.episode_useful_prefetches,
                "prefetch_accuracy_pct": round(prefetch_accuracy, 2),
                "tier_counts": dict(self.episode_hits),
            }

        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        """
        Build the observation vector:
        [query_embedding(384) | cache_stats(3) | candidate_recency(16)]
        """
        # Query embedding
        if self.current_step < self.n_queries:
            embedding = self.embeddings[self.current_step]
        else:
            embedding = np.zeros(self.cfg.embed_dim, dtype=np.float32)

        # Cache stats: [l1_frac, l2_mb_normalized, l3_mb_normalized]
        l1_frac, l2_mb, l3_mb = self.sim.get_stats()
        l2_norm = l2_mb / (self.cfg.l2_capacity_bytes / (1024 * 1024))
        l3_norm = l3_mb / (self.cfg.l3_capacity_bytes / (1024 * 1024))
        cache_stats = np.array([l1_frac, l2_norm, l3_norm], dtype=np.float32)

        # L3 candidates + recency scores
        self.candidate_ids = self.sim.get_all_candidates(
            self.cfg.max_candidate_chunks
        )

        # Recency score: how recently each candidate was accessed
        # 1.0 = accessed in the last query, 0.0 = never accessed recently
        recent = self.sim.get_recent_accesses(
            self.cfg.history_len * 10  # larger window for scoring
        )
        recent_set = set(recent[-self.cfg.history_len * 5:]) if recent else set()

        recency_scores = np.zeros(self.cfg.max_candidate_chunks, dtype=np.float32)
        for i, cid in enumerate(self.candidate_ids):
            if cid in recent_set:
                # Score based on recency position
                try:
                    pos = len(recent) - 1 - recent[::-1].index(cid)
                    recency_scores[i] = 1.0 - (pos / max(len(recent), 1))
                except ValueError:
                    recency_scores[i] = 0.0

        obs = np.concatenate([embedding, cache_stats, recency_scores])
        return obs.astype(np.float32)
