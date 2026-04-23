"""
Hardware Cache Gymnasium Environment
====================================

This is the REAL HARDWARE version of cache_env.py.

It does the exact same thing as CacheEnv, but uses HardwareCache
(real GPU/CPU/Disk) instead of CacheSimulator (fake numbers).

What's the same:
  • Observation space: [query_embedding(384) | cache_stats(3) | candidate_scores(16)]
  • Action space: MultiBinary(16) — which candidates to prefetch
  • Reward function: R = α·time_saved − β·bytes_migrated − γ·unused_count
  • Trace format: reads the same CSV files
  • Episode structure: one episode = one workload trace

What's different:
  • access_latency_ms comes from REAL data transfer timing
  • Chunks are REAL tensors occupying actual GPU/CPU/Disk memory
  • The agent learns from real hardware behavior, not approximations

Drop-in usage:
  Replace CacheEnv with HardwareCacheEnv in train.py:
    # OLD: env = CacheEnv(trace_path, config=tier_cfg, query_embeddings=embs)
    # NEW: env = HardwareCacheEnv(trace_path, config=hw_cfg, query_embeddings=embs)
"""

from __future__ import annotations

import json
import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Any

from .hardware_cache import HardwareCache
from .hardware_config import HardwareConfig

# Import reward from the existing (unchanged) reward module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from env.reward import compute_reward, compute_baseline_latency


class HardwareCacheEnv(gym.Env):
    """
    RL Environment for KV-Cache prefetching with REAL hardware.

    This is a drop-in replacement for CacheEnv that uses actual
    GPU VRAM, CPU RAM, and NVMe disk instead of simulated caches.

    Observation: [query_embedding(384) | cache_stats(3) | candidate_recency(16)]
    Action:      MultiBinary(16) — which L3 candidates to prefetch
    Reward:      α·time_saved − β·bytes_migrated − γ·unused_prefetches

    The key insight: time_saved is now computed from REAL measured latencies
    versus an estimated baseline (what latency WOULD be without prefetch).

    Parameters
    ----------
    trace_path : str | Path
        Path to a trace CSV (same format as CacheEnv).
    config : HardwareConfig, optional
        Hardware configuration. Uses defaults if not provided.
    query_embeddings : np.ndarray, optional
        Pre-computed query embeddings of shape (n_queries, 384).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_path: str | Path,
        config: Optional[HardwareConfig] = None,
        query_embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__()

        # ─── Configuration ────────────────────────────────────
        self.cfg = config or HardwareConfig()

        # ─── Real hardware cache (instead of CacheSimulator) ──
        #     This creates actual GPU tensors, CPU pinned memory,
        #     and disk files for L3!
        self.hw_cache = HardwareCache(self.cfg)

        # ─── Load trace CSV ───────────────────────────────────
        #     Same format as CacheEnv: query_id, query_text,
        #     input_tokens, chunk_ids_needed
        self.trace_path = Path(trace_path)
        self.trace_df = pd.read_csv(self.trace_path)
        self.n_queries = len(self.trace_df)

        if self.cfg.verbose:
            print(f"\n[HardwareCacheEnv] 📂 Loaded trace: {self.trace_path.name}")
            print(f"[HardwareCacheEnv]    {self.n_queries} queries")

        # ─── Parse chunk_ids_needed from CSV ──────────────────
        self.chunk_lists: List[List[int]] = []
        for _, row in self.trace_df.iterrows():
            raw = row["chunk_ids_needed"]
            if isinstance(raw, str):
                chunks = json.loads(raw)
            else:
                chunks = [raw]
            self.chunk_lists.append(chunks)

        # ─── Query embeddings ─────────────────────────────────
        if query_embeddings is not None:
            assert query_embeddings.shape[0] == self.n_queries, (
                f"Embeddings shape {query_embeddings.shape[0]} != "
                f"trace length {self.n_queries}"
            )
            self.embeddings = query_embeddings
        else:
            self.embeddings = np.zeros(
                (self.n_queries, self.cfg.embed_dim), dtype=np.float32
            )

        # ─── Gymnasium spaces ─────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.obs_dim,), dtype=np.float32,
        )
        # Action space: first max_candidate_chunks for prefetch, then max_eviction_candidates for eviction
        total_action_dim = self.cfg.max_candidate_chunks + self.cfg.max_eviction_candidates
        self.action_space = gym.spaces.MultiBinary(total_action_dim)

        # ─── Episode tracking ─────────────────────────────────
        self.current_step = 0
        self.candidate_ids: List[int] = []
        self.eviction_candidate_ids: List[int] = []

        # ─── Metrics ──────────────────────────────────────────
        self.episode_rewards: List[float] = []
        self.episode_hits = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        self.episode_prefetches = 0
        self.episode_useful_prefetches = 0
        self.episode_measured_latencies: List[float] = []
        self.episode_baseline_latencies: List[float] = []

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset for a new episode.
        Clears the hardware cache (frees GPU memory, deletes disk files).
        """
        super().reset(seed=seed)

        if self.cfg.verbose:
            print(f"\n{'═' * 60}")
            print(f"  🔄 NEW EPISODE — {self.trace_path.name}")
            print(f"{'═' * 60}")

        # Reset the real hardware cache
        self.hw_cache.reset()

        # Reset episode state
        self.current_step = 0
        self.candidate_ids = []
        self.episode_rewards = []
        self.episode_hits = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        self.episode_prefetches = 0
        self.episode_useful_prefetches = 0
        self.episode_measured_latencies = []
        self.episode_baseline_latencies = []

        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step (one query arriving):

        1. Decode action → decide which cached chunks to evict and which L3 candidates to prefetch
        2. Execute evictions (free memory on hardware)
        3. Execute prefetch (REAL disk → CPU transfer!)
        4. Process the query (access needed chunks with REAL latency)
        5. Compute reward based on measured vs baseline latency (prefetch cost not included)
        6. Advance to next query

        The flow is identical to CacheEnv.step(), but with real hardware.
        """
        if self.current_step >= self.n_queries:
            obs = self._build_obs()
            return obs, 0.0, True, False, {}

        # Current query's needed chunks
        needed_chunks = self.chunk_lists[self.current_step]
        needed_set = set(needed_chunks)

        # ══════════════════════════════════════════════════════
        # STEP 1: Snapshot WHERE each needed chunk currently is
        # ══════════════════════════════════════════════════════
        # Before prefetching, check which tier holds each chunk.
        # This is used to compute the BASELINE latency (what would
        # have happened without any prefetching).
        pre_tiers = {}
        for cid in needed_chunks:
            pre_tiers[cid] = self.hw_cache.chunk_in_cache(cid)

        # Compute baseline using the CONFIG's latency estimates.
        # (We can't measure the "no-prefetch" latency without
        #  actually doing it, so we estimate from the config.)
        baseline_latency_ms = compute_baseline_latency(
            needed_chunks, pre_tiers, self.cfg
        )

        # ══════════════════════════════════════════════════════
        # STEP 2: Decode actions
        # ══════════════════════════════════════════════════════
        # First max_candidate_chunks for prefetch, rest for eviction
        prefetch_action = action[:self.cfg.max_candidate_chunks]
        eviction_action = action[self.cfg.max_candidate_chunks:]

        # ══════════════════════════════════════════════════════
        # STEP 3: Execute eviction actions (RL agent's decision!)
        # ══════════════════════════════════════════════════════
        evicted_ids = []
        for i, do_evict in enumerate(eviction_action):
            if do_evict and i < len(self.eviction_candidate_ids):
                cid = self.eviction_candidate_ids[i]
                if self.hw_cache.evict_chunk(cid):
                    evicted_ids.append(cid)

        # ══════════════════════════════════════════════════════
        # STEP 4: Execute prefetch actions (RL agent's decision!)
        # ══════════════════════════════════════════════════════
        # The agent outputs a binary vector: [0,1,0,1,...] indicating
        # which of the 16 candidate chunks to prefetch from L3 → L2.
        prefetched_ids = []

        for i, do_prefetch in enumerate(prefetch_action):
            if do_prefetch and i < len(self.candidate_ids):
                cid = self.candidate_ids[i]
                # REAL prefetch: loads chunk from NVMe disk → CPU pinned RAM
                cost = self.hw_cache.prefetch(cid)
                if cost > 0:
                    prefetched_ids.append(cid)

        self.episode_prefetches += len(prefetched_ids)
        useful = set(prefetched_ids) & needed_set
        self.episode_useful_prefetches += len(useful)

        # ══════════════════════════════════════════════════════
        # STEP 5: Process query — access all needed chunks
        # ══════════════════════════════════════════════════════
        # This is where REAL data movement happens!
        # Chunks in L1 → trivial GPU read
        # Chunks in L2 → CPU → GPU transfer (PCIe)
        # Chunks in L3 → Disk → CPU → GPU
        # Missing → create new tensor (simulates LLM forward pass)
        access_latency_ms, tier_counts = self.hw_cache.access_chunks(needed_chunks)

        for tier, count in tier_counts.items():
            self.episode_hits[tier] += count

        self.episode_measured_latencies.append(access_latency_ms)
        self.episode_baseline_latencies.append(baseline_latency_ms)

        # ══════════════════════════════════════════════════════
        # STEP 6: Compute reward (prefetch cost not included)
        # ══════════════════════════════════════════════════════
        # R = α × (baseline - actual) - β × MB_migrated - γ × unused
        #
        # If prefetching helped → baseline > actual → positive reward
        # If prefetching wasted → extra migration cost → negative penalty
        reward = compute_reward(
            prefetched_chunk_ids=prefetched_ids,
            actually_accessed_chunk_ids=needed_set,
            access_latency_ms=access_latency_ms,
            baseline_latency_ms=baseline_latency_ms,
            config=self.cfg,
        )
        self.episode_rewards.append(reward)

        # ══════════════════════════════════════════════════════
        # STEP 7: Advance to next query
        # ══════════════════════════════════════════════════════
        self.current_step += 1
        terminated = self.current_step >= self.n_queries
        truncated = False

        obs = self._build_obs()

        # ── Build info dict ──
        info = {
            "step": self.current_step,
            "reward": reward,
            "access_latency_ms": access_latency_ms,
            "baseline_latency_ms": baseline_latency_ms,
            "tier_counts": tier_counts,
            "n_prefetched": len(prefetched_ids),
            "n_evicted": len(evicted_ids),
            "n_useful": len(useful),
            "hardware_mode": "CUDA" if self.hw_cache.use_cuda else "CPU-only",
        }

        if self.cfg.verbose:
            time_saved = baseline_latency_ms - access_latency_ms
            print(f"\n  📊 Step {self.current_step}/{self.n_queries}  "
                  f"reward={reward:+.3f}  "
                  f"access={access_latency_ms:.2f}ms  "
                  f"baseline={baseline_latency_ms:.2f}ms  "
                  f"saved={time_saved:+.2f}ms  "
                  f"prefetched={len(prefetched_ids)} evicted={len(evicted_ids)} (useful={len(useful)})")

        # ── End-of-episode summary ──
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

            episode_summary = {
                "total_reward": sum(self.episode_rewards),
                "hit_rate_pct": round(hit_rate, 2),
                "total_prefetches": self.episode_prefetches,
                "useful_prefetches": self.episode_useful_prefetches,
                "prefetch_accuracy_pct": round(prefetch_accuracy, 2),
                "tier_counts": dict(self.episode_hits),
                "avg_measured_latency_ms": round(
                    np.mean(self.episode_measured_latencies), 2
                ),
                "avg_baseline_latency_ms": round(
                    np.mean(self.episode_baseline_latencies), 2
                ),
                "total_measured_latency_ms": round(
                    sum(self.episode_measured_latencies), 2
                ),
                "total_baseline_latency_ms": round(
                    sum(self.episode_baseline_latencies), 2
                ),
            }
            info["episode_summary"] = episode_summary

            if self.cfg.verbose:
                print(f"\n{'═' * 60}")
                print(f"  🏁 EPISODE COMPLETE — {self.trace_path.name}")
                print(f"{'═' * 60}")
                print(f"  Total reward:       {episode_summary['total_reward']:+.2f}")
                print(f"  Hit rate:           {episode_summary['hit_rate_pct']:.1f}%")
                print(f"  Prefetch accuracy:  {episode_summary['prefetch_accuracy_pct']:.1f}%")
                print(f"  Avg latency (real): {episode_summary['avg_measured_latency_ms']:.2f} ms")
                print(f"  Avg latency (base): {episode_summary['avg_baseline_latency_ms']:.2f} ms")
                print(f"  Tier hits: {dict(self.episode_hits)}")
                self.hw_cache.print_stats_summary()

        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        """
        Build the observation vector (identical logic to CacheEnv):
        [query_embedding(384) | cache_stats(3) | candidate_recency(16) | eviction_scores(16)]

        The observation is what the RL agent "sees" at each step.
        It contains:
          1. Embedding of the current query (what's being asked)
          2. Cache utilization stats (how full is each tier)
          3. Recency scores for candidate chunks (how recently each was used)
          4. Eviction opportunity scores for cached chunks
        """
        # ── Part 1: Query embedding (384 dims) ──
        if self.current_step < self.n_queries:
            embedding = self.embeddings[self.current_step]
        else:
            embedding = np.zeros(self.cfg.embed_dim, dtype=np.float32)

        # ── Part 2: Cache stats (3 dims) ──
        l1_frac, l2_mb, l3_mb = self.hw_cache.get_stats()
        l2_norm = l2_mb / (self.cfg.l2_capacity_bytes / (1024 * 1024))
        l3_norm = l3_mb / (self.cfg.l3_capacity_bytes / (1024 * 1024))
        cache_stats = np.array([l1_frac, l2_norm, l3_norm], dtype=np.float32)

        # ── Part 3: Candidate chunks + recency scores (16 dims) ──
        self.candidate_ids = self.hw_cache.get_all_candidates(
            self.cfg.max_candidate_chunks
        )

        # Recency score: how recently each candidate was accessed
        recent = self.hw_cache.get_recent_accesses(
            self.cfg.history_len * 10
        )
        recent_set = set(recent[-self.cfg.history_len * 5:]) if recent else set()

        recency_scores = np.zeros(self.cfg.max_candidate_chunks, dtype=np.float32)
        for i, cid in enumerate(self.candidate_ids):
            if cid in recent_set:
                try:
                    pos = len(recent) - 1 - recent[::-1].index(cid)
                    recency_scores[i] = 1.0 - (pos / max(len(recent), 1))
                except ValueError:
                    recency_scores[i] = 0.0

        # ── Part 4: Eviction candidates + eviction opportunity scores (16 dims) ──
        # Get chunks in L2+L3 (prioritize by LRU, most recent first)
        l2_ids = list(reversed(list(self.hw_cache.l2.keys())))
        l3_ids = list(reversed(list(self.hw_cache.l3.keys())))
        self.eviction_candidate_ids = (l2_ids + l3_ids)[:self.cfg.max_eviction_candidates]

        eviction_scores = np.zeros(self.cfg.max_eviction_candidates, dtype=np.float32)
        for i, cid in enumerate(self.eviction_candidate_ids):
            if cid in recent_set:
                # Recent chunks have high scores (harder to evict)
                try:
                    pos = len(recent) - 1 - recent[::-1].index(cid)
                    eviction_scores[i] = 1.0 - (pos / max(len(recent), 1))
                except ValueError:
                    eviction_scores[i] = 0.0
            else:
                # Old chunks have low scores (easier to evict)
                eviction_scores[i] = 0.0

        obs = np.concatenate([embedding, cache_stats, recency_scores, eviction_scores])
        return obs.astype(np.float32)

    def get_operation_log(self):
        """Get the hardware cache's operation log as a DataFrame."""
        return self.hw_cache.get_operation_log_df()

    def save_operation_log(self, filepath: str | Path):
        """Save the hardware cache's operation log to CSV."""
        self.hw_cache.save_operation_log(filepath)
