"""
Gymnasium Environment for RL Cache Prefetching
================================================

Wraps HardwareCache into a Gymnasium env for RL training.

One episode = one workload trace (50 queries).
Each step = one query arriving.

Observation: [query_embedding(384) | cache_stats(3) | candidate_recency(16)] = 403 dims
Action:      MultiBinary(16) — which L2/L3 candidates to prefetch to a warmer tier
Reward:      α × time_saved - β × MB_migrated - γ × unused_prefetches
"""

from __future__ import annotations

import json
import collections
import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Set, Tuple

from cache import HardwareCache


# ═══════════════════════════════════════════════════════════════
# Reward Computation
# ═══════════════════════════════════════════════════════════════

def compute_reward(
    prefetched_ids: List[int],
    needed_ids: Set[int],
    prefetch_cost_ms: float,
    access_latency_ms: float,
    baseline_latency_ms: float,
    alpha: float,
    beta: float,
    gamma: float,
    chunk_size_bytes: int,
) -> float:
    """
    R = α × time_saved - β × MB_migrated - γ × unused_prefetches

    time_saved     = baseline_latency - actual_access_latency
    MB_migrated    = num_prefetched_chunks × chunk_size / 1e6
    unused_count   = |prefetched - needed|
    """
    time_saved_ms = baseline_latency_ms - access_latency_ms
    bytes_migrated_mb = len(prefetched_ids) * chunk_size_bytes / 1e6
    unused_count = len(set(prefetched_ids) - needed_ids)

    return alpha * time_saved_ms - beta * bytes_migrated_mb - gamma * unused_count


def compute_baseline_latency(
    chunk_ids: List[int],
    chunk_tiers: dict,
    l1_lat: float,
    l2_lat: float,
    l3_lat: float,
    cold_lat: float,
) -> float:
    """
    Compute what the latency WOULD have been without any prefetching.
    Each chunk is accessed from whatever tier it sits in right now.
    """
    total = 0.0
    for cid in chunk_ids:
        tier = chunk_tiers.get(cid)
        if tier == "L1":
            total += l1_lat
        elif tier == "L2":
            total += l2_lat
        elif tier == "L3":
            total += l3_lat
        else:
            total += cold_lat
    return total


# ═══════════════════════════════════════════════════════════════
# Simulated Cache (for fast behavioral cloning in Stage A)
# ═══════════════════════════════════════════════════════════════

class SimulatedCache:
    """
    Lightweight cache simulator using OrderedDicts (no GPU tensors).
    Used only in Stage A (behavioral cloning) for speed.
    Same eviction/access logic as HardwareCache but with fake latencies.
    """

    def __init__(self, l1_cap: int, l2_cap: int, l3_cap: int, chunk_size: int,
                 l1_lat: float, l2_lat: float, l3_lat: float, cold_lat: float,
                 prefetch_lat: float):
        self.l1: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.l2: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.l3: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.l1_bytes = 0
        self.l2_bytes = 0
        self.l3_bytes = 0
        self.l1_cap = l1_cap
        self.l2_cap = l2_cap
        self.l3_cap = l3_cap
        self.chunk_size = chunk_size
        self.l1_lat = l1_lat
        self.l2_lat = l2_lat
        self.l3_lat = l3_lat
        self.cold_lat = cold_lat
        self.prefetch_lat = prefetch_lat
        self.access_history: List[int] = []

    def reset(self):
        self.l1.clear(); self.l2.clear(); self.l3.clear()
        self.l1_bytes = self.l2_bytes = self.l3_bytes = 0
        self.access_history.clear()

    def _insert_l1(self, cid, size):
        while self.l1_bytes + size > self.l1_cap and self.l1:
            eid, esz = self.l1.popitem(last=False)
            self.l1_bytes -= esz
            self._insert_l2(eid, esz)
        self.l1[cid] = size
        self.l1_bytes += size

    def _insert_l2(self, cid, size):
        if cid in self.l2:
            self.l2.move_to_end(cid); return
        while self.l2_bytes + size > self.l2_cap and self.l2:
            eid, esz = self.l2.popitem(last=False)
            self.l2_bytes -= esz
            self._insert_l3(eid, esz)
        self.l2[cid] = size
        self.l2_bytes += size

    def _insert_l3(self, cid, size):
        if cid in self.l3:
            self.l3.move_to_end(cid); return
        while self.l3_bytes + size > self.l3_cap and self.l3:
            self.l3.popitem(last=False)
            self.l3_bytes -= size
        self.l3[cid] = size
        self.l3_bytes += size

    def access_chunk(self, cid):
        self.access_history.append(cid)
        sz = self.chunk_size
        if cid in self.l1:
            self.l1.move_to_end(cid)
            return "L1", self.l1_lat
        if cid in self.l2:
            del self.l2[cid]; self.l2_bytes -= sz
            self._insert_l1(cid, sz)
            return "L2", self.l2_lat
        if cid in self.l3:
            del self.l3[cid]; self.l3_bytes -= sz
            self._insert_l1(cid, sz)
            return "L3", self.l3_lat
        self._insert_l1(cid, sz)
        return "MISS", self.cold_lat

    def access_chunks(self, chunk_ids):
        total = 0.0
        counts = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        for cid in chunk_ids:
            tier, lat = self.access_chunk(cid)
            total += lat; counts[tier] += 1
        return total, counts

    def prefetch(self, cid):
        if cid in self.l1 or cid in self.l2: return 0.0
        if cid in self.l3:
            sz = self.l3.pop(cid); self.l3_bytes -= sz
            self._insert_l2(cid, sz)
            return self.prefetch_lat
        return 0.0

    def get_candidates(self, k):
        l3_ids = list(reversed(list(self.l3.keys())))
        l2_ids = list(reversed(list(self.l2.keys())))
        return (l3_ids + l2_ids)[:k]

    def get_stats(self):
        l1f = self.l1_bytes / self.l1_cap if self.l1_cap > 0 else 0
        l2f = self.l2_bytes / self.l2_cap if self.l2_cap > 0 else 0
        l3f = self.l3_bytes / self.l3_cap if self.l3_cap > 0 else 0
        return l1f, l2f, l3f

    def get_recent_accesses(self, n):
        return self.access_history[-n:]

    def chunk_in_cache(self, cid):
        if cid in self.l1: return "L1"
        if cid in self.l2: return "L2"
        if cid in self.l3: return "L3"
        return None


# ═══════════════════════════════════════════════════════════════
# Gymnasium Environment
# ═══════════════════════════════════════════════════════════════

class CachePrefetchEnv(gym.Env):
    """
    RL Environment for KV-Cache prefetching decisions.

    Works with EITHER HardwareCache (real GPU/CPU/Disk, for training Stage B)
    or SimulatedCache (fast, for behavioral cloning Stage A).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_path: str | Path,
        cache,  # HardwareCache or SimulatedCache
        query_embeddings: np.ndarray,
        embed_dim: int = 384,
        max_candidates: int = 16,
        history_len: int = 5,
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma_reward: float = 0.1,
        chunk_size_bytes: int = 3_145_728,
        l1_hit_latency_ms: float = 0.1,
        l2_hit_latency_ms: float = 0.24,
        l3_hit_latency_ms: float = 6.0,
        cold_compute_per_chunk_ms: float = 30.8,
    ):
        super().__init__()
        self.cache = cache
        self.embed_dim = embed_dim
        self.max_candidates = max_candidates
        self.history_len = history_len
        self.alpha = alpha
        self.beta = beta
        self.gamma_reward = gamma_reward
        self.chunk_size_bytes = chunk_size_bytes
        self.l1_lat = l1_hit_latency_ms
        self.l2_lat = l2_hit_latency_ms
        self.l3_lat = l3_hit_latency_ms
        self.cold_lat = cold_compute_per_chunk_ms

        # Observation dimension: embedding + 3 cache stats + candidate scores
        self.obs_dim = embed_dim + 3 + max_candidates

        # Load trace
        self.trace_path = Path(trace_path)
        self.trace_df = pd.read_csv(self.trace_path)
        self.n_queries = len(self.trace_df)

        # Parse chunk_ids_needed from CSV
        self.chunk_lists: List[List[int]] = []
        for _, row in self.trace_df.iterrows():
            raw = row["chunk_ids_needed"]
            if isinstance(raw, str):
                self.chunk_lists.append(json.loads(raw))
            else:
                self.chunk_lists.append([int(raw)])

        # Embeddings
        assert query_embeddings.shape[0] == self.n_queries, \
            f"Embeddings shape {query_embeddings.shape[0]} != trace length {self.n_queries}"
        self.embeddings = query_embeddings

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiBinary(max_candidates)

        # Episode state
        self.current_step = 0
        self.candidate_ids: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_hits = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        self.episode_prefetches = 0
        self.episode_useful_prefetches = 0
        self.episode_total_measured_ms = 0.0
        self.episode_total_baseline_ms = 0.0

        # Per-step detailed metrics (for CSV/JSON output)
        self.step_metrics: List[dict] = []
        # Tier latency accumulators (total ms spent in each tier)
        self.tier_latencies = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "MISS": 0.0}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cache.reset()
        self.current_step = 0
        self.candidate_ids = []
        self.episode_rewards = []
        self.episode_hits = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        self.episode_prefetches = 0
        self.episode_useful_prefetches = 0
        self.episode_total_measured_ms = 0.0
        self.episode_total_baseline_ms = 0.0
        self.step_metrics = []
        self.tier_latencies = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "MISS": 0.0}
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        """
        Execute one step:
          1. Snapshot pre-prefetch tier locations (for baseline)
          2. Execute prefetch actions
          3. Process the query (access all needed chunks)
          4. Compute reward
          5. Advance
        """
        if self.current_step >= self.n_queries:
            return self._build_obs(), 0.0, True, False, {}

        needed_chunks = self.chunk_lists[self.current_step]
        needed_set = set(needed_chunks)

        # 1. Snapshot tier locations BEFORE prefetch
        pre_tiers = {}
        for cid in needed_chunks:
            pre_tiers[cid] = self.cache.chunk_in_cache(cid)

        baseline_latency_ms = compute_baseline_latency(
            needed_chunks, pre_tiers,
            self.l1_lat, self.l2_lat, self.l3_lat, self.cold_lat,
        )

        # 2. Execute prefetch actions
        prefetched_ids = []
        total_prefetch_cost_ms = 0.0
        for i, do_prefetch in enumerate(action):
            if do_prefetch and i < len(self.candidate_ids):
                cid = self.candidate_ids[i]
                cost = self.cache.prefetch(cid)
                if cost > 0:
                    prefetched_ids.append(cid)
                    total_prefetch_cost_ms += cost

        self.episode_prefetches += len(prefetched_ids)
        useful = set(prefetched_ids) & needed_set
        self.episode_useful_prefetches += len(useful)

        # 3. Process query — access all needed chunks (with per-chunk detail)
        if hasattr(self.cache, 'access_chunks_detailed'):
            access_latency_ms, tier_counts, chunk_details = (
                self.cache.access_chunks_detailed(needed_chunks)
            )
        else:
            access_latency_ms, tier_counts = self.cache.access_chunks(needed_chunks)
            chunk_details = []

        for tier, count in tier_counts.items():
            self.episode_hits[tier] += count

        # Accumulate per-tier latencies from chunk details
        for cd in chunk_details:
            self.tier_latencies[cd["tier"]] += cd["latency_ms"]

        self.episode_total_measured_ms += access_latency_ms + total_prefetch_cost_ms
        self.episode_total_baseline_ms += baseline_latency_ms

        # Estimated TTFT: time from query arrival to first chunk being ready
        # = prefetch cost + first chunk's access latency
        est_ttft_ms = total_prefetch_cost_ms
        if chunk_details:
            est_ttft_ms += chunk_details[0]["latency_ms"]
        elif len(needed_chunks) > 0:
            est_ttft_ms += access_latency_ms / len(needed_chunks)

        # Cache occupancy snapshot
        if hasattr(self.cache, 'snapshot_occupancy'):
            self.cache.snapshot_occupancy(self.current_step)
        l1_frac, l2_frac, l3_frac = self.cache.get_stats()

        # 4. Compute reward
        reward = compute_reward(
            prefetched_ids, needed_set, total_prefetch_cost_ms,
            access_latency_ms, baseline_latency_ms,
            self.alpha, self.beta, self.gamma_reward, self.chunk_size_bytes,
        )
        self.episode_rewards.append(reward)

        # 5. Record per-step metrics
        step_metric = {
            "step": self.current_step + 1,
            "reward": round(reward, 4),
            "access_latency_ms": round(access_latency_ms, 4),
            "baseline_latency_ms": round(baseline_latency_ms, 4),
            "prefetch_cost_ms": round(total_prefetch_cost_ms, 4),
            "est_ttft_ms": round(est_ttft_ms, 4),
            "n_chunks": len(needed_chunks),
            "n_prefetched": len(prefetched_ids),
            "n_useful": len(useful),
            "tier_L1": tier_counts.get("L1", 0),
            "tier_L2": tier_counts.get("L2", 0),
            "tier_L3": tier_counts.get("L3", 0),
            "tier_MISS": tier_counts.get("MISS", 0),
            "cache_l1_pct": round(l1_frac * 100, 1),
            "cache_l2_pct": round(l2_frac * 100, 1),
        }
        self.step_metrics.append(step_metric)

        # 6. Advance
        self.current_step += 1
        terminated = self.current_step >= self.n_queries

        obs = self._build_obs()

        info = {
            "step": self.current_step,
            "reward": reward,
            "access_latency_ms": access_latency_ms,
            "baseline_latency_ms": baseline_latency_ms,
            "prefetch_cost_ms": total_prefetch_cost_ms,
            "est_ttft_ms": est_ttft_ms,
            "tier_counts": tier_counts,
            "n_prefetched": len(prefetched_ids),
            "n_useful": len(useful),
        }

        if terminated:
            total_accesses = sum(self.episode_hits.values())
            hit_rate = (
                (self.episode_hits["L1"] + self.episode_hits["L2"])
                / total_accesses * 100 if total_accesses > 0 else 0
            )
            prefetch_accuracy = (
                self.episode_useful_prefetches / self.episode_prefetches * 100
                if self.episode_prefetches > 0 else 0
            )
            n_steps = len(self.episode_rewards)
            avg_measured = self.episode_total_measured_ms / max(n_steps, 1)
            avg_baseline = self.episode_total_baseline_ms / max(n_steps, 1)

            # TTFT statistics
            ttft_vals = [s["est_ttft_ms"] for s in self.step_metrics]

            info["episode_summary"] = {
                "total_reward": sum(self.episode_rewards),
                "hit_rate_pct": round(hit_rate, 2),
                "total_prefetches": self.episode_prefetches,
                "useful_prefetches": self.episode_useful_prefetches,
                "prefetch_accuracy_pct": round(prefetch_accuracy, 2),
                "tier_counts": dict(self.episode_hits),
                "tier_latencies_ms": {k: round(v, 3) for k, v in self.tier_latencies.items()},
                "avg_measured_latency_ms": round(avg_measured, 3),
                "avg_baseline_latency_ms": round(avg_baseline, 3),
                "total_measured_latency_ms": round(self.episode_total_measured_ms, 3),
                "total_baseline_latency_ms": round(self.episode_total_baseline_ms, 3),
                "latency_reduction_pct": round(
                    (1 - self.episode_total_measured_ms / max(self.episode_total_baseline_ms, 0.001)) * 100, 2
                ),
                # TTFT metrics
                "avg_ttft_ms": round(float(np.mean(ttft_vals)), 3) if ttft_vals else 0,
                "median_ttft_ms": round(float(np.median(ttft_vals)), 3) if ttft_vals else 0,
                "p95_ttft_ms": round(float(np.percentile(ttft_vals, 95)), 3) if ttft_vals else 0,
                # Per-step detail (for evaluation CSV)
                "step_metrics": self.step_metrics,
            }

        return obs, reward, terminated, False, info

    def _build_obs(self) -> np.ndarray:
        """
        Observation vector:
          [query_embedding(384) | cache_stats(3) | candidate_recency(16)]
        """
        # Query embedding
        if self.current_step < self.n_queries:
            embedding = self.embeddings[self.current_step]
        else:
            embedding = np.zeros(self.embed_dim, dtype=np.float32)

        # Cache stats: [l1_frac, l2_frac, l3_frac]
        l1_frac, l2_frac, l3_frac = self.cache.get_stats()
        cache_stats = np.array([l1_frac, l2_frac, l3_frac], dtype=np.float32)

        # Candidates (chunks in L2/L3 that could be prefetched)
        self.candidate_ids = self.cache.get_candidates(self.max_candidates)

        # Recency score for each candidate
        recent = self.cache.get_recent_accesses(self.history_len * 10)
        recent_set = set(recent[-self.history_len * 5:]) if recent else set()

        recency_scores = np.zeros(self.max_candidates, dtype=np.float32)
        for i, cid in enumerate(self.candidate_ids):
            if cid in recent_set:
                try:
                    pos = len(recent) - 1 - recent[::-1].index(cid)
                    recency_scores[i] = 1.0 - (pos / max(len(recent), 1))
                except ValueError:
                    recency_scores[i] = 0.0

        obs = np.concatenate([embedding, cache_stats, recency_scores])
        return obs.astype(np.float32)
