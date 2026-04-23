"""
Reward function for the RL cache prefetching agent.
R = α × time_saved_ms − β × (bytes_migrated / 1e6) − γ × unused_count
"""

from __future__ import annotations
from typing import List, Set

from .tier_config import TierConfig


def compute_reward(
    prefetched_chunk_ids: List[int],
    actually_accessed_chunk_ids: Set[int],
    access_latency_ms: float,
    baseline_latency_ms: float,
    config: TierConfig,
) -> float:
    """
    Compute the reward for one step (one query).
    NOTE: Prefetch cost is not included (model assumes it knows next query and prefetch is free).

    Parameters
    ----------
    prefetched_chunk_ids : list[int]
        Chunk IDs the agent chose to prefetch this step.
    actually_accessed_chunk_ids : set[int]
        Chunk IDs that the query actually needed.
    access_latency_ms : float
        Actual time to access all needed chunks (after prefetching).
    baseline_latency_ms : float
        What the access latency *would have been* without any prefetching
        (i.e., all chunks fetched reactively from wherever they sit).
    config : TierConfig
        Tier configuration with reward knobs.

    Returns
    -------
    float
        The scalar reward.
    """
    # Time saved by prefetching (can be negative if prefetch overhead > savings)
    time_saved_ms = baseline_latency_ms - access_latency_ms

    # Bytes migrated = number of prefetched chunks × chunk size
    bytes_migrated = len(prefetched_chunk_ids) * config.chunk_size_bytes
    bytes_migrated_mb = bytes_migrated / 1e6

    # Unused prefetches = prefetched but not actually needed
    prefetched_set = set(prefetched_chunk_ids)
    unused_count = len(prefetched_set - actually_accessed_chunk_ids)

    # Reward: time saved minus migration penalty and unused prefetch penalty
    # (prefetch cost is not included)
    R = (
        config.alpha * time_saved_ms
        - config.beta * bytes_migrated_mb
        - config.gamma_reward * unused_count
    )

    return R


def compute_baseline_latency(
    chunk_ids: List[int],
    chunk_tiers: dict,
    config: TierConfig,
) -> float:
    """
    Compute the latency if no prefetching had happened (reactive baseline).
    Each chunk is accessed from whatever tier it's currently in.

    Parameters
    ----------
    chunk_ids : list[int]
        Chunks needed by the current query.
    chunk_tiers : dict
        Mapping chunk_id → tier ("L1", "L2", "L3", or None for cold miss).
    config : TierConfig
        Tier configuration.

    Returns
    -------
    float
        Total access latency in ms.
    """
    total_ms = 0.0
    for cid in chunk_ids:
        tier = chunk_tiers.get(cid)
        if tier == "L1":
            total_ms += config.l1_hit_latency_ms
        elif tier == "L2":
            total_ms += config.l2_hit_latency_ms
        elif tier == "L3":
            total_ms += config.l3_hit_latency_ms
        else:
            total_ms += config.cold_compute_per_chunk_ms
    return total_ms
