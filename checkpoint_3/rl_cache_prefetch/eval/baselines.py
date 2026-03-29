"""
Baseline prefetching strategies for comparison.
1. LRU (reactive) — no prefetching at all
2. NoCache — every access is a cold miss
3. OraclePrefetch — perfect knowledge of next query's chunks
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.cache_simulator import CacheSimulator
from env.tier_config import TierConfig


def run_lru_baseline(
    trace_path: str | Path,
    config: Optional[TierConfig] = None,
) -> dict:
    """
    LRU reactive baseline: no prefetching, just reactive cache access.
    Returns per-query and aggregate metrics.
    """
    cfg = config or TierConfig()
    sim = CacheSimulator(cfg)
    df = pd.read_csv(trace_path)

    results = []

    for _, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])

        # Access reactively — no prefetching
        total_ms, tier_counts = sim.access_chunks(chunks)

        total = sum(tier_counts.values())
        hits = tier_counts["L1"] + tier_counts["L2"]

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "tier_counts": tier_counts,
            "hit_rate": hits / total if total > 0 else 0,
            "num_chunks": len(chunks),
        })

    # Aggregate
    all_latencies = [r["access_latency_ms"] for r in results]
    all_hits = sum(r["hit_rate"] * r["num_chunks"] for r in results)
    all_total = sum(r["num_chunks"] for r in results)

    return {
        "strategy": "LRU (reactive)",
        "per_query": results,
        "avg_latency_ms": np.mean(all_latencies),
        "total_latency_ms": sum(all_latencies),
        "hit_rate_pct": (all_hits / all_total * 100) if all_total > 0 else 0,
        "total_prefetches": 0,
    }


def run_no_cache_baseline(
    trace_path: str | Path,
    config: Optional[TierConfig] = None,
) -> dict:
    """
    No-cache baseline: every chunk is a cold miss.
    This represents the worst-case scenario without any caching.
    """
    cfg = config or TierConfig()
    df = pd.read_csv(trace_path)

    results = []

    for _, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])
        # Everything is cold
        total_ms = len(chunks) * cfg.cold_compute_per_chunk_ms

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "tier_counts": {"L1": 0, "L2": 0, "L3": 0, "MISS": len(chunks)},
            "hit_rate": 0.0,
            "num_chunks": len(chunks),
        })

    all_latencies = [r["access_latency_ms"] for r in results]

    return {
        "strategy": "No Cache",
        "per_query": results,
        "avg_latency_ms": np.mean(all_latencies),
        "total_latency_ms": sum(all_latencies),
        "hit_rate_pct": 0.0,
        "total_prefetches": 0,
    }


def run_oracle_baseline(
    trace_path: str | Path,
    config: Optional[TierConfig] = None,
) -> dict:
    """
    Oracle baseline: perfect knowledge of the next query's chunks.
    Prefetches exactly the right chunks before each query.
    This is the theoretical upper bound.
    """
    cfg = config or TierConfig()
    sim = CacheSimulator(cfg)
    df = pd.read_csv(trace_path)

    results = []

    for i, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])

        # Oracle: if we know the next query, prefetch its chunks
        # For the current query, we prefetched based on previous step
        # (so first query is always cold)
        total_ms, tier_counts = sim.access_chunks(chunks)

        total = sum(tier_counts.values())
        hits = tier_counts["L1"] + tier_counts["L2"]

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "tier_counts": tier_counts,
            "hit_rate": hits / total if total > 0 else 0,
            "num_chunks": len(chunks),
        })

        # Oracle: prefetch next query's chunks from L3 → L2
        if i + 1 < len(df):
            next_chunks = json.loads(df.iloc[i + 1]["chunk_ids_needed"])
            for cid in next_chunks:
                sim.prefetch(cid)

    all_latencies = [r["access_latency_ms"] for r in results]
    all_hits = sum(r["hit_rate"] * r["num_chunks"] for r in results)
    all_total = sum(r["num_chunks"] for r in results)

    return {
        "strategy": "Oracle",
        "per_query": results,
        "avg_latency_ms": np.mean(all_latencies),
        "total_latency_ms": sum(all_latencies),
        "hit_rate_pct": (all_hits / all_total * 100) if all_total > 0 else 0,
        "total_prefetches": sum(1 for _ in range(len(df) - 1)),
    }
