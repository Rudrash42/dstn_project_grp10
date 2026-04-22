"""
Baseline strategies using REAL Hardware Cache
==============================================

Same baselines as baselines.py (LRU, No-Cache, Oracle), but running
on HardwareCache with MEASURED latencies instead of simulated ones.

This lets you compare the RL agent against baselines using the
actual performance of your GPU/CPU/Disk hardware.

Usage:
    # Called by evaluate_hardware.py, or use directly:
    from eval.baselines_hardware import run_lru_baseline_hw
    result = run_lru_baseline_hw("data/traces_rag.csv", config)
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.hardware.hardware_cache import HardwareCache
from env.hardware.hardware_config import HardwareConfig


def _aggregate_hit_rates(results: list[dict]) -> dict:
    """Compute L1/local/overall hit-rate variants from per-query tier counts."""
    total_accesses = 0
    l1_hits = 0
    local_hits = 0
    overall_hits = 0

    for row in results:
        tc = row.get("tier_counts", {})
        q_total = int(sum(tc.values()))
        row_l1 = int(tc.get("L1", 0))
        row_l2 = int(tc.get("L2", 0))
        row_l3 = int(tc.get("L3", 0))
        total_accesses += q_total
        l1_hits += row_l1
        local_hits += row_l1 + row_l2
        overall_hits += row_l1 + row_l2 + row_l3

    if total_accesses <= 0:
        return {
            "hit_rate_pct": 0.0,
            "l1_hit_rate_pct": 0.0,
            "local_hit_rate_pct": 0.0,
            "overall_hit_rate_pct": 0.0,
        }

    l1_rate = l1_hits / total_accesses * 100.0
    local_rate = local_hits / total_accesses * 100.0
    overall_rate = overall_hits / total_accesses * 100.0
    return {
        "hit_rate_pct": float(l1_rate),
        "l1_hit_rate_pct": float(l1_rate),
        "local_hit_rate_pct": float(local_rate),
        "overall_hit_rate_pct": float(overall_rate),
    }


def run_lru_baseline_hw(
    trace_path: str | Path,
    config: Optional[HardwareConfig] = None,
) -> dict:
    """
    LRU reactive baseline on REAL hardware: no prefetching,
    just reactive cache access with measured latencies.

    Every chunk access goes through real GPU/CPU/Disk transfers.

    Returns per-query and aggregate metrics with MEASURED times.
    """
    cfg = config or HardwareConfig(verbose=False, enable_operation_log=False)
    cache = HardwareCache(cfg)
    df = pd.read_csv(trace_path)

    results = []

    for _, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])

        # Access reactively — no prefetching, just LRU cache
        total_ms, tier_counts = cache.access_chunks(chunks)

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "end_to_end_latency_ms": total_ms,
            "tier_counts": tier_counts,
            "num_chunks": len(chunks),
        })

    # Aggregate
    all_latencies = [r["access_latency_ms"] for r in results]
    hit_rates = _aggregate_hit_rates(results)

    # Clean up
    cache.reset()

    return {
        "strategy": "LRU (reactive) [HARDWARE]",
        "per_query": results,
        "avg_latency_ms": float(np.mean(all_latencies)),
        "avg_end_to_end_latency_ms": float(np.mean(all_latencies)),
        "total_latency_ms": float(sum(all_latencies)),
        "total_end_to_end_latency_ms": float(sum(all_latencies)),
        **hit_rates,
        "total_prefetches": 0,
        "total_prefetch_cost_ms": 0.0,
    }


def run_no_cache_baseline_hw(
    trace_path: str | Path,
    config: Optional[HardwareConfig] = None,
) -> dict:
    """
    No-cache baseline on REAL hardware: every chunk is a cold miss.

    Each chunk triggers a new tensor creation on GPU (simulates
    having to recompute KV cache from scratch every time).
    """
    cfg = config or HardwareConfig(verbose=False, enable_operation_log=False)

    # For no-cache, we create a fresh HardwareCache per query
    # so nothing is ever warm
    df = pd.read_csv(trace_path)
    results = []

    for _, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])

        # Create a fresh cache for EVERY query (nothing cached)
        cache = HardwareCache(HardwareConfig(
            l1_capacity_mb=cfg.l1_capacity_mb,
            l2_capacity_mb=cfg.l2_capacity_mb,
            l3_capacity_mb=cfg.l3_capacity_mb,
            chunk_size_bytes=cfg.chunk_size_bytes,
            l3_disk_dir=cfg.l3_disk_dir,
            force_cpu_mode=cfg.force_cpu_mode,
            verbose=False,
            enable_operation_log=False,
            cuda_warmup_iterations=0,  # Skip warmup for speed
        ))

        total_ms, tier_counts = cache.access_chunks(chunks)
        cache.reset()

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "end_to_end_latency_ms": total_ms,
            "tier_counts": tier_counts,
            "num_chunks": len(chunks),
        })

    all_latencies = [r["access_latency_ms"] for r in results]

    return {
        "strategy": "No Cache [HARDWARE]",
        "per_query": results,
        "avg_latency_ms": float(np.mean(all_latencies)),
        "avg_end_to_end_latency_ms": float(np.mean(all_latencies)),
        "total_latency_ms": float(sum(all_latencies)),
        "total_end_to_end_latency_ms": float(sum(all_latencies)),
        "hit_rate_pct": 0.0,
        "l1_hit_rate_pct": 0.0,
        "local_hit_rate_pct": 0.0,
        "overall_hit_rate_pct": 0.0,
        "total_prefetches": 0,
        "total_prefetch_cost_ms": 0.0,
    }


def run_oracle_baseline_hw(
    trace_path: str | Path,
    config: Optional[HardwareConfig] = None,
) -> dict:
    """
    Oracle baseline on REAL hardware: perfect future knowledge.

    After processing each query, the oracle looks ahead at the
    NEXT query's chunks and prefetches them from L3 → L2.

    This represents the theoretical BEST a prefetching agent could do.
    The RL agent should try to approach (but never exceed) oracle.
    """
    cfg = config or HardwareConfig(verbose=False, enable_operation_log=False)
    cache = HardwareCache(cfg)
    df = pd.read_csv(trace_path)

    results = []

    for i, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])

        # Access this query's chunks (with real hardware timing)
        total_ms, tier_counts = cache.access_chunks(chunks)

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "end_to_end_latency_ms": total_ms,
            "tier_counts": tier_counts,
            "num_chunks": len(chunks),
        })

        # Oracle: prefetch NEXT query's chunks (perfect knowledge)
        if i + 1 < len(df):
            next_chunks = json.loads(df.iloc[i + 1]["chunk_ids_needed"])
            # Prevent blind thrashing by limiting prefetch to L2 capacity
            l2_cap = getattr(cfg, "l2_capacity_chunks", int(getattr(cfg, "l2_capacity_mb", 12.0) * 1024 * 1024 / getattr(cfg, "chunk_size_bytes", 3*1024*1024)))
            prefetched = 0
            for cid in next_chunks:
                if prefetched >= l2_cap:
                    break
                if hasattr(cache, "chunk_in_cache"):
                    loc = cache.chunk_in_cache(cid)
                    in_l1_l2 = loc in ["L1", "L2"]
                    in_l3 = loc == "L3"
                else:
                    in_l1_l2 = cid in cache.l1 or cid in cache.l2
                    in_l3 = cid in cache.l3

                if in_l1_l2:
                    prefetched += 1
                elif in_l3:
                    cost = cache.prefetch(cid)  # Real disk → CPU transfer
                    prefetched += 1
    all_latencies = [r["access_latency_ms"] for r in results]
    hit_rates = _aggregate_hit_rates(results)

    # Clean up
    cache.reset()

    return {
        "strategy": "Oracle [HARDWARE]",
        "per_query": results,
        "avg_latency_ms": float(np.mean(all_latencies)),
        "avg_end_to_end_latency_ms": float(np.mean(all_latencies)),
        "total_latency_ms": float(sum(all_latencies)),
        "total_end_to_end_latency_ms": float(sum(all_latencies)),
        **hit_rates,
        "total_prefetches": sum(1 for _ in range(len(df) - 1)),
        "total_prefetch_cost_ms": 0.0,
    }
