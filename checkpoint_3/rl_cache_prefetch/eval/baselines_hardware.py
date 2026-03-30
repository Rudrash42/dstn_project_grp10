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

    # Clean up
    cache.reset()

    return {
        "strategy": "LRU (reactive) [HARDWARE]",
        "per_query": results,
        "avg_latency_ms": float(np.mean(all_latencies)),
        "total_latency_ms": float(sum(all_latencies)),
        "hit_rate_pct": float((all_hits / all_total * 100) if all_total > 0 else 0),
        "total_prefetches": 0,
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
            "tier_counts": tier_counts,
            "hit_rate": 0.0,
            "num_chunks": len(chunks),
        })

    all_latencies = [r["access_latency_ms"] for r in results]

    return {
        "strategy": "No Cache [HARDWARE]",
        "per_query": results,
        "avg_latency_ms": float(np.mean(all_latencies)),
        "total_latency_ms": float(sum(all_latencies)),
        "hit_rate_pct": 0.0,
        "total_prefetches": 0,
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

        total = sum(tier_counts.values())
        hits = tier_counts["L1"] + tier_counts["L2"]

        results.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "tier_counts": tier_counts,
            "hit_rate": hits / total if total > 0 else 0,
            "num_chunks": len(chunks),
        })

        # Oracle: prefetch NEXT query's chunks (perfect knowledge)
        if i + 1 < len(df):
            next_chunks = json.loads(df.iloc[i + 1]["chunk_ids_needed"])
            for cid in next_chunks:
                cache.prefetch(cid)  # Real disk → CPU transfer

    all_latencies = [r["access_latency_ms"] for r in results]
    all_hits = sum(r["hit_rate"] * r["num_chunks"] for r in results)
    all_total = sum(r["num_chunks"] for r in results)

    # Clean up
    cache.reset()

    return {
        "strategy": "Oracle [HARDWARE]",
        "per_query": results,
        "avg_latency_ms": float(np.mean(all_latencies)),
        "total_latency_ms": float(sum(all_latencies)),
        "hit_rate_pct": float((all_hits / all_total * 100) if all_total > 0 else 0),
        "total_prefetches": sum(1 for _ in range(len(df) - 1)),
    }
