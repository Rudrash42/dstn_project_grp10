#!/usr/bin/env python3
"""
Evaluation: RL Agent + Baselines on Real Hardware
===================================================

Runs 4 strategies across all 4 workloads on REAL hardware:
  1. No Cache   — fresh cache per query (everything is a cold miss)
  2. LRU        — reactive caching, no prefetching
  3. Oracle     — perfect future knowledge (upper bound)
  4. RL Agent   — our trained PPO model

Outputs:
  results/evaluation_metrics.json          — per-workload, per-strategy metrics
  results/evaluation_detailed.csv          — per-query RL agent results
  results/evaluation_summary.csv           — summary table as CSV
  results/evaluation_ttft.csv              — TTFT comparison across strategies
  results/evaluation_tier_breakdown.csv    — tier breakdown per strategy per workload
  results/plots/evaluation_latency.png     — latency comparison bar chart
  results/plots/evaluation_hitrate.png     — hit rate comparison bar chart
  results/plots/evaluation_speedup.png     — speedup vs LRU baseline
  results/plots/evaluation_ttft.png        — TTFT comparison
  results/plots/evaluation_tier_breakdown.png — stacked tier breakdown
  results/plots/evaluation_per_query.png   — per-query latency progression
  results/plots/evaluation_cache_occ.png   — cache occupancy over queries

Usage:
    python evaluate.py                        # Evaluate with default model
    python evaluate.py --model path/to.zip    # Evaluate specific model
    python evaluate.py --l1-mb 24             # Custom cache sizes
    python evaluate.py --cpu-only             # CPU-only mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path=None):
    import yaml
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_embeddings(embed_dim=384):
    embeddings = {}
    for name in ["prefix", "rag", "nocontext", "multiturn"]:
        path = DATA_DIR / f"embeddings_{name}.npy"
        if path.exists():
            embeddings[name] = np.load(str(path))
        else:
            trace = DATA_DIR / f"traces_{name}.csv"
            if trace.exists():
                n = len(pd.read_csv(trace))
                embeddings[name] = np.zeros((n, embed_dim), dtype=np.float32)
    return embeddings


def _make_cache(cfg, suffix=""):
    """Create a HardwareCache from config."""
    from cache import HardwareCache
    return HardwareCache(
        chunk_size_bytes=cfg.get("chunk_size_bytes", 3_145_728),
        l1_capacity_mb=cfg.get("l1_capacity_mb", 48.0),
        l2_capacity_mb=cfg.get("l2_capacity_mb", 51.2),
        l3_capacity_mb=cfg.get("l3_capacity_mb", 5120.0),
        l3_disk_dir=cfg.get("l3_disk_dir", "./data/cache_store") + f"_eval_{suffix}",
        force_cpu_mode=cfg.get("force_cpu_mode", False),
        cuda_warmup_iterations=cfg.get("cuda_warmup_iterations", 5),
        cold_compute_ms=cfg.get("cold_compute_per_chunk_ms", 30.8),
    )


# ═══════════════════════════════════════════════════════════════
# BASELINE STRATEGIES
# ═══════════════════════════════════════════════════════════════

def run_lru_baseline(trace_path, cfg) -> dict:
    """LRU reactive baseline: no prefetching, just reactive cache access."""
    cache = _make_cache(cfg, "lru")
    df = pd.read_csv(trace_path)
    per_query = []

    for _, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])
        total_ms, tier_counts, details = cache.access_chunks_detailed(chunks)
        total = sum(tier_counts.values())
        hits = tier_counts["L1"] + tier_counts["L2"]

        # Estimate TTFT from first chunk access
        est_ttft = details[0]["latency_ms"] if details else total_ms / max(len(chunks), 1)

        # Occupancy snapshot
        cache.snapshot_occupancy(row["query_id"])

        per_query.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "est_ttft_ms": est_ttft,
            "tier_counts": tier_counts,
            "hit_rate": hits / total if total > 0 else 0,
            "n_chunks": len(chunks),
            "per_chunk": details,
        })

    occupancy = cache.get_occupancy_log()
    cache.reset()
    cache.cleanup_disk()

    all_lats = [r["access_latency_ms"] for r in per_query]
    all_ttft = [r["est_ttft_ms"] for r in per_query]
    all_hits = sum(r["hit_rate"] for r in per_query)

    # Aggregate tier counts
    agg_tiers = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
    agg_tier_ms = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "MISS": 0.0}
    for r in per_query:
        for t in agg_tiers:
            agg_tiers[t] += r["tier_counts"].get(t, 0)
        for d in r.get("per_chunk", []):
            agg_tier_ms[d["tier"]] += d["latency_ms"]

    total_chunks = sum(agg_tiers.values()) or 1

    return {
        "strategy": "LRU",
        "avg_latency_ms": float(np.mean(all_lats)),
        "total_latency_ms": float(sum(all_lats)),
        "hit_rate_pct": round(all_hits / len(per_query) * 100, 2) if per_query else 0,
        "total_prefetches": 0,
        "avg_ttft_ms": round(float(np.mean(all_ttft)), 3),
        "median_ttft_ms": round(float(np.median(all_ttft)), 3),
        "p95_ttft_ms": round(float(np.percentile(all_ttft, 95)), 3),
        "tier_counts": agg_tiers,
        "tier_pcts": {t: round(c / total_chunks * 100, 1) for t, c in agg_tiers.items()},
        "tier_latencies_ms": {t: round(v, 3) for t, v in agg_tier_ms.items()},
        "occupancy_log": occupancy,
        "per_query": per_query,
    }


def run_no_cache_baseline(trace_path, cfg) -> dict:
    """No-cache baseline: fresh cache per query, everything is a cold miss."""
    df = pd.read_csv(trace_path)
    per_query = []

    for _, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])
        cache = _make_cache(cfg, "nc")
        cache._cuda_warmup = lambda n: None  # Skip warmup for speed
        total_ms, tier_counts, details = cache.access_chunks_detailed(chunks)

        est_ttft = details[0]["latency_ms"] if details else total_ms / max(len(chunks), 1)

        per_query.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms,
            "est_ttft_ms": est_ttft,
            "tier_counts": tier_counts,
            "hit_rate": 0.0,
            "n_chunks": len(chunks),
        })
        cache.reset()
        cache.cleanup_disk()

    all_lats = [r["access_latency_ms"] for r in per_query]
    all_ttft = [r["est_ttft_ms"] for r in per_query]

    # Aggregate tiers (everything should be MISS)
    agg_tiers = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
    for r in per_query:
        for t in agg_tiers:
            agg_tiers[t] += r["tier_counts"].get(t, 0)
    total_chunks = sum(agg_tiers.values()) or 1

    return {
        "strategy": "No Cache",
        "avg_latency_ms": float(np.mean(all_lats)),
        "total_latency_ms": float(sum(all_lats)),
        "hit_rate_pct": 0.0,
        "total_prefetches": 0,
        "avg_ttft_ms": round(float(np.mean(all_ttft)), 3),
        "median_ttft_ms": round(float(np.median(all_ttft)), 3),
        "p95_ttft_ms": round(float(np.percentile(all_ttft, 95)), 3),
        "tier_counts": agg_tiers,
        "tier_pcts": {t: round(c / total_chunks * 100, 1) for t, c in agg_tiers.items()},
        "tier_latencies_ms": {"L1": 0, "L2": 0, "L3": 0, "MISS": float(sum(all_lats))},
        "occupancy_log": [],
        "per_query": per_query,
    }


def run_oracle_baseline(trace_path, cfg) -> dict:
    """Oracle baseline: perfect future knowledge prefetching."""
    cache = _make_cache(cfg, "oracle")
    df = pd.read_csv(trace_path)
    per_query = []

    for i, row in df.iterrows():
        chunks = json.loads(row["chunk_ids_needed"])
        total_ms, tier_counts, details = cache.access_chunks_detailed(chunks)
        total = sum(tier_counts.values())
        hits = tier_counts["L1"] + tier_counts["L2"]

        est_ttft = details[0]["latency_ms"] if details else total_ms / max(len(chunks), 1)

        prefetch_cost = 0.0
        # Oracle: prefetch NEXT query's chunks
        if i + 1 < len(df):
            next_chunks = json.loads(df.iloc[i + 1]["chunk_ids_needed"])
            for cid in next_chunks:
                loc = cache.chunk_in_cache(cid)
                if loc == "L3":
                    cost = cache.prefetch(cid)
                    prefetch_cost += cost

        cache.snapshot_occupancy(row["query_id"])

        per_query.append({
            "query_id": row["query_id"],
            "access_latency_ms": total_ms + prefetch_cost,
            "est_ttft_ms": est_ttft,
            "prefetch_cost_ms": prefetch_cost,
            "tier_counts": tier_counts,
            "hit_rate": hits / total if total > 0 else 0,
            "n_chunks": len(chunks),
            "per_chunk": details,
        })

    occupancy = cache.get_occupancy_log()
    cache.reset()
    cache.cleanup_disk()

    all_lats = [r["access_latency_ms"] for r in per_query]
    all_ttft = [r["est_ttft_ms"] for r in per_query]
    all_hits = sum(r["hit_rate"] for r in per_query)

    agg_tiers = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
    agg_tier_ms = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "MISS": 0.0}
    for r in per_query:
        for t in agg_tiers:
            agg_tiers[t] += r["tier_counts"].get(t, 0)
        for d in r.get("per_chunk", []):
            agg_tier_ms[d["tier"]] += d["latency_ms"]
    total_chunks = sum(agg_tiers.values()) or 1

    return {
        "strategy": "Oracle",
        "avg_latency_ms": float(np.mean(all_lats)),
        "total_latency_ms": float(sum(all_lats)),
        "hit_rate_pct": round(all_hits / len(per_query) * 100, 2) if per_query else 0,
        "total_prefetches": len(df) - 1,
        "avg_ttft_ms": round(float(np.mean(all_ttft)), 3),
        "median_ttft_ms": round(float(np.median(all_ttft)), 3),
        "p95_ttft_ms": round(float(np.percentile(all_ttft, 95)), 3),
        "tier_counts": agg_tiers,
        "tier_pcts": {t: round(c / total_chunks * 100, 1) for t, c in agg_tiers.items()},
        "tier_latencies_ms": {t: round(v, 3) for t, v in agg_tier_ms.items()},
        "occupancy_log": occupancy,
        "per_query": per_query,
    }


# ═══════════════════════════════════════════════════════════════
# RL AGENT EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_rl_agent(trace_path, model_path, cfg, embeddings) -> dict:
    """Run the trained RL agent on real hardware cache."""
    from stable_baselines3 import PPO
    from environment import CachePrefetchEnv

    hw_cache = _make_cache(cfg, "rl")
    env = CachePrefetchEnv(
        trace_path=trace_path,
        cache=hw_cache,
        query_embeddings=embeddings,
        embed_dim=cfg.get("embed_dim", 384),
        max_candidates=cfg.get("max_candidates", 16),
        history_len=cfg.get("history_len", 5),
        chunk_size_bytes=cfg.get("chunk_size_bytes", 3_145_728),
        alpha=cfg.get("alpha", 1.0),
        beta=cfg.get("beta", 0.01),
        gamma_reward=cfg.get("gamma_reward", 0.1),
        l1_hit_latency_ms=cfg.get("l1_hit_latency_ms", 0.1),
        l2_hit_latency_ms=cfg.get("l2_hit_latency_ms", 0.24),
        l3_hit_latency_ms=cfg.get("l3_hit_latency_ms", 6.0),
        cold_compute_per_chunk_ms=cfg.get("cold_compute_per_chunk_ms", 30.8),
    )

    model = PPO.load(str(model_path))
    obs, _ = env.reset()
    total_reward = 0.0
    per_query = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        per_query.append({
            "query_id": info.get("step", 0),
            "reward": reward,
            "access_latency_ms": info.get("access_latency_ms", 0),
            "baseline_latency_ms": info.get("baseline_latency_ms", 0),
            "prefetch_cost_ms": info.get("prefetch_cost_ms", 0),
            "est_ttft_ms": info.get("est_ttft_ms", 0),
            "n_prefetched": info.get("n_prefetched", 0),
            "n_useful": info.get("n_useful", 0),
            "tier_counts": info.get("tier_counts", {}),
        })

        if terminated or truncated:
            break

    occupancy = hw_cache.get_occupancy_log()
    hw_cache.reset()
    hw_cache.cleanup_disk()

    episode = info.get("episode_summary", {})
    all_lats = [r["access_latency_ms"] + r.get("prefetch_cost_ms", 0) for r in per_query]
    all_ttft = [r["est_ttft_ms"] for r in per_query]

    agg_tiers = episode.get("tier_counts", {"L1": 0, "L2": 0, "L3": 0, "MISS": 0})
    total_chunks = sum(agg_tiers.values()) or 1

    return {
        "strategy": "RL Agent",
        "avg_latency_ms": float(np.mean(all_lats)),
        "total_latency_ms": float(sum(all_lats)),
        "total_reward": float(total_reward),
        "hit_rate_pct": float(episode.get("hit_rate_pct", 0)),
        "total_prefetches": episode.get("total_prefetches", 0),
        "useful_prefetches": episode.get("useful_prefetches", 0),
        "prefetch_accuracy_pct": float(episode.get("prefetch_accuracy_pct", 0)),
        "avg_ttft_ms": round(float(np.mean(all_ttft)), 3) if all_ttft else 0,
        "median_ttft_ms": round(float(np.median(all_ttft)), 3) if all_ttft else 0,
        "p95_ttft_ms": round(float(np.percentile(all_ttft, 95)), 3) if all_ttft else 0,
        "latency_reduction_pct": float(episode.get("latency_reduction_pct", 0)),
        "tier_counts": agg_tiers,
        "tier_pcts": {t: round(c / total_chunks * 100, 1) for t, c in agg_tiers.items()},
        "tier_latencies_ms": episode.get("tier_latencies_ms", {}),
        "occupancy_log": occupancy,
        "per_query": per_query,
        "step_metrics": episode.get("step_metrics", []),
    }


# ═══════════════════════════════════════════════════════════════
# FULL EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_full_evaluation(cfg, model_path=None):
    """Run all strategies across all workloads. Returns all_results dict."""

    if model_path is None:
        model_path = MODELS_DIR / "ppo_final.zip"

    print("\n" + "=" * 64)
    print("  EVALUATION (Real Hardware)")
    print("=" * 64)
    print(f"  Model: {model_path}")
    print(f"  L1: {cfg.get('l1_capacity_mb', 48.0)} MB | "
          f"L2: {cfg.get('l2_capacity_mb', 51.2)} MB | "
          f"Cold compute: {cfg.get('cold_compute_per_chunk_ms', 30.8)} ms/chunk")

    traces = {
        "prefix":    DATA_DIR / "traces_prefix.csv",
        "rag":       DATA_DIR / "traces_rag.csv",
        "nocontext": DATA_DIR / "traces_nocontext.csv",
        "multiturn": DATA_DIR / "traces_multiturn.csv",
    }

    embeddings = load_embeddings(cfg.get("embed_dim", 384))
    all_results = {}
    eval_rows = []       # Per-query RL details
    ttft_rows = []       # TTFT comparison
    tier_rows = []       # Tier breakdown
    summary_rows = []    # Summary table

    for wl_name, trace_path in traces.items():
        if not trace_path.exists():
            print(f"\n  Skipping {wl_name}: trace not found")
            continue

        print(f"\n{'─' * 55}")
        print(f"  Workload: {wl_name}")
        print(f"{'─' * 55}")

        wl_results = {}

        # ── Run baselines ─────────────────────────────────────────

        print(f"  Running LRU baseline...")
        t0 = time.time()
        lru = run_lru_baseline(trace_path, cfg)
        print(f"    LRU:      avg={lru['avg_latency_ms']:>8.2f}ms  "
              f"hit={lru['hit_rate_pct']:5.1f}%  "
              f"TTFT={lru['avg_ttft_ms']:.2f}ms  [{time.time()-t0:.1f}s]")

        print(f"  Running No-Cache baseline...")
        t0 = time.time()
        nc = run_no_cache_baseline(trace_path, cfg)
        print(f"    NoCache:  avg={nc['avg_latency_ms']:>8.2f}ms  "
              f"hit={nc['hit_rate_pct']:5.1f}%  "
              f"TTFT={nc['avg_ttft_ms']:.2f}ms  [{time.time()-t0:.1f}s]")

        print(f"  Running Oracle baseline...")
        t0 = time.time()
        oracle = run_oracle_baseline(trace_path, cfg)
        print(f"    Oracle:   avg={oracle['avg_latency_ms']:>8.2f}ms  "
              f"hit={oracle['hit_rate_pct']:5.1f}%  "
              f"TTFT={oracle['avg_ttft_ms']:.2f}ms  [{time.time()-t0:.1f}s]")

        # Store baseline results (without per_query for JSON)
        for strat_name, strat_data in [("no_cache", nc), ("lru", lru), ("oracle", oracle)]:
            wl_results[strat_name] = {
                k: v for k, v in strat_data.items()
                if k not in ("per_query", "occupancy_log", "per_chunk")
            }

        # ── RL Agent ──────────────────────────────────────────────

        if Path(model_path).exists() and wl_name in embeddings:
            print(f"  Running RL Agent...")
            t0 = time.time()
            rl = run_rl_agent(trace_path, model_path, cfg, embeddings[wl_name])

            # Speedup vs LRU
            speedup_lru = lru["avg_latency_ms"] / rl["avg_latency_ms"] if rl["avg_latency_ms"] > 0 else 0
            speedup_nc = nc["avg_latency_ms"] / rl["avg_latency_ms"] if rl["avg_latency_ms"] > 0 else 0

            wl_results["rl_agent"] = {
                k: v for k, v in rl.items()
                if k not in ("per_query", "occupancy_log", "step_metrics")
            }
            wl_results["rl_agent"]["speedup_vs_lru"] = round(speedup_lru, 3)
            wl_results["rl_agent"]["speedup_vs_nocache"] = round(speedup_nc, 3)

            print(f"    RL Agent: avg={rl['avg_latency_ms']:>8.2f}ms  "
                  f"hit={rl['hit_rate_pct']:5.1f}%  "
                  f"TTFT={rl['avg_ttft_ms']:.2f}ms  "
                  f"speedup={speedup_lru:.2f}x  [{time.time()-t0:.1f}s]")

            # Collect per-query detail for CSV
            for pq in rl["per_query"]:
                eval_rows.append({
                    "workload": wl_name,
                    "query_id": pq["query_id"],
                    "access_latency_ms": pq["access_latency_ms"],
                    "baseline_latency_ms": pq["baseline_latency_ms"],
                    "prefetch_cost_ms": pq.get("prefetch_cost_ms", 0),
                    "est_ttft_ms": pq.get("est_ttft_ms", 0),
                    "reward": pq["reward"],
                    "n_prefetched": pq["n_prefetched"],
                    "n_useful": pq["n_useful"],
                })
        else:
            rl = None
            print(f"    RL Agent: SKIPPED (model not found)")

        # ── TTFT comparison rows ──────────────────────────────────

        for strat_name, strat_data in [("No Cache", nc), ("LRU", lru),
                                        ("Oracle", oracle)]:
            ttft_rows.append({
                "workload": wl_name,
                "strategy": strat_name,
                "avg_ttft_ms": strat_data["avg_ttft_ms"],
                "median_ttft_ms": strat_data["median_ttft_ms"],
                "p95_ttft_ms": strat_data["p95_ttft_ms"],
                "avg_latency_ms": strat_data["avg_latency_ms"],
            })
        if rl:
            ttft_rows.append({
                "workload": wl_name,
                "strategy": "RL Agent",
                "avg_ttft_ms": rl["avg_ttft_ms"],
                "median_ttft_ms": rl["median_ttft_ms"],
                "p95_ttft_ms": rl["p95_ttft_ms"],
                "avg_latency_ms": rl["avg_latency_ms"],
            })

        # ── Tier breakdown rows ───────────────────────────────────

        for strat_name, strat_data in [("No Cache", nc), ("LRU", lru),
                                        ("Oracle", oracle)]:
            pcts = strat_data.get("tier_pcts", {})
            tier_rows.append({
                "workload": wl_name, "strategy": strat_name,
                "L1_pct": pcts.get("L1", 0), "L2_pct": pcts.get("L2", 0),
                "L3_pct": pcts.get("L3", 0), "MISS_pct": pcts.get("MISS", 0),
            })
        if rl:
            pcts = rl.get("tier_pcts", {})
            tier_rows.append({
                "workload": wl_name, "strategy": "RL Agent",
                "L1_pct": pcts.get("L1", 0), "L2_pct": pcts.get("L2", 0),
                "L3_pct": pcts.get("L3", 0), "MISS_pct": pcts.get("MISS", 0),
            })

        # ── Summary rows ─────────────────────────────────────────

        lru_lat = lru["avg_latency_ms"] or 1
        for strat_name, strat_data in [("No Cache", nc), ("LRU", lru),
                                        ("Oracle", oracle)]:
            summary_rows.append({
                "workload": wl_name,
                "strategy": strat_name,
                "avg_latency_ms": round(strat_data["avg_latency_ms"], 2),
                "hit_rate_pct": strat_data["hit_rate_pct"],
                "avg_ttft_ms": strat_data["avg_ttft_ms"],
                "speedup_vs_lru": round(lru_lat / strat_data["avg_latency_ms"], 3) if strat_data["avg_latency_ms"] > 0 else 0,
            })
        if rl:
            summary_rows.append({
                "workload": wl_name,
                "strategy": "RL Agent",
                "avg_latency_ms": round(rl["avg_latency_ms"], 2),
                "hit_rate_pct": rl["hit_rate_pct"],
                "avg_ttft_ms": rl["avg_ttft_ms"],
                "speedup_vs_lru": round(lru_lat / rl["avg_latency_ms"], 3) if rl["avg_latency_ms"] > 0 else 0,
                "prefetch_accuracy_pct": rl.get("prefetch_accuracy_pct", 0),
                "latency_reduction_pct": rl.get("latency_reduction_pct", 0),
            })

        all_results[wl_name] = wl_results

    # ── Save ALL results ──────────────────────────────────────────

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Full metrics JSON
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Metrics → evaluation_metrics.json")

    # 2. Per-query RL details CSV
    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(
            RESULTS_DIR / "evaluation_detailed.csv", index=False)
        print(f"  Details → evaluation_detailed.csv")

    # 3. TTFT comparison CSV
    if ttft_rows:
        pd.DataFrame(ttft_rows).to_csv(
            RESULTS_DIR / "evaluation_ttft.csv", index=False)
        print(f"  TTFT    → evaluation_ttft.csv")

    # 4. Tier breakdown CSV
    if tier_rows:
        pd.DataFrame(tier_rows).to_csv(
            RESULTS_DIR / "evaluation_tier_breakdown.csv", index=False)
        print(f"  Tiers   → evaluation_tier_breakdown.csv")

    # 5. Summary CSV
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            RESULTS_DIR / "evaluation_summary.csv", index=False)
        print(f"  Summary → evaluation_summary.csv")

    # ── Plots ─────────────────────────────────────────────────────

    _plot_evaluation(all_results, ttft_rows, tier_rows, eval_rows)
    _print_summary_table(all_results)

    return all_results


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def _plot_evaluation(all_results, ttft_rows, tier_rows, eval_rows):
    """Generate comprehensive evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available. Skipping plots.")
        return

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    workloads = list(all_results.keys())
    if not workloads:
        return

    strategies = ["no_cache", "lru", "oracle", "rl_agent"]
    labels = ["No Cache", "LRU", "Oracle", "RL Agent"]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"]

    x = np.arange(len(workloads))
    width = 0.18

    # ── 1. Latency comparison ──

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = [all_results.get(wl, {}).get(strat, {}).get("avg_latency_ms", 0) for wl in workloads]
        if any(v > 0 for v in vals):
            bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("Avg Latency (ms)", fontsize=12)
    ax.set_title("Average Access Latency by Strategy (Real Hardware)", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([w.capitalize() for w in workloads])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "evaluation_latency.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Hit rate comparison ──

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = [all_results.get(wl, {}).get(strat, {}).get("hit_rate_pct", 0) for wl in workloads]
        if any(v > 0 for v in vals):
            ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("Hit Rate (%)", fontsize=12)
    ax.set_title("Cache Hit Rate by Strategy (Real Hardware)", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([w.capitalize() for w in workloads])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "evaluation_hitrate.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Speedup chart ──

    fig, ax = plt.subplots(figsize=(10, 5))
    speedups = []
    wl_labels = []
    for wl in workloads:
        rl_data = all_results.get(wl, {}).get("rl_agent", {})
        sp = rl_data.get("speedup_vs_lru", 0)
        if sp > 0:
            speedups.append(sp)
            wl_labels.append(wl.capitalize())
    if speedups:
        bar_colors = ["#e74c3c" if s > 1 else "#3498db" for s in speedups]
        bars = ax.bar(wl_labels, speedups, color=bar_colors, alpha=0.85)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="LRU baseline")
        for bar, v in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{v:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel("Speedup vs LRU", fontsize=12)
        ax.set_title("RL Agent Speedup over LRU Baseline", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "evaluation_speedup.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. TTFT comparison ──

    if ttft_rows:
        ttft_df = pd.DataFrame(ttft_rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        strat_names = ["No Cache", "LRU", "Oracle", "RL Agent"]
        for i, (sname, color) in enumerate(zip(strat_names, colors)):
            subset = ttft_df[ttft_df["strategy"] == sname]
            if not subset.empty:
                vals = [subset[subset["workload"] == wl]["avg_ttft_ms"].values[0]
                        if not subset[subset["workload"] == wl].empty else 0
                        for wl in workloads]
                ax.bar(x + i * width, vals, width, label=sname, color=color, alpha=0.85)
        ax.set_xlabel("Workload", fontsize=12)
        ax.set_ylabel("Avg TTFT (ms)", fontsize=12)
        ax.set_title("Estimated Time To First Token by Strategy", fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([w.capitalize() for w in workloads])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(plots_dir / "evaluation_ttft.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── 5. Tier breakdown (stacked bars) ──

    if tier_rows:
        tier_df = pd.DataFrame(tier_rows)
        fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 5), sharey=True)
        if len(workloads) == 1:
            axes = [axes]
        tier_colors = {"L1_pct": "#e74c3c", "L2_pct": "#3498db",
                       "L3_pct": "#2ecc71", "MISS_pct": "#95a5a6"}
        tier_labels = {"L1_pct": "L1 (GPU)", "L2_pct": "L2 (CPU)",
                       "L3_pct": "L3 (Disk)", "MISS_pct": "MISS"}

        for ax_idx, wl in enumerate(workloads):
            ax = axes[ax_idx]
            wl_tier = tier_df[tier_df["workload"] == wl]
            strats = wl_tier["strategy"].tolist()
            bottom = np.zeros(len(strats))
            for tier_key in ["L1_pct", "L2_pct", "L3_pct", "MISS_pct"]:
                vals = wl_tier[tier_key].values
                ax.bar(strats, vals, bottom=bottom, label=tier_labels[tier_key],
                       color=tier_colors[tier_key], alpha=0.8)
                bottom += vals
            ax.set_title(wl.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylabel("Percentage (%)" if ax_idx == 0 else "")
            ax.tick_params(axis='x', rotation=30)
            if ax_idx == len(workloads) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.suptitle("Cache Tier Breakdown by Strategy", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "evaluation_tier_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── 6. Per-query latency progression (RL agent) ──

    if eval_rows:
        eval_df = pd.DataFrame(eval_rows)
        wl_groups = eval_df.groupby("workload")
        n_wl = len(wl_groups)
        if n_wl > 0:
            fig, axes = plt.subplots(1, n_wl, figsize=(6 * n_wl, 4), sharey=True)
            if n_wl == 1:
                axes = [axes]
            for ax_idx, (wl_name, wl_df) in enumerate(wl_groups):
                ax = axes[ax_idx]
                ax.plot(wl_df["query_id"], wl_df["access_latency_ms"],
                        'o-', color="#e74c3c", alpha=0.7, markersize=4, label="RL Agent")
                ax.plot(wl_df["query_id"], wl_df["baseline_latency_ms"],
                        's--', color="#95a5a6", alpha=0.5, markersize=3, label="Baseline (no prefetch)")
                ax.set_xlabel("Query ID")
                ax.set_ylabel("Latency (ms)" if ax_idx == 0 else "")
                ax.set_title(wl_name.capitalize())
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            plt.suptitle("Per-Query Latency: RL Agent vs Baseline", fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(plots_dir / "evaluation_per_query.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"  Plots saved → {plots_dir}/")


def _print_summary_table(all_results):
    """Print a formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 80}")

    print(f"  {'Workload':<12} {'Strategy':<12} "
          f"{'Avg Lat(ms)':<13} {'Hit Rate':<10} {'Avg TTFT':<10} {'vs LRU':<10}")
    print("  " + "─" * 68)

    for wl_name, wl_data in all_results.items():
        lru_lat = wl_data.get("lru", {}).get("avg_latency_ms", 1)

        for strat_key, strat_label in [
            ("no_cache", "No Cache"),
            ("lru", "LRU"),
            ("oracle", "Oracle"),
            ("rl_agent", "RL Agent"),
        ]:
            if strat_key in wl_data:
                d = wl_data[strat_key]
                lat = d.get("avg_latency_ms", 0)
                hr = d.get("hit_rate_pct", 0)
                ttft = d.get("avg_ttft_ms", 0)
                sp = f"{lru_lat / lat:.2f}x" if lat > 0 else "N/A"
                print(f"  {wl_name:<12} {strat_label:<12} "
                      f"{lat:<13.2f} {hr:<10.1f} {ttft:<10.2f} {sp:<10}")

        print("  " + "─" * 68)

    print(f"{'=' * 80}\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL cache prefetching agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--l1-mb", type=float, default=None)
    parser.add_argument("--l2-mb", type=float, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.l1_mb is not None:
        cfg["l1_capacity_mb"] = args.l1_mb
    if args.l2_mb is not None:
        cfg["l2_capacity_mb"] = args.l2_mb
    if args.cpu_only:
        cfg["force_cpu_mode"] = True

    model_path = args.model or str(MODELS_DIR / "ppo_final.zip")
    run_full_evaluation(cfg, model_path)


if __name__ == "__main__":
    main()
