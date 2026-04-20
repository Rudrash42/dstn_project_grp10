#!/usr/bin/env python3
"""
Evaluation script: runs the trained RL agent + baselines across all workloads.
Outputs results/metrics.json and prints a comparison table.

Usage:
    python eval/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.tier_config import TierConfig
from env.cache_env import CacheEnv
from agent.state_encoder import StateEncoder
from eval.baselines import run_lru_baseline, run_no_cache_baseline, run_oracle_baseline


REQUIRED_TRACE_COLUMNS = {
    "query_id",
    "query_text",
    "embedding_text",
    "chunk_ids_needed",
    "runtime_chunk_ids",
    "chunk_event_source",
    "runtime_event_count",
    "tier_transition_event",
    "cache_file_count_before",
    "cache_file_count_after",
    "cache_disk_mb_before",
    "cache_disk_mb_after",
}
VALID_CHUNK_EVENT_SOURCES = {
    "direct_runtime",
    "lmcache_store_runtime_snapshot",
    "lmcache_store_coldpass",
}


def validate_trace_schema(trace_path: Path):
    """Fail fast when traces are stale or missing runtime provenance columns."""
    df = pd.read_csv(trace_path)
    missing = [c for c in REQUIRED_TRACE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Trace {trace_path} is stale/incompatible. Missing columns: {missing}. "
            "Regenerate traces with: python data/generate_traces.py"
        )

    sources = set(str(v).strip() for v in df["chunk_event_source"].dropna().unique())
    invalid = {s for s in sources if s and s not in VALID_CHUNK_EVENT_SOURCES}
    if invalid:
        raise RuntimeError(
            f"Trace {trace_path} has non-runtime chunk rows: {sorted(invalid)}. "
            "Regenerate traces with strict runtime chunk capture."
        )


def evaluate_rl_agent(
    trace_path: str | Path,
    model_path: str | Path,
    config: TierConfig,
    embeddings: np.ndarray,
) -> dict:
    """
    Run the trained RL agent on a trace and collect metrics.
    """
    from stable_baselines3 import PPO

    env = CacheEnv(trace_path, config=config, query_embeddings=embeddings)
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
            "prefetch_cost_ms": info.get("prefetch_cost_ms", 0),
            "total_ttft_ms": info.get("access_latency_ms", 0) + info.get("prefetch_cost_ms", 0),
            "baseline_latency_ms": info.get("baseline_latency_ms", 0),
            "n_prefetched": info.get("n_prefetched", 0),
            "n_useful": info.get("n_useful", 0),
            "tier_counts": info.get("tier_counts", {}),
        })

        if terminated or truncated:
            break

    episode = info.get("episode_summary", {})

    return {
        "strategy": "RL Agent (PPO)",
        "per_query": per_query,
        "avg_ttft_ms": np.mean([r["total_ttft_ms"] for r in per_query]),
        "total_ttft_ms": sum(r["total_ttft_ms"] for r in per_query),
        "avg_latency_ms": np.mean([r["total_ttft_ms"] for r in per_query]),
        "total_latency_ms": sum(r["total_ttft_ms"] for r in per_query),
        "total_reward": total_reward,
        "hit_rate_pct": episode.get("hit_rate_pct", 0),
        "cache_hit_rate_pct": episode.get("cache_hit_rate_pct", episode.get("hit_rate_pct", 0)),
        "local_hit_rate_pct": episode.get("local_hit_rate_pct", episode.get("hit_rate_pct", 0)),
        "total_prefetches": episode.get("total_prefetches", 0),
        "useful_prefetches": episode.get("useful_prefetches", 0),
        "prefetch_accuracy_pct": episode.get("prefetch_accuracy_pct", 0),
    }


def run_full_evaluation(model_path: Optional[str] = None, include_interleaved: bool = False):
    """
    Run all strategies across all workloads and save results.
    """
    tier_cfg = TierConfig.from_yaml(PROJECT_ROOT / "configs" / "ppo_config.yaml")

    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "ppo_final.zip"

    traces = {
        "prefix": PROJECT_ROOT / "data" / "traces_prefix.csv",
        "rag": PROJECT_ROOT / "data" / "traces_rag.csv",
        "nocontext": PROJECT_ROOT / "data" / "traces_nocontext.csv",
        "multiturn": PROJECT_ROOT / "data" / "traces_multiturn.csv",
    }

    if include_interleaved:
        traces["interleaved"] = PROJECT_ROOT / "data" / "traces_interleaved.csv"
        print("[evaluate] Including synthetic interleaved workload.")

    # Load or compute embeddings
    encoder = StateEncoder(embed_dim=tier_cfg.embed_dim)
    embeddings = {}
    for name, path in traces.items():
        if not path.exists():
            print(f"[evaluate] Skipping {name}: trace file not found at {path}")
            continue
        validate_trace_schema(path)
        emb_path = PROJECT_ROOT / "data" / f"embeddings_{name}.npy"
        if emb_path.exists():
            embeddings[name] = StateEncoder.load_embeddings(emb_path)
        else:
            df = pd.read_csv(path)
            text_col = "embedding_text" if "embedding_text" in df.columns else "query_text"
            embeddings[name] = encoder.encode_and_save(
                df[text_col].tolist(), emb_path
            )

    # Results container
    all_results = {}

    for wl_name, trace_path in traces.items():
        if wl_name not in embeddings:
            continue
        print(f"\n{'─' * 50}")
        print(f"  Workload: {wl_name}")
        print(f"{'─' * 50}")

        wl_results = {}

        # LRU Baseline
        lru = run_lru_baseline(trace_path, tier_cfg)
        lru_avg_ttft = lru.get("avg_ttft_ms", lru["avg_latency_ms"])
        lru_total_ttft = lru.get("total_ttft_ms", lru["total_latency_ms"])
        wl_results["lru"] = {
            "avg_ttft_ms": lru_avg_ttft,
            "avg_latency_ms": lru_avg_ttft,
            "hit_rate_pct": lru["hit_rate_pct"],
                        "cache_hit_rate_pct": lru.get("cache_hit_rate_pct", lru["hit_rate_pct"]),
                        "local_hit_rate_pct": lru.get("local_hit_rate_pct", lru["hit_rate_pct"]),
            "total_ttft_ms": lru_total_ttft,
            "total_latency_ms": lru_total_ttft,
        }
        print(f"  LRU:    avg_ttft={lru_avg_ttft:7.2f}ms  "
                            f"hit_rate={lru['hit_rate_pct']:5.1f}%  "
                            f"local={lru.get('local_hit_rate_pct', lru['hit_rate_pct']):5.1f}%")

        # No-Cache Baseline
        nc = run_no_cache_baseline(trace_path, tier_cfg)
        nc_avg_ttft = nc.get("avg_ttft_ms", nc["avg_latency_ms"])
        nc_total_ttft = nc.get("total_ttft_ms", nc["total_latency_ms"])
        wl_results["no_cache"] = {
            "avg_ttft_ms": nc_avg_ttft,
            "avg_latency_ms": nc_avg_ttft,
            "hit_rate_pct": nc["hit_rate_pct"],
                        "cache_hit_rate_pct": nc.get("cache_hit_rate_pct", nc["hit_rate_pct"]),
                        "local_hit_rate_pct": nc.get("local_hit_rate_pct", nc["hit_rate_pct"]),
            "total_ttft_ms": nc_total_ttft,
            "total_latency_ms": nc_total_ttft,
        }
        print(f"  NoCache: avg_ttft={nc_avg_ttft:7.2f}ms  "
                            f"hit_rate={nc['hit_rate_pct']:5.1f}%  "
                            f"local={nc.get('local_hit_rate_pct', nc['hit_rate_pct']):5.1f}%")

        # Oracle Baseline
        oracle = run_oracle_baseline(trace_path, tier_cfg)
        oracle_avg_ttft = oracle.get("avg_ttft_ms", oracle["avg_latency_ms"])
        oracle_total_ttft = oracle.get("total_ttft_ms", oracle["total_latency_ms"])
        wl_results["oracle"] = {
            "avg_ttft_ms": oracle_avg_ttft,
            "avg_latency_ms": oracle_avg_ttft,
            "hit_rate_pct": oracle["hit_rate_pct"],
                        "cache_hit_rate_pct": oracle.get("cache_hit_rate_pct", oracle["hit_rate_pct"]),
                        "local_hit_rate_pct": oracle.get("local_hit_rate_pct", oracle["hit_rate_pct"]),
            "total_ttft_ms": oracle_total_ttft,
            "total_latency_ms": oracle_total_ttft,
        }
        print(f"  Oracle: avg_ttft={oracle_avg_ttft:7.2f}ms  "
                            f"hit_rate={oracle['hit_rate_pct']:5.1f}%  "
                            f"local={oracle.get('local_hit_rate_pct', oracle['hit_rate_pct']):5.1f}%")

        # RL Agent
        if Path(model_path).exists():
            rl = evaluate_rl_agent(
                trace_path, model_path, tier_cfg, embeddings[wl_name]
            )
            rl_avg_ttft = rl.get("avg_ttft_ms", rl["avg_latency_ms"])
            rl_total_ttft = rl.get("total_ttft_ms", rl["total_latency_ms"])
            wl_results["rl_agent"] = {
                "avg_ttft_ms": rl_avg_ttft,
                "avg_latency_ms": rl_avg_ttft,
                "hit_rate_pct": rl["hit_rate_pct"],
                                "cache_hit_rate_pct": rl.get("cache_hit_rate_pct", rl["hit_rate_pct"]),
                                "local_hit_rate_pct": rl.get("local_hit_rate_pct", rl["hit_rate_pct"]),
                "total_ttft_ms": rl_total_ttft,
                "total_latency_ms": rl_total_ttft,
                "total_prefetches": rl["total_prefetches"],
                "useful_prefetches": rl.get("useful_prefetches", 0),
                "prefetch_accuracy_pct": rl.get("prefetch_accuracy_pct", 0),
                "total_reward": rl["total_reward"],
            }
            print(f"  RL:     avg_ttft={rl_avg_ttft:7.2f}ms  "
                                    f"hit_rate={rl['hit_rate_pct']:5.1f}%  "
                                    f"local={rl.get('local_hit_rate_pct', rl['hit_rate_pct']):5.1f}%  "
                  f"prefetch_acc={rl.get('prefetch_accuracy_pct', 0):5.1f}%")

            # Speedup
            if lru_avg_ttft > 0:
                speedup = lru_avg_ttft / rl_avg_ttft
                wl_results["rl_agent"]["speedup_vs_lru"] = round(speedup, 3)
                print(f"  RL vs LRU speedup: {speedup:.3f}x")
        else:
            print(f"  RL:     [SKIP] Model not found at {model_path}")

        all_results[wl_name] = wl_results

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[evaluate] Results saved → {metrics_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL policy and baselines.")
    parser.add_argument("--model", type=str, default=None, help="Path to PPO model zip.")
    parser.add_argument(
        "--include-interleaved",
        action="store_true",
        help="Include synthetic interleaved workload in primary evaluation.",
    )
    args = parser.parse_args()

    run_full_evaluation(model_path=args.model, include_interleaved=args.include_interleaved)
