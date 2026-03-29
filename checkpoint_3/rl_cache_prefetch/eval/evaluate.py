#!/usr/bin/env python3
"""
Evaluation script: runs the trained RL agent + baselines across all workloads.
Outputs results/metrics.json and prints a comparison table.

Usage:
    python eval/evaluate.py
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

from env.tier_config import TierConfig
from env.cache_env import CacheEnv
from agent.state_encoder import StateEncoder
from eval.baselines import run_lru_baseline, run_no_cache_baseline, run_oracle_baseline


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
        "avg_latency_ms": np.mean([r["access_latency_ms"] for r in per_query]),
        "total_latency_ms": sum(r["access_latency_ms"] for r in per_query),
        "total_reward": total_reward,
        "hit_rate_pct": episode.get("hit_rate_pct", 0),
        "total_prefetches": episode.get("total_prefetches", 0),
        "useful_prefetches": episode.get("useful_prefetches", 0),
        "prefetch_accuracy_pct": episode.get("prefetch_accuracy_pct", 0),
    }


def run_full_evaluation(model_path: Optional[str] = None):
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

    # Load or compute embeddings
    encoder = StateEncoder(embed_dim=tier_cfg.embed_dim)
    embeddings = {}
    for name, path in traces.items():
        emb_path = PROJECT_ROOT / "data" / f"embeddings_{name}.npy"
        if emb_path.exists():
            embeddings[name] = StateEncoder.load_embeddings(emb_path)
        else:
            df = pd.read_csv(path)
            embeddings[name] = encoder.encode_and_save(
                df["query_text"].tolist(), emb_path
            )

    # Results container
    all_results = {}

    for wl_name, trace_path in traces.items():
        print(f"\n{'─' * 50}")
        print(f"  Workload: {wl_name}")
        print(f"{'─' * 50}")

        wl_results = {}

        # LRU Baseline
        lru = run_lru_baseline(trace_path, tier_cfg)
        wl_results["lru"] = {
            "avg_latency_ms": lru["avg_latency_ms"],
            "hit_rate_pct": lru["hit_rate_pct"],
            "total_latency_ms": lru["total_latency_ms"],
        }
        print(f"  LRU:    avg_lat={lru['avg_latency_ms']:7.2f}ms  "
              f"hit_rate={lru['hit_rate_pct']:5.1f}%")

        # No-Cache Baseline
        nc = run_no_cache_baseline(trace_path, tier_cfg)
        wl_results["no_cache"] = {
            "avg_latency_ms": nc["avg_latency_ms"],
            "hit_rate_pct": nc["hit_rate_pct"],
            "total_latency_ms": nc["total_latency_ms"],
        }
        print(f"  NoCache: avg_lat={nc['avg_latency_ms']:7.2f}ms  "
              f"hit_rate={nc['hit_rate_pct']:5.1f}%")

        # Oracle Baseline
        oracle = run_oracle_baseline(trace_path, tier_cfg)
        wl_results["oracle"] = {
            "avg_latency_ms": oracle["avg_latency_ms"],
            "hit_rate_pct": oracle["hit_rate_pct"],
            "total_latency_ms": oracle["total_latency_ms"],
        }
        print(f"  Oracle: avg_lat={oracle['avg_latency_ms']:7.2f}ms  "
              f"hit_rate={oracle['hit_rate_pct']:5.1f}%")

        # RL Agent
        if Path(model_path).exists():
            rl = evaluate_rl_agent(
                trace_path, model_path, tier_cfg, embeddings[wl_name]
            )
            wl_results["rl_agent"] = {
                "avg_latency_ms": rl["avg_latency_ms"],
                "hit_rate_pct": rl["hit_rate_pct"],
                "total_latency_ms": rl["total_latency_ms"],
                "total_prefetches": rl["total_prefetches"],
                "useful_prefetches": rl.get("useful_prefetches", 0),
                "prefetch_accuracy_pct": rl.get("prefetch_accuracy_pct", 0),
                "total_reward": rl["total_reward"],
            }
            print(f"  RL:     avg_lat={rl['avg_latency_ms']:7.2f}ms  "
                  f"hit_rate={rl['hit_rate_pct']:5.1f}%  "
                  f"prefetch_acc={rl.get('prefetch_accuracy_pct', 0):5.1f}%")

            # Speedup
            if lru["avg_latency_ms"] > 0:
                speedup = lru["avg_latency_ms"] / rl["avg_latency_ms"]
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
    run_full_evaluation()
