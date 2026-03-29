#!/usr/bin/env python3
"""
Ablation study: remove state features one at a time to measure
their contribution to the agent's performance.

Features ablated:
  1. Query embedding (zero out 384-dim embedding)
  2. Cache stats (zero out 3-dim cache usage)
  3. Candidate recency (zero out 16-dim recency scores)

Usage:
    python eval/ablation.py
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.tier_config import TierConfig
from env.cache_env import CacheEnv
from agent.state_encoder import StateEncoder


def run_ablation(
    trace_path: str | Path,
    model_path: str | Path,
    config: TierConfig,
    embeddings: np.ndarray,
    ablation_name: str,
    zero_range: tuple,
) -> dict:
    """
    Run the RL agent with a portion of the observation zeroed out.
    zero_range: (start_idx, end_idx) to zero out in the obs vector.
    """
    from stable_baselines3 import PPO

    env = CacheEnv(trace_path, config=config, query_embeddings=embeddings)
    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    total_reward = 0.0

    while True:
        # Zero out the ablated features
        ablated_obs = obs.copy()
        ablated_obs[zero_range[0]:zero_range[1]] = 0.0

        action, _ = model.predict(ablated_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    episode = info.get("episode_summary", {})

    return {
        "ablation": ablation_name,
        "total_reward": total_reward,
        "hit_rate_pct": episode.get("hit_rate_pct", 0),
        "prefetch_accuracy_pct": episode.get("prefetch_accuracy_pct", 0),
    }


def run_full_ablation(model_path: str | Path = None):
    """Run all ablation experiments on the RAG workload."""
    tier_cfg = TierConfig.from_yaml(PROJECT_ROOT / "configs" / "ppo_config.yaml")

    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "ppo_final.zip"

    if not Path(model_path).exists():
        print(f"[ablation] Model not found: {model_path}")
        print("  Run train.py first.")
        return

    trace_path = PROJECT_ROOT / "data" / "traces_rag.csv"
    emb_path = PROJECT_ROOT / "data" / "embeddings_rag.npy"
    embeddings = np.load(str(emb_path))

    embed_dim = tier_cfg.embed_dim  # 384
    cache_start = embed_dim          # 384
    cache_end = cache_start + 3      # 387
    recency_start = cache_end        # 387
    recency_end = recency_start + tier_cfg.max_candidate_chunks  # 403

    ablations = [
        ("full (no ablation)", None),
        ("no embedding", (0, embed_dim)),
        ("no cache stats", (cache_start, cache_end)),
        ("no recency scores", (recency_start, recency_end)),
    ]

    print("\n" + "=" * 60)
    print("  ABLATION STUDY (RAG workload)")
    print("=" * 60)

    results = []
    for name, zero_range in ablations:
        if zero_range is None:
            # Full model, no ablation
            r = run_ablation(trace_path, model_path, tier_cfg, embeddings,
                             name, (0, 0))
        else:
            r = run_ablation(trace_path, model_path, tier_cfg, embeddings,
                             name, zero_range)
        results.append(r)
        print(f"  {name:24s} → reward={r['total_reward']:7.2f}  "
              f"hit_rate={r['hit_rate_pct']:5.1f}%  "
              f"prefetch_acc={r['prefetch_accuracy_pct']:5.1f}%")

    # Save
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ablation] Saved → results/ablation_results.json")

    return results


if __name__ == "__main__":
    run_full_ablation()
