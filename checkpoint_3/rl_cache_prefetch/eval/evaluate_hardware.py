#!/usr/bin/env python3
"""
Evaluation on REAL Hardware
============================

Runs the trained RL agent + all baselines on REAL hardware cache,
comparing measured latencies across all workloads.

Outputs:
  results/hardware_metrics.json         — per-workload metric comparisons
  results/hardware_evaluation.csv       — detailed per-query results
  results/plots/hardware_evaluation.png — comparison bar charts

Usage:
    # Evaluate the hardware-trained model:
    python eval/evaluate_hardware.py

    # Evaluate a specific model:
    python eval/evaluate_hardware.py --model models/ppo_hardware_final.zip

    # Custom cache sizes:
    python eval/evaluate_hardware.py --l1-mb 24 --l2-mb 32
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

from env.hardware.hardware_config import HardwareConfig
from env.hardware.hardware_cache_env import HardwareCacheEnv
from agent.state_encoder import StateEncoder
from eval.baselines_hardware import (
    run_lru_baseline_hw,
    run_no_cache_baseline_hw,
    run_oracle_baseline_hw,
)


def evaluate_rl_agent_hw(
    trace_path: str | Path,
    model_path: str | Path,
    config: HardwareConfig,
    embeddings: np.ndarray,
) -> dict:
    """
    Run the trained RL agent on REAL hardware cache.

    The agent makes prefetch decisions, and those decisions cause
    REAL data movement on GPU/CPU/Disk. Latencies are measured.
    """
    from stable_baselines3 import PPO

    # Create hardware environment (quiet mode for evaluation)
    eval_cfg = HardwareConfig(
        chunk_size_tokens=config.chunk_size_tokens,
        kv_bytes_per_token=config.kv_bytes_per_token,
        chunk_size_bytes=config.chunk_size_bytes,
        l1_capacity_mb=config.l1_capacity_mb,
        l2_capacity_mb=config.l2_capacity_mb,
        l3_capacity_mb=config.l3_capacity_mb,
        l3_disk_dir=config.l3_disk_dir,
        l1_hit_latency_ms=config.l1_hit_latency_ms,
        l2_hit_latency_ms=config.l2_hit_latency_ms,
        l3_hit_latency_ms=config.l3_hit_latency_ms,
        cold_compute_per_chunk_ms=config.cold_compute_per_chunk_ms,
        prefetch_l3_to_l2_ms=config.prefetch_l3_to_l2_ms,
        alpha=config.alpha,
        beta=config.beta,
        gamma_reward=config.gamma_reward,
        embed_dim=config.embed_dim,
        max_candidate_chunks=config.max_candidate_chunks,
        history_len=config.history_len,
        force_cpu_mode=config.force_cpu_mode,
        verbose=False,
        enable_operation_log=True,  # Keep logs for analysis
    )

    env = HardwareCacheEnv(trace_path, config=eval_cfg, query_embeddings=embeddings)
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
            "n_prefetched": info.get("n_prefetched", 0),
            "n_useful": info.get("n_useful", 0),
            "tier_counts": info.get("tier_counts", {}),
        })

        if terminated or truncated:
            break

    episode = info.get("episode_summary", {})

    return {
        "strategy": "RL Agent (PPO) [HARDWARE]",
        "per_query": per_query,
        "avg_latency_ms": float(np.mean([r["access_latency_ms"] for r in per_query])),
        "total_latency_ms": float(sum(r["access_latency_ms"] for r in per_query)),
        "total_reward": float(total_reward),
        "hit_rate_pct": float(episode.get("hit_rate_pct", 0)),
        "total_prefetches": episode.get("total_prefetches", 0),
        "useful_prefetches": episode.get("useful_prefetches", 0),
        "prefetch_accuracy_pct": float(episode.get("prefetch_accuracy_pct", 0)),
        "avg_measured_latency_ms": float(episode.get("avg_measured_latency_ms", 0)),
        "avg_baseline_latency_ms": float(episode.get("avg_baseline_latency_ms", 0)),
    }


def run_full_evaluation_hw(
    config: Optional[HardwareConfig] = None,
    model_path: Optional[str] = None,
):
    """
    Run all strategies across all workloads on REAL hardware.
    Saves results and generates comparison plots.
    """
    cfg = config or HardwareConfig.from_yaml(
        PROJECT_ROOT / "configs" / "hardware_config.yaml"
    )

    # Use hardware-trained model by default, fall back to simulated model
    if model_path is None:
        hw_model = PROJECT_ROOT / "models" / "ppo_hardware_final.zip"
        sim_model = PROJECT_ROOT / "models" / "ppo_final.zip"
        if hw_model.exists():
            model_path = hw_model
        elif sim_model.exists():
            model_path = sim_model
            print(f"  ⚠️  No hardware model found. Using simulated model.")
        else:
            model_path = hw_model  # Will show "not found" later

    print("\n" + "=" * 64)
    print("  🔧 HARDWARE EVALUATION")
    print("=" * 64)
    print(f"  Model: {model_path}")
    print(f"  L1: {cfg.l1_capacity_mb} MB | L2: {cfg.l2_capacity_mb} MB")
    print(f"  Hardware: {'CUDA' if not cfg.force_cpu_mode else 'CPU-only'}")

    traces = {
        "prefix": PROJECT_ROOT / "data" / "traces_prefix.csv",
        "rag": PROJECT_ROOT / "data" / "traces_rag.csv",
        "nocontext": PROJECT_ROOT / "data" / "traces_nocontext.csv",
        "multiturn": PROJECT_ROOT / "data" / "traces_multiturn.csv",
    }

    # Load embeddings
    encoder = StateEncoder(embed_dim=cfg.embed_dim)
    embeddings = {}
    for name, path in traces.items():
        if not path.exists():
            print(f"  ⚠️  Skipping embeddings for {name}: trace file not found")
            continue
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
    eval_rows = []  # For detailed CSV export

    # Quiet config for baselines
    quiet_cfg = HardwareConfig(
        chunk_size_tokens=cfg.chunk_size_tokens,
        kv_bytes_per_token=cfg.kv_bytes_per_token,
        chunk_size_bytes=cfg.chunk_size_bytes,
        l1_capacity_mb=cfg.l1_capacity_mb,
        l2_capacity_mb=cfg.l2_capacity_mb,
        l3_capacity_mb=cfg.l3_capacity_mb,
        l3_disk_dir=cfg.l3_disk_dir + "_eval",
        l1_hit_latency_ms=cfg.l1_hit_latency_ms,
        l2_hit_latency_ms=cfg.l2_hit_latency_ms,
        l3_hit_latency_ms=cfg.l3_hit_latency_ms,
        cold_compute_per_chunk_ms=cfg.cold_compute_per_chunk_ms,
        prefetch_l3_to_l2_ms=cfg.prefetch_l3_to_l2_ms,
        force_cpu_mode=cfg.force_cpu_mode,
        verbose=False,
        enable_operation_log=False,
    )

    for wl_name, trace_path in traces.items():
        if not trace_path.exists():
            print(f"\n  ⚠️  Skipping {wl_name}: trace not found")
            continue
        if wl_name not in embeddings:
            print(f"\n  ⚠️  Skipping {wl_name}: embeddings unavailable")
            continue

        print(f"\n{'─' * 55}")
        print(f"  Workload: {wl_name}")
        print(f"{'─' * 55}")

        wl_results = {}

        # ── LRU Baseline (hardware) ──
        print(f"  Running LRU baseline on real hardware...")
        lru = run_lru_baseline_hw(trace_path, quiet_cfg)
        wl_results["lru_hw"] = {
            "avg_latency_ms": lru["avg_latency_ms"],
            "hit_rate_pct": lru["hit_rate_pct"],
            "total_latency_ms": lru["total_latency_ms"],
        }
        print(f"  LRU [HW]:    avg_lat={lru['avg_latency_ms']:>8.2f}ms  "
              f"hit_rate={lru['hit_rate_pct']:5.1f}%")

        # ── No-Cache Baseline (hardware) ──
        print(f"  Running No-Cache baseline on real hardware...")
        nc = run_no_cache_baseline_hw(trace_path, quiet_cfg)
        wl_results["no_cache_hw"] = {
            "avg_latency_ms": nc["avg_latency_ms"],
            "hit_rate_pct": nc["hit_rate_pct"],
            "total_latency_ms": nc["total_latency_ms"],
        }
        print(f"  NoCache[HW]: avg_lat={nc['avg_latency_ms']:>8.2f}ms  "
              f"hit_rate={nc['hit_rate_pct']:5.1f}%")

        # ── Oracle Baseline (hardware) ──
        print(f"  Running Oracle baseline on real hardware...")
        oracle = run_oracle_baseline_hw(trace_path, quiet_cfg)
        wl_results["oracle_hw"] = {
            "avg_latency_ms": oracle["avg_latency_ms"],
            "hit_rate_pct": oracle["hit_rate_pct"],
            "total_latency_ms": oracle["total_latency_ms"],
        }
        print(f"  Oracle[HW]:  avg_lat={oracle['avg_latency_ms']:>8.2f}ms  "
              f"hit_rate={oracle['hit_rate_pct']:5.1f}%")

        # ── RL Agent (hardware) ──
        if Path(model_path).exists():
            print(f"  Running RL agent on real hardware...")
            rl = evaluate_rl_agent_hw(
                trace_path, model_path, cfg, embeddings[wl_name]
            )
            wl_results["rl_agent_hw"] = {
                "avg_latency_ms": rl["avg_latency_ms"],
                "hit_rate_pct": rl["hit_rate_pct"],
                "total_latency_ms": rl["total_latency_ms"],
                "total_prefetches": rl["total_prefetches"],
                "useful_prefetches": rl.get("useful_prefetches", 0),
                "prefetch_accuracy_pct": rl.get("prefetch_accuracy_pct", 0),
                "total_reward": rl["total_reward"],
                "avg_measured_latency_ms": rl.get("avg_measured_latency_ms", 0),
            }
            print(f"  RL [HW]:     avg_lat={rl['avg_latency_ms']:>8.2f}ms  "
                  f"hit_rate={rl['hit_rate_pct']:5.1f}%  "
                  f"pf_acc={rl.get('prefetch_accuracy_pct', 0):5.1f}%")

            # Speedup vs LRU
            if lru["avg_latency_ms"] > 0:
                speedup = lru["avg_latency_ms"] / rl["avg_latency_ms"]
                wl_results["rl_agent_hw"]["speedup_vs_lru"] = round(speedup, 3)
                print(f"  RL vs LRU speedup: {speedup:.3f}x")

            # Collect per-query data for CSV export
            for pq in rl["per_query"]:
                eval_rows.append({
                    "workload": wl_name,
                    "strategy": "RL_Agent",
                    "query_id": pq["query_id"],
                    "access_latency_ms": pq["access_latency_ms"],
                    "baseline_latency_ms": pq["baseline_latency_ms"],
                    "reward": pq["reward"],
                    "n_prefetched": pq["n_prefetched"],
                    "n_useful": pq["n_useful"],
                })
        else:
            print(f"  RL:     [SKIP] Model not found at {model_path}")

        all_results[wl_name] = wl_results

    # ── Save results ──────────────────────────────────────────
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    metrics_path = results_dir / "hardware_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  💾 Metrics saved → {metrics_path.name}")

    # Detailed CSV
    if eval_rows:
        eval_df = pd.DataFrame(eval_rows)
        eval_csv = results_dir / "hardware_evaluation.csv"
        eval_df.to_csv(eval_csv, index=False)
        print(f"  💾 Detailed results → {eval_csv.name}")

    # ── Generate comparison plots ─────────────────────────────
    _plot_evaluation(all_results, results_dir)

    # ── Print summary table ───────────────────────────────────
    _print_summary_table(all_results)

    return all_results


def _plot_evaluation(all_results: dict, results_dir: Path):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not installed. Skipping plots.")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    workloads = list(all_results.keys())
    if not workloads:
        return

    strategies = ["no_cache_hw", "lru_hw", "oracle_hw", "rl_agent_hw"]
    labels = ["No Cache", "LRU", "Oracle", "RL Agent"]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Average latency comparison
    x = np.arange(len(workloads))
    width = 0.18

    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = []
        for wl in workloads:
            wl_data = all_results.get(wl, {})
            strat_data = wl_data.get(strat, {})
            vals.append(strat_data.get("avg_latency_ms", 0))
        if any(v > 0 for v in vals):
            ax1.bar(x + i * width, vals, width, label=label,
                   color=color, alpha=0.8)

    ax1.set_xlabel("Workload")
    ax1.set_ylabel("Avg Latency (ms)")
    ax1.set_title("Average Access Latency [HARDWARE]")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(workloads)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Hit rate comparison
    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = []
        for wl in workloads:
            wl_data = all_results.get(wl, {})
            strat_data = wl_data.get(strat, {})
            vals.append(strat_data.get("hit_rate_pct", 0))
        if any(v > 0 for v in vals):
            ax2.bar(x + i * width, vals, width, label=label,
                   color=color, alpha=0.8)

    ax2.set_xlabel("Workload")
    ax2.set_ylabel("Hit Rate (%)")
    ax2.set_title("Cache Hit Rate [HARDWARE]")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(workloads)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "hardware_evaluation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Evaluation plot → {path.name}")


def _print_summary_table(all_results: dict):
    """Print a formatted summary table."""
    print(f"\n{'=' * 72}")
    print(f"  📊 HARDWARE EVALUATION SUMMARY")
    print(f"{'=' * 72}")

    header = (f"  {'Workload':<12} {'Strategy':<18} "
              f"{'Avg Lat (ms)':<14} {'Hit Rate':<10} {'Speedup':<10}")
    print(header)
    print("  " + "─" * 66)

    for wl_name, wl_data in all_results.items():
        lru_lat = wl_data.get("lru_hw", {}).get("avg_latency_ms", 1)

        for strat_key, strat_label in [
            ("no_cache_hw", "No Cache"),
            ("lru_hw", "LRU"),
            ("oracle_hw", "Oracle"),
            ("rl_agent_hw", "RL Agent"),
        ]:
            if strat_key in wl_data:
                d = wl_data[strat_key]
                lat = d.get("avg_latency_ms", 0)
                hr = d.get("hit_rate_pct", 0)
                sp = f"{lru_lat / lat:.2f}x" if lat > 0 else "N/A"
                print(f"  {wl_name:<12} {strat_label:<18} "
                      f"{lat:<14.2f} {hr:<10.1f} {sp:<10}")

        print("  " + "─" * 66)

    print(f"{'=' * 72}\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL agent on REAL hardware cache"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to hardware_config.yaml")
    parser.add_argument("--l1-mb", type=float, default=None,
                        help="Override L1 capacity (MB)")
    parser.add_argument("--l2-mb", type=float, default=None,
                        help="Override L2 capacity (MB)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU-only mode")
    args = parser.parse_args()

    # Load config
    yaml_path = args.config or (PROJECT_ROOT / "configs" / "hardware_config.yaml")
    cfg = HardwareConfig.from_yaml(yaml_path)

    if args.l1_mb is not None:
        cfg.l1_capacity_mb = args.l1_mb
    if args.l2_mb is not None:
        cfg.l2_capacity_mb = args.l2_mb
    if args.cpu_only:
        cfg.force_cpu_mode = True

    run_full_evaluation_hw(config=cfg, model_path=args.model)


if __name__ == "__main__":
    main()
