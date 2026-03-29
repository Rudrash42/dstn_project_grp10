#!/usr/bin/env python3
"""
Plot evaluation results: hit rate, latency, reward curves, etc.

Usage:
    python eval/plot_results.py
"""

from __future__ import annotations

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def plot_hit_rate_comparison(metrics: dict, save_dir: Path):
    """Bar chart: hit rate per workload × strategy."""
    workloads = list(metrics.keys())
    strategies = ["no_cache", "lru", "rl_agent", "oracle"]
    labels = ["No Cache", "LRU", "RL Agent", "Oracle"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(workloads))
    width = 0.2

    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = []
        for wl in workloads:
            wl_data = metrics[wl]
            if strat in wl_data:
                vals.append(wl_data[strat]["hit_rate_pct"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("Hit Rate (%)", fontsize=12)
    ax.set_title("Cache Hit Rate: RL Agent vs Baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([w.capitalize() for w in workloads])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "hit_rate_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def plot_latency_comparison(metrics: dict, save_dir: Path):
    """Bar chart: average access latency per workload × strategy."""
    workloads = list(metrics.keys())
    strategies = ["no_cache", "lru", "rl_agent", "oracle"]
    labels = ["No Cache", "LRU", "RL Agent", "Oracle"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(workloads))
    width = 0.2

    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = []
        for wl in workloads:
            wl_data = metrics[wl]
            if strat in wl_data:
                vals.append(wl_data[strat]["avg_latency_ms"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("Avg Access Latency (ms)", fontsize=12)
    ax.set_title("Access Latency: RL Agent vs Baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([w.capitalize() for w in workloads])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def plot_reward_curve(save_dir: Path):
    """Training reward curve over episodes."""
    rewards_path = PROJECT_ROOT / "results" / "training_rewards.json"
    if not rewards_path.exists():
        print("  [SKIP] No training_rewards.json found")
        return

    with open(rewards_path) as f:
        rewards = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards, alpha=0.3, color="#3498db", linewidth=0.8, label="Episode reward")

    # Rolling average
    if len(rewards) > 10:
        window = min(20, len(rewards) // 3)
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), rolling,
                color="#e74c3c", linewidth=2, label=f"Rolling avg ({window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("PPO Training Reward Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = save_dir / "reward_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def plot_speedup_bars(metrics: dict, save_dir: Path):
    """Bar chart: RL speedup vs LRU per workload."""
    workloads = []
    speedups = []

    for wl, data in metrics.items():
        if "rl_agent" in data and "speedup_vs_lru" in data["rl_agent"]:
            workloads.append(wl.capitalize())
            speedups.append(data["rl_agent"]["speedup_vs_lru"])

    if not workloads:
        print("  [SKIP] No speedup data available")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2ecc71" if s >= 1 else "#e74c3c" for s in speedups]
    bars = ax.bar(workloads, speedups, color=colors, alpha=0.85, edgecolor="white")

    ax.axhline(y=1.0, color="#95a5a6", linestyle="--", linewidth=1, label="Break-even")
    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("Speedup (RL / LRU)", fontsize=12)
    ax.set_title("RL Agent Speedup vs LRU Baseline", fontsize=14, fontweight="bold")

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{s:.2f}x", ha="center", va="bottom", fontweight="bold")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "speedup_vs_lru.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def plot_prefetch_efficiency(metrics: dict, save_dir: Path):
    """Bar chart: prefetch accuracy per workload."""
    workloads = []
    accuracies = []

    for wl, data in metrics.items():
        if "rl_agent" in data and "prefetch_accuracy_pct" in data["rl_agent"]:
            workloads.append(wl.capitalize())
            accuracies.append(data["rl_agent"]["prefetch_accuracy_pct"])

    if not workloads:
        print("  [SKIP] No prefetch accuracy data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2ecc71" if a >= 50 else "#f39c12" if a >= 25 else "#e74c3c"
              for a in accuracies]
    bars = ax.bar(workloads, accuracies, color=colors, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Workload", fontsize=12)
    ax.set_ylabel("Prefetch Accuracy (%)", fontsize=12)
    ax.set_title("Migration Efficiency: Useful vs Wasted Prefetches",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)

    for bar, a in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f"{a:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "prefetch_efficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def plot_ablation(save_dir: Path):
    """Bar chart: ablation study results."""
    abl_path = PROJECT_ROOT / "results" / "ablation_results.json"
    if not abl_path.exists():
        print("  [SKIP] No ablation_results.json")
        return

    with open(abl_path) as f:
        results = json.load(f)

    names = [r["ablation"] for r in results]
    hit_rates = [r["hit_rate_pct"] for r in results]
    rewards = [r["total_reward"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71"] + ["#e74c3c"] * (len(names) - 1)

    ax1.barh(names, hit_rates, color=colors, alpha=0.85)
    ax1.set_xlabel("Hit Rate (%)")
    ax1.set_title("Hit Rate by Feature Ablation", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(names, rewards, color=colors, alpha=0.85)
    ax2.set_xlabel("Total Reward")
    ax2.set_title("Reward by Feature Ablation", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "ablation_study.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path.name}")


def main():
    print("\n[plot_results] Generating plots...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics_path = PROJECT_ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        plot_hit_rate_comparison(metrics, PLOTS_DIR)
        plot_latency_comparison(metrics, PLOTS_DIR)
        plot_speedup_bars(metrics, PLOTS_DIR)
        plot_prefetch_efficiency(metrics, PLOTS_DIR)
    else:
        print("  [SKIP] No metrics.json — run evaluate.py first")

    plot_reward_curve(PLOTS_DIR)
    plot_ablation(PLOTS_DIR)

    print("[plot_results] Done!\n")


if __name__ == "__main__":
    main()
