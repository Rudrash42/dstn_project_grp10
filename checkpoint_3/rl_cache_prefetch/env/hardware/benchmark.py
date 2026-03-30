#!/usr/bin/env python3
"""
Hardware Cache Benchmark & Validation
======================================

Run this script to:
  1. Test that the hardware cache works correctly
  2. Measure REAL latencies for each tier (L1/L2/L3/MISS)
  3. Compare measured latencies vs simulated (hardcoded) latencies
  4. Run a sample episode through HardwareCacheEnv
  5. Generate CSV results and plots

Usage:
    # From the rl_cache_prefetch/ directory:
    python -m env.hardware.benchmark

    # Or directly:
    python env/hardware/benchmark.py

    # With custom config:
    python env/hardware/benchmark.py --l1-mb 24 --l2-mb 32 --verbose

Output:
    results/hardware_benchmark_latencies.csv
    results/hardware_benchmark_operations.csv
    results/hardware_benchmark_episode.csv
    results/plots/hardware_latency_comparison.png
    results/plots/hardware_tier_distribution.png
    results/plots/hardware_episode_timeline.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.hardware.hardware_config import HardwareConfig
from env.hardware.hardware_cache import HardwareCache
from env.hardware.hardware_cache_env import HardwareCacheEnv


# ═══════════════════════════════════════════════════════════════
# TEST 1: Tier Latency Measurement
# ═══════════════════════════════════════════════════════════════

def benchmark_tier_latencies(config: HardwareConfig, n_trials: int = 20) -> pd.DataFrame:
    """
    Measure REAL latencies for each cache tier.

    Creates chunks, puts them in specific tiers, then accesses them
    while measuring wall-clock time.

    Returns a DataFrame with columns:
        trial, tier, operation, latency_ms
    """
    print("\n" + "=" * 64)
    print("  TEST 1: Tier Latency Measurement")
    print("=" * 64)
    print(f"  Running {n_trials} trials per tier...")

    # Use small config for testing
    test_cfg = HardwareConfig(
        l1_capacity_mb=config.l1_capacity_mb,
        l2_capacity_mb=config.l2_capacity_mb,
        l3_capacity_mb=config.l3_capacity_mb,
        chunk_size_bytes=config.chunk_size_bytes,
        l3_disk_dir=str(PROJECT_ROOT / "results" / "benchmark_cache_store"),
        verbose=False,
        enable_operation_log=False,
    )
    cache = HardwareCache(test_cfg)

    rows = []

    # ── Measure L1 (GPU) hit latency ──
    print(f"\n  📏 Measuring L1 (GPU VRAM) latency...")
    for trial in range(n_trials):
        cache.reset()
        cid = 1000 + trial
        cache.insert_chunks([cid])  # Put chunk in L1

        # Now access it — should be an L1 hit
        tier, latency = cache.access_chunk(cid)
        assert tier == "L1", f"Expected L1 but got {tier}"
        rows.append({
            "trial": trial, "tier": "L1", "operation": "access",
            "latency_ms": latency,
        })

    l1_avg = np.mean([r["latency_ms"] for r in rows if r["tier"] == "L1"])
    print(f"     L1 avg: {l1_avg:.4f} ms  (range: "
          f"{min(r['latency_ms'] for r in rows if r['tier'] == 'L1'):.4f} - "
          f"{max(r['latency_ms'] for r in rows if r['tier'] == 'L1'):.4f})")

    # ── Measure L2 (CPU) hit latency ──
    print(f"\n  📏 Measuring L2 (CPU RAM) hit latency...")
    for trial in range(n_trials):
        cache.reset()
        cid = 2000 + trial

        # Insert enough chunks to push our target to L2
        # First fill L1, then the next insert will evict LRU to L2
        n_l1 = test_cfg.l1_capacity_chunks
        fill_ids = list(range(5000, 5000 + n_l1 + 1))  # One more than L1 capacity
        for fid in fill_ids:
            cache.insert_chunks([fid])

        # The first inserted chunk should now be in L2
        evicted_cid = fill_ids[0]
        tier_check = cache.chunk_in_cache(evicted_cid)

        if tier_check == "L2":
            tier, latency = cache.access_chunk(evicted_cid)
            rows.append({
                "trial": trial, "tier": "L2", "operation": "access",
                "latency_ms": latency,
            })
        else:
            print(f"     ⚠️  Trial {trial}: Expected L2 but chunk is in {tier_check}")

    l2_rows = [r for r in rows if r["tier"] == "L2"]
    if l2_rows:
        l2_avg = np.mean([r["latency_ms"] for r in l2_rows])
        print(f"     L2 avg: {l2_avg:.4f} ms  (range: "
              f"{min(r['latency_ms'] for r in l2_rows):.4f} - "
              f"{max(r['latency_ms'] for r in l2_rows):.4f})")

    # ── Measure L3 (Disk) hit latency ──
    print(f"\n  📏 Measuring L3 (NVMe Disk) hit latency...")
    for trial in range(n_trials):
        cache.reset()

        # Fill L1 and L2 to push chunks to L3
        n_fill = test_cfg.l1_capacity_chunks + test_cfg.l2_capacity_chunks + 2
        fill_ids = list(range(8000, 8000 + n_fill))
        for fid in fill_ids:
            cache.insert_chunks([fid])

        # The first inserted chunk should be in L3
        target_cid = fill_ids[0]
        tier_check = cache.chunk_in_cache(target_cid)

        if tier_check == "L3":
            tier, latency = cache.access_chunk(target_cid)
            rows.append({
                "trial": trial, "tier": "L3", "operation": "access",
                "latency_ms": latency,
            })
        else:
            print(f"     ⚠️  Trial {trial}: Expected L3 but chunk is in {tier_check}")

    l3_rows = [r for r in rows if r["tier"] == "L3"]
    if l3_rows:
        l3_avg = np.mean([r["latency_ms"] for r in l3_rows])
        print(f"     L3 avg: {l3_avg:.4f} ms  (range: "
              f"{min(r['latency_ms'] for r in l3_rows):.4f} - "
              f"{max(r['latency_ms'] for r in l3_rows):.4f})")

    # ── Measure MISS latency ──
    print(f"\n  📏 Measuring MISS (cold compute) latency...")
    for trial in range(n_trials):
        cache.reset()
        cid = 9000 + trial

        # Access a chunk that doesn't exist
        tier, latency = cache.access_chunk(cid)
        assert tier == "MISS", f"Expected MISS but got {tier}"
        rows.append({
            "trial": trial, "tier": "MISS", "operation": "access",
            "latency_ms": latency,
        })

    miss_avg = np.mean([r["latency_ms"] for r in rows if r["tier"] == "MISS"])
    print(f"     MISS avg: {miss_avg:.4f} ms  (range: "
          f"{min(r['latency_ms'] for r in rows if r['tier'] == 'MISS'):.4f} - "
          f"{max(r['latency_ms'] for r in rows if r['tier'] == 'MISS'):.4f})")

    # ── Measure Prefetch latency ──
    print(f"\n  📏 Measuring PREFETCH (L3→L2) latency...")
    for trial in range(n_trials):
        cache.reset()

        # Put a chunk in L3 by filling L1+L2
        n_fill = test_cfg.l1_capacity_chunks + test_cfg.l2_capacity_chunks + 2
        fill_ids = list(range(10000 + trial * n_fill, 10000 + trial * n_fill + n_fill))
        for fid in fill_ids:
            cache.insert_chunks([fid])

        target_cid = fill_ids[0]
        if cache.chunk_in_cache(target_cid) == "L3":
            cost = cache.hw_prefetch_measured(target_cid) if hasattr(cache, 'hw_prefetch_measured') else cache.prefetch(target_cid)
            rows.append({
                "trial": trial, "tier": "PREFETCH", "operation": "prefetch_l3_to_l2",
                "latency_ms": cost,
            })

    pf_rows = [r for r in rows if r["tier"] == "PREFETCH"]
    if pf_rows:
        pf_avg = np.mean([r["latency_ms"] for r in pf_rows])
        print(f"     PREFETCH avg: {pf_avg:.4f} ms  (range: "
              f"{min(r['latency_ms'] for r in pf_rows):.4f} - "
              f"{max(r['latency_ms'] for r in pf_rows):.4f})")

    # Cleanup
    cache.reset()

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════
# TEST 2: Sample Episode
# ═══════════════════════════════════════════════════════════════

def run_sample_episode(config: HardwareConfig) -> pd.DataFrame:
    """
    Run a single episode using HardwareCacheEnv on the RAG trace.
    Uses random actions to show that the pipeline works.

    Returns a DataFrame with per-step metrics.
    """
    print("\n" + "=" * 64)
    print("  TEST 2: Sample Episode (Random Agent)")
    print("=" * 64)

    # Find a trace file
    data_dir = PROJECT_ROOT / "data"
    trace_candidates = ["traces_rag.csv", "traces_prefix.csv", "traces_nocontext.csv"]
    trace_path = None
    for tc in trace_candidates:
        if (data_dir / tc).exists():
            trace_path = data_dir / tc
            break

    if trace_path is None:
        print("  ⚠️  No trace CSV found in data/. Skipping episode test.")
        return pd.DataFrame()

    print(f"  Using trace: {trace_path.name}")

    # Create environment with quiet config
    episode_cfg = HardwareConfig(
        l1_capacity_mb=config.l1_capacity_mb,
        l2_capacity_mb=config.l2_capacity_mb,
        l3_capacity_mb=config.l3_capacity_mb,
        chunk_size_bytes=config.chunk_size_bytes,
        l3_disk_dir=str(PROJECT_ROOT / "results" / "episode_cache_store"),
        verbose=False,
        enable_operation_log=True,
    )

    # Load embeddings if available
    emb_path = data_dir / f"embeddings_{trace_path.stem.replace('traces_', '')}.npy"
    embeddings = None
    if emb_path.exists():
        embeddings = np.load(str(emb_path))
        print(f"  Loaded embeddings: {emb_path.name}")

    env = HardwareCacheEnv(trace_path, config=episode_cfg, query_embeddings=embeddings)

    # Run one episode with random actions
    obs, _ = env.reset()
    step_rows = []
    total_reward = 0.0

    print(f"\n  Running {env.n_queries} steps with random actions...")

    while True:
        # Random action: randomly select which candidates to prefetch
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        step_rows.append({
            "step": info.get("step", 0),
            "reward": round(reward, 4),
            "access_latency_ms": round(info.get("access_latency_ms", 0), 4),
            "baseline_latency_ms": round(info.get("baseline_latency_ms", 0), 4),
            "prefetch_cost_ms": round(info.get("prefetch_cost_ms", 0), 4),
            "n_prefetched": info.get("n_prefetched", 0),
            "n_useful": info.get("n_useful", 0),
            "l1_hits": info.get("tier_counts", {}).get("L1", 0),
            "l2_hits": info.get("tier_counts", {}).get("L2", 0),
            "l3_hits": info.get("tier_counts", {}).get("L3", 0),
            "misses": info.get("tier_counts", {}).get("MISS", 0),
        })

        if terminated or truncated:
            break

    # Print episode summary
    if "episode_summary" in info:
        summ = info["episode_summary"]
        print(f"\n  📊 Episode Summary:")
        print(f"     Total reward:       {summ['total_reward']:+.2f}")
        print(f"     Hit rate:           {summ['hit_rate_pct']:.1f}%")
        print(f"     Prefetch accuracy:  {summ['prefetch_accuracy_pct']:.1f}%")
        print(f"     Avg latency (real): {summ['avg_measured_latency_ms']:.2f} ms")
        print(f"     Avg latency (base): {summ['avg_baseline_latency_ms']:.2f} ms")
        print(f"     Tier counts:        {summ['tier_counts']}")

    # Save operation log
    ops_path = PROJECT_ROOT / "results" / "hardware_benchmark_operations.csv"
    env.save_operation_log(ops_path)

    df = pd.DataFrame(step_rows)
    return df


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def generate_plots(latency_df: pd.DataFrame, episode_df: pd.DataFrame,
                   config: HardwareConfig):
    """Generate comparison plots and save to results/plots/."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not installed. Skipping plots.")
        return

    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Latency Comparison (Measured vs Simulated) ────
    if not latency_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Box plot of measured latencies
        tiers = ["L1", "L2", "L3", "MISS", "PREFETCH"]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6", "#f39c12"]
        data_by_tier = []
        labels = []
        plot_colors = []

        for tier, color in zip(tiers, colors):
            tier_data = latency_df[latency_df["tier"] == tier]["latency_ms"]
            if len(tier_data) > 0:
                data_by_tier.append(tier_data.values)
                labels.append(tier)
                plot_colors.append(color)

        if data_by_tier:
            bp = ax1.boxplot(data_by_tier, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Measured Latencies per Tier")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        # Right: Bar comparison — Measured avg vs Simulated constant
        measured_avgs = {}
        for tier in ["L1", "L2", "L3", "MISS", "PREFETCH"]:
            tier_data = latency_df[latency_df["tier"] == tier]["latency_ms"]
            if len(tier_data) > 0:
                measured_avgs[tier] = tier_data.mean()

        simulated_vals = {
            "L1": config.l1_hit_latency_ms,
            "L2": config.l2_hit_latency_ms,
            "L3": config.l3_hit_latency_ms,
            "MISS": config.cold_compute_per_chunk_ms,
            "PREFETCH": config.prefetch_l3_to_l2_ms,
        }

        common_tiers = [t for t in tiers if t in measured_avgs]
        if common_tiers:
            x = np.arange(len(common_tiers))
            width = 0.35

            meas_vals = [measured_avgs[t] for t in common_tiers]
            sim_vals = [simulated_vals[t] for t in common_tiers]

            bars1 = ax2.bar(x - width/2, meas_vals, width, label="Measured (Real HW)",
                           color="#e74c3c", alpha=0.8)
            bars2 = ax2.bar(x + width/2, sim_vals, width, label="Simulated (Config)",
                           color="#3498db", alpha=0.8)

            ax2.set_xlabel("Tier")
            ax2.set_ylabel("Latency (ms)")
            ax2.set_title("Measured vs Simulated Latencies")
            ax2.set_xticks(x)
            ax2.set_xticklabels(common_tiers)
            ax2.legend()
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars1, meas_vals):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars2, sim_vals):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        path = plots_dir / "hardware_latency_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Saved: {path.name}")

    # ── Plot 2: Episode Timeline ──────────────────────────────
    if not episode_df.empty and len(episode_df) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        steps = episode_df["step"]

        # Subplot 1: Latencies
        axes[0].plot(steps, episode_df["access_latency_ms"],
                    color="#e74c3c", label="Measured", alpha=0.8, linewidth=1.5)
        axes[0].plot(steps, episode_df["baseline_latency_ms"],
                    color="#3498db", label="Baseline", alpha=0.6, linestyle="--")
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_title("Per-Step Access Latency: Measured vs Baseline")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Rewards
        axes[1].bar(steps, episode_df["reward"], color="#2ecc71", alpha=0.7)
        axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        axes[1].set_ylabel("Reward")
        axes[1].set_title("Per-Step Reward")
        axes[1].grid(True, alpha=0.3)

        # Subplot 3: Tier hit distribution (stacked)
        axes[2].bar(steps, episode_df["l1_hits"], color="#e74c3c",
                   label="L1 (GPU)", alpha=0.8)
        axes[2].bar(steps, episode_df["l2_hits"], bottom=episode_df["l1_hits"],
                   color="#3498db", label="L2 (CPU)", alpha=0.8)
        axes[2].bar(steps, episode_df["l3_hits"],
                   bottom=episode_df["l1_hits"] + episode_df["l2_hits"],
                   color="#2ecc71", label="L3 (Disk)", alpha=0.8)
        axes[2].bar(steps, episode_df["misses"],
                   bottom=episode_df["l1_hits"] + episode_df["l2_hits"] + episode_df["l3_hits"],
                   color="#95a5a6", label="MISS", alpha=0.8)
        axes[2].set_ylabel("Chunk Count")
        axes[2].set_xlabel("Step (Query #)")
        axes[2].set_title("Per-Step Tier Hit Distribution")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = plots_dir / "hardware_episode_timeline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Saved: {path.name}")

    print(f"\n  All plots saved to: {plots_dir}/")


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

def print_summary_table(latency_df: pd.DataFrame, config: HardwareConfig):
    """Print a nice comparison table."""
    print("\n" + "=" * 64)
    print("  📊 BENCHMARK SUMMARY")
    print("=" * 64)

    header = f"  {'Tier':<10} {'Measured (avg)':<16} {'Measured (std)':<16} {'Simulated':<12} {'Ratio':<10}"
    print(header)
    print("  " + "─" * 60)

    simulated = {
        "L1": config.l1_hit_latency_ms,
        "L2": config.l2_hit_latency_ms,
        "L3": config.l3_hit_latency_ms,
        "MISS": config.cold_compute_per_chunk_ms,
        "PREFETCH": config.prefetch_l3_to_l2_ms,
    }

    for tier in ["L1", "L2", "L3", "MISS", "PREFETCH"]:
        tier_data = latency_df[latency_df["tier"] == tier]["latency_ms"]
        if len(tier_data) > 0:
            avg = tier_data.mean()
            std = tier_data.std()
            sim = simulated.get(tier, 0)
            ratio = avg / sim if sim > 0 else float("inf")
            print(f"  {tier:<10} {avg:<16.4f} {std:<16.4f} {sim:<12.2f} {ratio:<10.2f}x")
        else:
            print(f"  {tier:<10} {'N/A':<16} {'N/A':<16} {simulated.get(tier, 0):<12.2f} {'N/A':<10}")

    print("\n  Note: Ratio = Measured / Simulated")
    print("    > 1.0 means hardware is SLOWER than the simulation assumed")
    print("    < 1.0 means hardware is FASTER than the simulation assumed")
    print("=" * 64 + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the real hardware cache and compare with simulation"
    )
    parser.add_argument("--l1-mb", type=float, default=48.0,
                        help="L1 (GPU VRAM) capacity in MB (default: 48)")
    parser.add_argument("--l2-mb", type=float, default=51.2,
                        help="L2 (CPU RAM) capacity in MB (default: 51.2)")
    parser.add_argument("--l3-mb", type=float, default=5120.0,
                        help="L3 (Disk) capacity in MB (default: 5120)")
    parser.add_argument("--chunk-mb", type=float, default=3.0,
                        help="Chunk size in MB (default: 3.0)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials per tier (default: 20)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--skip-episode", action="store_true",
                        help="Skip the sample episode test")
    args = parser.parse_args()

    # Build config from CLI args
    chunk_bytes = int(args.chunk_mb * 1024 * 1024)
    config = HardwareConfig(
        l1_capacity_mb=args.l1_mb,
        l2_capacity_mb=args.l2_mb,
        l3_capacity_mb=args.l3_mb,
        chunk_size_bytes=chunk_bytes,
        verbose=args.verbose,
    )

    print("\n" + "🔧" * 32)
    print("  HARDWARE CACHE BENCHMARK")
    print("🔧" * 32)
    config.print_summary()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Test 1: Tier latencies ──
    latency_df = benchmark_tier_latencies(config, n_trials=args.trials)

    # Save latency results
    latency_path = results_dir / "hardware_benchmark_latencies.csv"
    latency_df.to_csv(latency_path, index=False)
    print(f"\n  💾 Latency results saved: {latency_path.name}")

    # Print summary table
    print_summary_table(latency_df, config)

    # ── Test 2: Sample episode ──
    episode_df = pd.DataFrame()
    if not args.skip_episode:
        episode_df = run_sample_episode(config)
        if not episode_df.empty:
            ep_path = results_dir / "hardware_benchmark_episode.csv"
            episode_df.to_csv(ep_path, index=False)
            print(f"  💾 Episode results saved: {ep_path.name}")

    # ── Generate plots ──
    print("\n  📊 Generating plots...")
    generate_plots(latency_df, episode_df, config)

    elapsed = time.time() - t_start
    print(f"\n  ⏱️  Total benchmark time: {elapsed:.1f}s")
    print("  ✅ Benchmark complete!\n")


if __name__ == "__main__":
    main()
