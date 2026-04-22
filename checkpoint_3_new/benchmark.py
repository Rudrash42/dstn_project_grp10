#!/usr/bin/env python3
"""
Hardware Benchmark: Measure Real L1/L2/L3 Latencies
=====================================================

Calibrates the latency estimates in config.yaml by running REAL
data transfers on your GPU/CPU/Disk:

  L1: GPU tensor access (read from VRAM)
  L2: CPU→GPU transfer (pinned memory → CUDA via PCIe)
  L3: Disk→CPU→GPU transfer (NVMe read + PCIe transfer)
  Cold: New tensor allocation on GPU (simulates KV cache creation)
  Prefetch: Disk→CPU (NVMe read into pinned memory)

Also measures:
  - GPU memory (total, used, free)
  - Transfer bandwidth estimates
  - Latency distribution (mean, median, p95, p99)

Outputs:
  results/benchmark_results.json   — all measured latencies
  results/benchmark_results.csv    — per-trial raw data
  results/plots/benchmark.png      — latency distributions

Usage:
    python benchmark.py              # Full benchmark with config update
    python benchmark.py --no-update  # Benchmark only, don't update config
    python benchmark.py --trials 50  # More trials for accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DISK_DIR = PROJECT_ROOT / "data" / "benchmark_tmp"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def benchmark_hardware(cfg: dict, n_trials: int = 30) -> dict:
    """
    Run comprehensive hardware benchmarks.
    Returns dict with all measured latencies.
    """
    chunk_size_bytes = cfg.get("chunk_size_bytes", 3_145_728)
    tensor_elements = chunk_size_bytes // 4  # float32

    use_cuda = torch.cuda.is_available() and not cfg.get("force_cpu_mode", False)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  HARDWARE BENCHMARK")
    print(f"{'=' * 60}")

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {total_mem:.1f} GB")
    else:
        gpu_name = "CPU-only"
        total_mem = 0
        print(f"  Mode: CPU-only (no CUDA)")

    print(f"  Chunk size: {chunk_size_bytes / 1e6:.1f} MB ({tensor_elements} float32)")
    print(f"  Trials: {n_trials}")

    # Warmup GPU
    if use_cuda:
        for _ in range(10):
            t = torch.randn(tensor_elements, device=device)
            _ = t.sum()
            del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    results = {
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(total_mem, 2),
        "chunk_size_bytes": chunk_size_bytes,
        "n_trials": n_trials,
    }
    raw_trials = []

    # ── 1. L1 HIT: GPU tensor read ──────────────────────────────

    print(f"\n  Measuring L1 (GPU read)...")
    gpu_tensor = torch.randn(tensor_elements, device=device)
    torch.cuda.synchronize() if use_cuda else None
    l1_times = []

    for trial in range(n_trials):
        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = gpu_tensor.sum()
        if use_cuda: torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        l1_times.append(elapsed)
        raw_trials.append({"tier": "L1", "trial": trial, "latency_ms": elapsed})

    del gpu_tensor
    results["l1_hit_ms"] = {
        "mean": round(np.mean(l1_times), 4),
        "median": round(np.median(l1_times), 4),
        "p95": round(np.percentile(l1_times, 95), 4),
        "p99": round(np.percentile(l1_times, 99), 4),
        "std": round(np.std(l1_times), 4),
    }
    print(f"    L1: {results['l1_hit_ms']['mean']:.4f}ms "
          f"(median={results['l1_hit_ms']['median']:.4f}, "
          f"p95={results['l1_hit_ms']['p95']:.4f})")

    # ── 2. L2 HIT: CPU pinned → GPU transfer ────────────────────

    print(f"  Measuring L2 (CPU→GPU transfer)...")
    l2_times = []

    for trial in range(n_trials):
        cpu_tensor = torch.randn(tensor_elements, pin_memory=use_cuda)
        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_copy = cpu_tensor.to(device, non_blocking=False)
        _ = gpu_copy.sum()
        if use_cuda: torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        l2_times.append(elapsed)
        raw_trials.append({"tier": "L2", "trial": trial, "latency_ms": elapsed})
        del cpu_tensor, gpu_copy

    results["l2_hit_ms"] = {
        "mean": round(np.mean(l2_times), 4),
        "median": round(np.median(l2_times), 4),
        "p95": round(np.percentile(l2_times, 95), 4),
        "p99": round(np.percentile(l2_times, 99), 4),
        "std": round(np.std(l2_times), 4),
        "bandwidth_gbps": round(chunk_size_bytes / (np.mean(l2_times) / 1000) / 1e9, 2),
    }
    print(f"    L2: {results['l2_hit_ms']['mean']:.4f}ms "
          f"(~{results['l2_hit_ms']['bandwidth_gbps']:.1f} GB/s)")

    # ── 3. L3 HIT: Disk → CPU → GPU transfer ────────────────────

    print(f"  Measuring L3 (Disk→CPU→GPU)...")
    BENCHMARK_DISK_DIR.mkdir(parents=True, exist_ok=True)
    l3_times = []

    for trial in range(n_trials):
        # Write tensor to disk first
        disk_tensor = torch.randn(tensor_elements)
        path = str(BENCHMARK_DISK_DIR / f"bench_{trial}.pt")
        torch.save(disk_tensor, path)
        del disk_tensor

        # Flush OS cache (best effort)
        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        gpu_copy = loaded.to(device)
        _ = gpu_copy.sum()
        if use_cuda: torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        l3_times.append(elapsed)
        raw_trials.append({"tier": "L3", "trial": trial, "latency_ms": elapsed})
        del loaded, gpu_copy
        os.remove(path)

    results["l3_hit_ms"] = {
        "mean": round(np.mean(l3_times), 4),
        "median": round(np.median(l3_times), 4),
        "p95": round(np.percentile(l3_times, 95), 4),
        "p99": round(np.percentile(l3_times, 99), 4),
        "std": round(np.std(l3_times), 4),
        "bandwidth_mbps": round(chunk_size_bytes / (np.mean(l3_times) / 1000) / 1e6, 1),
    }
    print(f"    L3: {results['l3_hit_ms']['mean']:.4f}ms "
          f"(~{results['l3_hit_ms']['bandwidth_mbps']:.0f} MB/s)")

    # ── 4. COLD MISS: New tensor on GPU ───────────────────────────

    print(f"  Measuring COLD MISS (GPU tensor allocation)...")
    cold_times = []

    for trial in range(n_trials):
        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_tensor = torch.randn(tensor_elements, device=device)
        _ = new_tensor.sum()
        if use_cuda: torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        cold_times.append(elapsed)
        raw_trials.append({"tier": "COLD", "trial": trial, "latency_ms": elapsed})
        del new_tensor

    results["cold_alloc_ms"] = {
        "mean": round(np.mean(cold_times), 4),
        "median": round(np.median(cold_times), 4),
        "p95": round(np.percentile(cold_times, 95), 4),
        "p99": round(np.percentile(cold_times, 99), 4),
        "std": round(np.std(cold_times), 4),
    }
    print(f"    COLD: {results['cold_alloc_ms']['mean']:.4f}ms "
          f"(tensor alloc only, NOT LLM forward pass)")

    # ── 5. PREFETCH: Disk → CPU pinned ────────────────────────────

    print(f"  Measuring PREFETCH (Disk→CPU)...")
    prefetch_times = []

    for trial in range(n_trials):
        disk_tensor = torch.randn(tensor_elements)
        path = str(BENCHMARK_DISK_DIR / f"bench_pf_{trial}.pt")
        torch.save(disk_tensor, path)
        del disk_tensor

        t0 = time.perf_counter()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        pinned = torch.empty(tensor_elements, pin_memory=use_cuda)
        pinned.copy_(loaded)
        elapsed = (time.perf_counter() - t0) * 1000
        prefetch_times.append(elapsed)
        raw_trials.append({"tier": "PREFETCH", "trial": trial, "latency_ms": elapsed})
        del loaded, pinned
        os.remove(path)

    results["prefetch_ms"] = {
        "mean": round(np.mean(prefetch_times), 4),
        "median": round(np.median(prefetch_times), 4),
        "p95": round(np.percentile(prefetch_times, 95), 4),
        "p99": round(np.percentile(prefetch_times, 99), 4),
        "std": round(np.std(prefetch_times), 4),
    }
    print(f"    PREFETCH: {results['prefetch_ms']['mean']:.4f}ms")

    # ── GPU Memory ────────────────────────────────────────────────

    if use_cuda:
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        mem_reserved = torch.cuda.memory_reserved() / (1024**2)
        mem_free = (torch.cuda.get_device_properties(0).total_mem
                    - torch.cuda.memory_reserved()) / (1024**2)
        results["gpu_memory"] = {
            "allocated_mb": round(mem_alloc, 1),
            "reserved_mb": round(mem_reserved, 1),
            "free_mb": round(mem_free, 1),
        }
        print(f"\n  GPU Memory: {mem_free:.0f} MB free / "
              f"{total_mem * 1024:.0f} MB total")

    # Cleanup
    if BENCHMARK_DISK_DIR.exists():
        shutil.rmtree(BENCHMARK_DISK_DIR)

    results["raw_trials"] = raw_trials
    return results


def save_results(results: dict):
    """Save benchmark results to JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON (without raw trials for readability)
    summary = {k: v for k, v in results.items() if k != "raw_trials"}
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results → benchmark_results.json")

    # CSV (raw trial data)
    import pandas as pd
    if results.get("raw_trials"):
        pd.DataFrame(results["raw_trials"]).to_csv(
            RESULTS_DIR / "benchmark_raw.csv", index=False)
        print(f"  Raw data → benchmark_raw.csv")


def plot_results(results: dict):
    """Generate benchmark visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available.")
        return

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw = results.get("raw_trials", [])
    if not raw:
        return

    import pandas as pd
    df = pd.DataFrame(raw)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Box plot of latencies
    tiers = ["L1", "L2", "L3", "COLD", "PREFETCH"]
    tier_data = [df[df["tier"] == t]["latency_ms"].values for t in tiers]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6", "#f39c12"]

    bp = axes[0].boxplot(tier_data, labels=tiers, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency Distribution by Tier")
    axes[0].grid(True, alpha=0.3)

    # 2. Bar chart of mean latencies (log scale)
    means = [results.get(f"{t.lower()}_hit_ms", results.get(f"{t.lower()}_alloc_ms",
             results.get(f"{t.lower()}_ms", {"mean": 0})))["mean"]
             for t in ["l1", "l2", "l3", "cold", "prefetch"]]
    axes[1].bar(tiers, means, color=colors, alpha=0.8)
    axes[1].set_ylabel("Mean Latency (ms)")
    axes[1].set_title("Mean Latency (log scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(means):
        axes[1].text(i, v * 1.2, f"{v:.3f}", ha="center", fontsize=9)

    # 3. Speedup relative to cold miss
    cold_mean = results.get("cold_alloc_ms", {"mean": 1})["mean"]
    speedups = [cold_mean / m if m > 0 else 0 for m in means]
    axes[2].bar(tiers, speedups, color=colors, alpha=0.8)
    axes[2].set_ylabel("Speedup vs Cold Miss")
    axes[2].set_title("Cache Tier Speedup")
    axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → benchmark.png")


def update_config(results: dict):
    """Update config.yaml with measured latencies."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    old_vals = {}
    updates = {
        "l1_hit_latency_ms": results["l1_hit_ms"]["mean"],
        "l2_hit_latency_ms": results["l2_hit_ms"]["mean"],
        "l3_hit_latency_ms": results["l3_hit_ms"]["mean"],
        "prefetch_l3_to_l2_ms": results["prefetch_ms"]["mean"],
    }

    print(f"\n  Updating config.yaml:")
    for key, new_val in updates.items():
        old_val = cfg.get(key, "N/A")
        old_vals[key] = old_val
        cfg[key] = round(new_val, 4)
        print(f"    {key}: {old_val} → {new_val:.4f}")

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Config updated ✓")


def print_summary(results: dict):
    """Print a formatted summary table."""
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Tier':<12} {'Mean (ms)':<12} {'Median':<12} {'P95':<12} {'P99':<12}")
    print(f"  {'─' * 56}")

    for label, key in [("L1 (GPU)", "l1_hit_ms"),
                        ("L2 (CPU→GPU)", "l2_hit_ms"),
                        ("L3 (Disk)", "l3_hit_ms"),
                        ("Cold Miss", "cold_alloc_ms"),
                        ("Prefetch", "prefetch_ms")]:
        d = results.get(key, {})
        print(f"  {label:<12} {d.get('mean', 0):<12.4f} "
              f"{d.get('median', 0):<12.4f} "
              f"{d.get('p95', 0):<12.4f} "
              f"{d.get('p99', 0):<12.4f}")

    print(f"  {'─' * 56}")

    # Speed ratios
    l1 = results.get("l1_hit_ms", {}).get("mean", 0.1)
    l2 = results.get("l2_hit_ms", {}).get("mean", 0.24)
    l3 = results.get("l3_hit_ms", {}).get("mean", 6.0)
    print(f"\n  Speed ratios:")
    print(f"    L2/L1 = {l2/l1:.1f}x slower")
    print(f"    L3/L1 = {l3/l1:.1f}x slower")
    print(f"    L3/L2 = {l3/l2:.1f}x slower")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark hardware cache latencies")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of trials per measurement (default: 30)")
    parser.add_argument("--no-update", action="store_true",
                        help="Don't update config.yaml")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU-only mode")
    args = parser.parse_args()

    cfg = load_config()
    if args.cpu_only:
        cfg["force_cpu_mode"] = True

    results = benchmark_hardware(cfg, n_trials=args.trials)
    print_summary(results)
    save_results(results)
    plot_results(results)

    if not args.no_update:
        update_config(results)


if __name__ == "__main__":
    main()
