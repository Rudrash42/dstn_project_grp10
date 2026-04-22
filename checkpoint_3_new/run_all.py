#!/usr/bin/env python3
"""
Run All: Single-command pipeline for RL KV Cache Prefetching
==============================================================

Runs the complete pipeline:
  1. Generate traces (or skip with --skip-traces)
  2. Train (behavioral cloning → PPO on hardware)
  3. Evaluate (RL agent + baselines on hardware)

Usage:
    python run_all.py                  # Full pipeline (skip trace gen)
    python run_all.py --quick          # Quick smoke test (~3-5 min)
    python run_all.py --with-traces    # Include trace generation (needs vLLM)
    python run_all.py --train-only     # Only train, skip evaluation
    python run_all.py --eval-only      # Only evaluate (needs trained model)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, cmd: list, cwd: Path = PROJECT_ROOT) -> bool:
    """Run a subprocess step. Returns True if successful."""
    print(f"\n{'═' * 64}")
    print(f"  STEP: {name}")
    print(f"{'═' * 64}\n")

    result = subprocess.run(cmd, cwd=str(cwd))

    if result.returncode != 0:
        print(f"\n  ❌ FAILED: {name} (exit code {result.returncode})")
        return False

    print(f"\n  ✅ {name} — complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full RL cache prefetching pipeline")
    parser.add_argument("--with-traces", action="store_true",
                        help="Run trace generation (needs vLLM + GPU)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (500 timesteps)")
    parser.add_argument("--train-only", action="store_true",
                        help="Only run training, skip evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation (needs trained model)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--l1-mb", type=float, default=None,
                        help="Override L1 capacity (MB)")
    parser.add_argument("--l2-mb", type=float, default=None,
                        help="Override L2 capacity (MB)")
    args = parser.parse_args()

    t_start = time.time()
    python = sys.executable

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║   RL-Based KV Cache Prefetching — Full Pipeline              ║")
    print("╚" + "═" * 62 + "╝")

    # ── Build CLI args to forward ──
    extra_args = []
    if args.cpu_only:
        extra_args.extend(["--cpu-only"])
    if args.l1_mb is not None:
        extra_args.extend(["--l1-mb", str(args.l1_mb)])
    if args.l2_mb is not None:
        extra_args.extend(["--l2-mb", str(args.l2_mb)])

    # ── Step 1: Trace generation (optional) ──
    if args.with_traces and not args.eval_only:
        trace_cmd = [python, "generate_traces.py"]
        if args.quick:
            trace_cmd.extend(["--max-queries", "10"])
        if not run_step("Trace Generation", trace_cmd):
            sys.exit(1)
    else:
        # Check if traces exist, generate without LLM if needed
        data_dir = PROJECT_ROOT / "data"
        traces_exist = all(
            (data_dir / f"traces_{name}.csv").exists()
            for name in ["prefix", "rag", "nocontext", "multiturn"]
        )
        if not traces_exist and not args.eval_only:
            print("\n  Traces not found. Generating without LLM (--skip-llm)...")
            trace_cmd = [python, "generate_traces.py", "--skip-llm"]
            if args.quick:
                trace_cmd.extend(["--max-queries", "10"])
            if not run_step("Trace Generation (no LLM)", trace_cmd):
                sys.exit(1)
        elif traces_exist:
            print("\n  ✅ Traces already exist in data/")

    # ── Step 2: Training ──
    if not args.eval_only:
        train_cmd = [python, "train.py"] + extra_args
        if args.quick:
            train_cmd.append("--quick")
        if not run_step("Training (BC → PPO)", train_cmd):
            sys.exit(1)

    # ── Step 3: Evaluation ──
    if not args.train_only:
        eval_cmd = [python, "evaluate.py"] + extra_args
        if not run_step("Evaluation", eval_cmd):
            sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\n{'╔' + '═' * 62 + '╗'}")
    print(f"║   PIPELINE COMPLETE — {elapsed/60:.1f} minutes{'':>30}║")
    print(f"{'╚' + '═' * 62 + '╝'}")
    print(f"\n  Results in: {PROJECT_ROOT / 'results'}/")
    print(f"  Models in:  {PROJECT_ROOT / 'models'}/\n")


if __name__ == "__main__":
    main()
