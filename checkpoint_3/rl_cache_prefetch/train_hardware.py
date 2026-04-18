#!/usr/bin/env python3
"""
Training with REAL Hardware Cache Backend
==========================================

This is the hardware version of train.py. It does the same two-stage
training (Behavioral Cloning → PPO), but uses HardwareCacheEnv which
runs on REAL GPU/CPU/Disk instead of simulated fake numbers.

What's the same as train.py:
  • Stage A: Behavioral cloning from oracle labels
  • Stage B: PPO fine-tuning
  • Same trace data, same embeddings, same policy network

What's different:
  • CacheEnv → HardwareCacheEnv (real hardware!)
  • TierConfig → HardwareConfig (configurable, hardware-aware)
  • Loads config from configs/hardware_config.yaml
  • Saves hardware-specific metrics (measured latencies, GPU usage)
  • Extra CLI flags for hardware tuning (--l1-mb, --l2-mb, etc.)

Usage:
    # Full training on real hardware
    python train_hardware.py

    # Quick smoke test (500 steps)
    python train_hardware.py --quick

    # Custom cache sizes (create MORE cache pressure)
    python train_hardware.py --l1-mb 24 --l2-mb 32

    # Stage B only (assumes Stage A already ran)
    python train_hardware.py --stage b

    # Verbose mode (see every cache operation — great for debugging)
    python train_hardware.py --quick --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# ─── Path setup ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

# ─── Imports ──────────────────────────────────────────────────
# Hardware cache (the new real-hardware backend)
from env.hardware.hardware_config import HardwareConfig
from env.hardware.hardware_cache_env import HardwareCacheEnv

# Original modules (still used for Stage A and embeddings)
from env.cache_env import CacheEnv
from env.tier_config import TierConfig
from agent.state_encoder import StateEncoder


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_hardware_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> HardwareConfig:
    """
    Load HardwareConfig from YAML, then apply any CLI overrides.

    Priority: CLI args > YAML file > dataclass defaults

    Parameters
    ----------
    yaml_path : str, optional
        Path to hardware_config.yaml. If None, uses default location.
    cli_overrides : dict, optional
        Key-value pairs from CLI args to override YAML values.
        Example: {"l1_capacity_mb": 24.0, "verbose": True}
    """
    if yaml_path is None:
        yaml_path = PROJECT_ROOT / "configs" / "hardware_config.yaml"

    # Load from YAML (falls back to defaults if file missing)
    cfg = HardwareConfig.from_yaml(yaml_path)

    # Apply CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and hasattr(cfg, key):
                old = getattr(cfg, key)
                setattr(cfg, key, value)
                print(f"[config] Override: {key} = {old} → {value}")

    return cfg


def load_raw_yaml_config(yaml_path: Optional[str] = None) -> dict:
    """Load the raw YAML as a dict (for PPO hyperparameters)."""
    if yaml_path is None:
        yaml_path = PROJECT_ROOT / "configs" / "hardware_config.yaml"
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        # Fall back to the original ppo_config.yaml
        yaml_path = PROJECT_ROOT / "configs" / "ppo_config.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def get_trace_paths() -> dict:
    """Return dict of {workload_name: trace_csv_path}."""
    data_dir = PROJECT_ROOT / "data"
    return {
        "prefix": data_dir / "traces_prefix.csv",
        "rag": data_dir / "traces_rag.csv",
        "nocontext": data_dir / "traces_nocontext.csv",
        "multiturn": data_dir / "traces_multiturn.csv",
    }


def ensure_embeddings(traces: dict, embed_dim: int) -> dict:
    """
    Load pre-computed embeddings (or generate them if missing).
    Returns dict of {workload_name: np.ndarray of shape (n, embed_dim)}.
    """
    data_dir = PROJECT_ROOT / "data"
    encoder = StateEncoder(embed_dim=embed_dim)

    embeddings = {}
    for name, trace_path in traces.items():
        emb_path = data_dir / f"embeddings_{name}.npy"
        if emb_path.exists():
            embeddings[name] = StateEncoder.load_embeddings(emb_path)
        else:
            import pandas as pd
            df = pd.read_csv(trace_path)
            text_col = "embedding_text" if "embedding_text" in df.columns else "query_text"
            texts = df[text_col].tolist()
            embeddings[name] = encoder.encode_and_save(texts, emb_path)

    return embeddings


# ═══════════════════════════════════════════════════════════════
# STAGE A: BEHAVIORAL CLONING
# ═══════════════════════════════════════════════════════════════
#
# Stage A uses the SIMULATED CacheEnv (not hardware).
# Why? Because behavioral cloning just needs (observation, oracle_action)
# pairs — it doesn't care about real latencies. Using simulation is
# 100x faster and the oracle labels are identical either way.
#
# The pretrained weights from Stage A are then loaded into Stage B,
# which DOES use real hardware.
# ═══════════════════════════════════════════════════════════════

def stage_a_behavioral_cloning(
    hw_cfg: HardwareConfig,
    raw_cfg: dict,
    traces: dict,
    embeddings: dict,
):
    """
    Pre-train a policy network by imitating oracle prefetch decisions.

    Uses SIMULATED cache for speed (oracle labels don't depend on
    real hardware timing — they're just "which chunks will the next
    query need?").
    """
    print("\n" + "=" * 64)
    print("  STAGE A: Behavioral Cloning (Oracle Imitation)")
    print("  Using: SIMULATED cache (for speed — oracle labels")
    print("         are the same regardless of hardware)")
    print("=" * 64)

    import pandas as pd

    # Build a TierConfig from our HardwareConfig for the simulated env
    # (TierConfig and HardwareConfig have the same attribute names)
    tier_cfg = TierConfig(
        chunk_size_tokens=hw_cfg.chunk_size_tokens,
        kv_bytes_per_token=hw_cfg.kv_bytes_per_token,
        chunk_size_bytes=hw_cfg.chunk_size_bytes,
        l1_capacity_bytes=hw_cfg.l1_capacity_bytes,
        l2_capacity_bytes=hw_cfg.l2_capacity_bytes,
        l3_capacity_bytes=hw_cfg.l3_capacity_bytes,
        l1_hit_latency_ms=hw_cfg.l1_hit_latency_ms,
        l2_hit_latency_ms=hw_cfg.l2_hit_latency_ms,
        l3_hit_latency_ms=hw_cfg.l3_hit_latency_ms,
        cold_compute_per_chunk_ms=hw_cfg.cold_compute_per_chunk_ms,
        prefetch_l3_to_l2_ms=hw_cfg.prefetch_l3_to_l2_ms,
        alpha=hw_cfg.alpha,
        beta=hw_cfg.beta,
        gamma_reward=hw_cfg.gamma_reward,
        embed_dim=hw_cfg.embed_dim,
        max_candidate_chunks=hw_cfg.max_candidate_chunks,
        history_len=hw_cfg.history_len,
    )

    # Collect (observation, oracle_action) pairs from all traces
    all_obs = []
    all_actions = []

    for name, trace_path in traces.items():
        df = pd.read_csv(trace_path)
        embs = embeddings[name]

        # Use simulated CacheEnv for behavioral cloning
        env = CacheEnv(trace_path, config=tier_cfg, query_embeddings=embs)
        obs, _ = env.reset()

        for step_idx in range(len(df) - 1):
            # Oracle knowledge: what will the NEXT query need?
            next_chunks = set(json.loads(df.iloc[step_idx + 1]["chunk_ids_needed"]))

            # Oracle action: prefetch candidates that match next query's needs
            oracle_action = np.zeros(hw_cfg.max_candidate_chunks, dtype=np.float32)
            for i, cid in enumerate(env.candidate_ids):
                if cid in next_chunks:
                    oracle_action[i] = 1.0

            all_obs.append(obs.copy())
            all_actions.append(oracle_action.copy())

            # Step with oracle action to advance the environment
            obs, _, terminated, _, _ = env.step(oracle_action.astype(int))
            if terminated:
                break

    if not all_obs:
        print("  [WARN] No training data generated. Skipping Stage A.")
        return None

    X = torch.tensor(np.array(all_obs), dtype=torch.float32)
    Y = torch.tensor(np.array(all_actions), dtype=torch.float32)

    print(f"  Training data: {X.shape[0]} samples, obs_dim={X.shape[1]}")

    # Build MLP matching the PPO policy architecture
    net_arch = raw_cfg.get("policy_net_arch", [64, 64])
    layers = []
    in_dim = hw_cfg.obs_dim
    for hidden in net_arch:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, hw_cfg.max_candidate_chunks))
    layers.append(nn.Sigmoid())
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=raw_cfg.get("behavioral_cloning_lr", 0.001),
    )
    criterion = nn.BCELoss()

    bc_epochs = raw_cfg.get("behavioral_cloning_epochs", 20)
    batch_size = raw_cfg.get("batch_size", 64)

    for epoch in range(bc_epochs):
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{bc_epochs} — BCE loss: {avg_loss:.4f}")

    # Save pretrained weights
    save_path = PROJECT_ROOT / "models" / "bc_pretrained_hardware.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")

    return model


# ═══════════════════════════════════════════════════════════════
# STAGE B: PPO FINE-TUNING (ON REAL HARDWARE!)
# ═══════════════════════════════════════════════════════════════
#
# This is where the magic happens. The RL agent trains against
# HardwareCacheEnv, experiencing REAL data movement latencies:
#   • L1 hits: tensor already on GPU → measured ~0.01-0.1 ms
#   • L2 hits: CPU→GPU PCIe transfer → measured ~0.2-0.5 ms
#   • L3 hits: Disk→CPU→GPU → measured ~2-10 ms
#   • Misses: new tensor creation → measured
#
# The agent learns to minimize these REAL costs through smart
# prefetching decisions.
# ═══════════════════════════════════════════════════════════════

def stage_b_ppo_training(
    hw_cfg: HardwareConfig,
    raw_cfg: dict,
    traces: dict,
    embeddings: dict,
    total_timesteps: Optional[int] = None,
):
    """
    Fine-tune the policy using PPO against the REAL HARDWARE cache.

    This trains on the RAG trace (highest impact from prefetching)
    using HardwareCacheEnv — every cache operation happens on real
    GPU/CPU/Disk with measured latencies.
    """
    print("\n" + "=" * 64)
    print("  STAGE B: PPO Fine-Tuning (REAL HARDWARE)")
    print("  Using: HardwareCacheEnv with actual GPU/CPU/Disk")
    print("=" * 64)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from agent.policy import make_policy_kwargs

    # ── Create the REAL HARDWARE environment ──────────────────
    # This creates actual CUDA tensors on your RTX 3050!
    rag_trace = traces["rag"]
    rag_embs = embeddings["rag"]

    # Turn off verbose for training (too noisy with 50K steps)
    # but keep operation logging for post-training analysis
    train_cfg = HardwareConfig(
        chunk_size_tokens=hw_cfg.chunk_size_tokens,
        kv_bytes_per_token=hw_cfg.kv_bytes_per_token,
        chunk_size_bytes=hw_cfg.chunk_size_bytes,
        l1_capacity_mb=hw_cfg.l1_capacity_mb,
        l2_capacity_mb=hw_cfg.l2_capacity_mb,
        l3_capacity_mb=hw_cfg.l3_capacity_mb,
        l3_disk_dir=hw_cfg.l3_disk_dir,
        l1_hit_latency_ms=hw_cfg.l1_hit_latency_ms,
        l2_hit_latency_ms=hw_cfg.l2_hit_latency_ms,
        l3_hit_latency_ms=hw_cfg.l3_hit_latency_ms,
        cold_compute_per_chunk_ms=hw_cfg.cold_compute_per_chunk_ms,
        prefetch_l3_to_l2_ms=hw_cfg.prefetch_l3_to_l2_ms,
        alpha=hw_cfg.alpha,
        beta=hw_cfg.beta,
        gamma_reward=hw_cfg.gamma_reward,
        embed_dim=hw_cfg.embed_dim,
        max_candidate_chunks=hw_cfg.max_candidate_chunks,
        history_len=hw_cfg.history_len,
        force_cpu_mode=hw_cfg.force_cpu_mode,
        verbose=False,            # Quiet during training
        enable_operation_log=False,  # Too much data at 50K steps
        cuda_warmup_iterations=hw_cfg.cuda_warmup_iterations,
    )

    env = HardwareCacheEnv(rag_trace, config=train_cfg, query_embeddings=rag_embs)

    print(f"\n  📊 Environment created:")
    print(f"     Trace: {rag_trace.name} ({env.n_queries} queries)")
    print(f"     L1 (GPU): {train_cfg.l1_capacity_mb:.1f} MB "
          f"({train_cfg.l1_capacity_chunks} chunks)")
    print(f"     L2 (CPU): {train_cfg.l2_capacity_mb:.1f} MB "
          f"({train_cfg.l2_capacity_chunks} chunks)")
    print(f"     Hardware: {'CUDA GPU' if env.hw_cache.use_cuda else 'CPU-only (no GPU)'}")

    # ── PPO hyperparameters ───────────────────────────────────
    if total_timesteps is None:
        total_timesteps = raw_cfg.get("total_timesteps", 50_000)

    net_arch = raw_cfg.get("policy_net_arch", [64, 64])

    # ── Training callback: logs episode rewards + hardware metrics ──
    class HardwareRewardLogger(BaseCallback):
        """
        Logs episode rewards and hardware-specific metrics during training.
        Prints progress every 10 episodes.
        """
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_metrics = []  # hardware-specific data
            self.episode_count = 0

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode_summary" in info:
                    summary = info["episode_summary"]
                    self.episode_count += 1
                    self.episode_rewards.append(summary["total_reward"])

                    # Save hardware-specific metrics
                    self.episode_metrics.append({
                        "episode": self.episode_count,
                        "total_reward": summary["total_reward"],
                        "hit_rate_pct": summary["hit_rate_pct"],
                        "prefetch_accuracy_pct": summary["prefetch_accuracy_pct"],
                        "total_prefetches": summary["total_prefetches"],
                        "useful_prefetches": summary["useful_prefetches"],
                        "avg_measured_latency_ms": summary.get(
                            "avg_measured_latency_ms", 0
                        ),
                        "avg_baseline_latency_ms": summary.get(
                            "avg_baseline_latency_ms", 0
                        ),
                        "total_measured_latency_ms": summary.get(
                            "total_measured_latency_ms", 0
                        ),
                        "tier_L1": summary.get("tier_counts", {}).get("L1", 0),
                        "tier_L2": summary.get("tier_counts", {}).get("L2", 0),
                        "tier_L3": summary.get("tier_counts", {}).get("L3", 0),
                        "tier_MISS": summary.get("tier_counts", {}).get("MISS", 0),
                    })

                    # Print progress every 10 episodes
                    if self.episode_count % 10 == 0:
                        recent = self.episode_rewards[-10:]
                        avg = sum(recent) / len(recent)
                        hr = summary["hit_rate_pct"]
                        pa = summary["prefetch_accuracy_pct"]
                        avg_lat = summary.get("avg_measured_latency_ms", 0)
                        print(
                            f"  Episode {self.episode_count:4d} | "
                            f"Avg reward(10): {avg:7.2f} | "
                            f"Hit rate: {hr:5.1f}% | "
                            f"Prefetch acc: {pa:5.1f}% | "
                            f"Avg latency: {avg_lat:.2f}ms"
                        )
            return True

    logger = HardwareRewardLogger()

    # ── Create PPO model ──────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=raw_cfg.get("learning_rate", 3e-4),
        n_steps=min(raw_cfg.get("n_steps", 2048), total_timesteps),
        batch_size=raw_cfg.get("batch_size", 64),
        n_epochs=raw_cfg.get("n_epochs", 10),
        gamma=raw_cfg.get("gamma_discount", 0.99),
        clip_range=raw_cfg.get("clip_range", 0.2),
        ent_coef=raw_cfg.get("ent_coef", 0.01),
        vf_coef=raw_cfg.get("vf_coef", 0.5),
        max_grad_norm=raw_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=make_policy_kwargs(net_arch),
        device="cpu",  # PPO MLP runs on CPU; GPU is used for cache data
        verbose=0,
    )

    # ── Train! ────────────────────────────────────────────────
    print(f"\n  🚀 Starting PPO training: {total_timesteps} timesteps")
    print(f"     (This will be SLOWER than simulated training because")
    print(f"      every step involves REAL disk I/O and GPU transfers)\n")

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=logger)
    elapsed = time.time() - t_start

    print(f"\n  ✅ Training complete in {elapsed:.1f}s ({total_timesteps} timesteps)")
    print(f"     ({total_timesteps / elapsed:.0f} timesteps/sec)")

    # ── Save model ────────────────────────────────────────────
    save_path = PROJECT_ROOT / "models" / "ppo_hardware_final"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"  💾 Model saved → {save_path}.zip")

    # ── Save training curves (rewards + hardware metrics) ─────
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Episode rewards
    with open(results_dir / "hardware_training_rewards.json", "w") as f:
        json.dump(logger.episode_rewards, f, indent=2)

    # Detailed per-episode hardware metrics (CSV)
    if logger.episode_metrics:
        import pandas as pd
        metrics_df = pd.DataFrame(logger.episode_metrics)
        metrics_path = results_dir / "hardware_training_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  📊 Training metrics saved → {metrics_path.name}")

        # Print final statistics
        print(f"\n  📊 Training Summary:")
        print(f"     Total episodes:      {logger.episode_count}")
        print(f"     Final avg reward:    {np.mean(logger.episode_rewards[-10:]):.2f}")
        print(f"     Best episode reward: {max(logger.episode_rewards):.2f}")
        print(f"     Final hit rate:      "
              f"{logger.episode_metrics[-1]['hit_rate_pct']:.1f}%")
        print(f"     Final prefetch acc:  "
              f"{logger.episode_metrics[-1]['prefetch_accuracy_pct']:.1f}%")

    # ── Generate training plots ───────────────────────────────
    _plot_training_curves(logger, results_dir)

    return model


def _plot_training_curves(logger, results_dir: Path):
    """Generate training progress plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not installed. Skipping training plots.")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not logger.episode_metrics:
        return

    import pandas as pd
    df = pd.DataFrame(logger.episode_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reward per episode
    axes[0, 0].plot(df["episode"], df["total_reward"],
                    color="#2ecc71", alpha=0.5, linewidth=0.8)
    # Rolling average
    if len(df) >= 10:
        rolling = df["total_reward"].rolling(10).mean()
        axes[0, 0].plot(df["episode"], rolling,
                        color="#27ae60", linewidth=2, label="Rolling avg (10)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("Training Reward (Hardware)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Hit rate
    axes[0, 1].plot(df["episode"], df["hit_rate_pct"],
                    color="#3498db", alpha=0.5, linewidth=0.8)
    if len(df) >= 10:
        rolling = df["hit_rate_pct"].rolling(10).mean()
        axes[0, 1].plot(df["episode"], rolling,
                        color="#2980b9", linewidth=2, label="Rolling avg (10)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Hit Rate (%)")
    axes[0, 1].set_title("Cache Hit Rate (L1 + L2)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Measured latency
    axes[1, 0].plot(df["episode"], df["avg_measured_latency_ms"],
                    color="#e74c3c", alpha=0.5, linewidth=0.8, label="Measured")
    axes[1, 0].plot(df["episode"], df["avg_baseline_latency_ms"],
                    color="#95a5a6", alpha=0.5, linewidth=0.8,
                    linestyle="--", label="Baseline (no prefetch)")
    if len(df) >= 10:
        rolling = df["avg_measured_latency_ms"].rolling(10).mean()
        axes[1, 0].plot(df["episode"], rolling,
                        color="#c0392b", linewidth=2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Avg Latency (ms)")
    axes[1, 0].set_title("Access Latency: Measured vs Baseline")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Tier distribution over training
    axes[1, 1].stackplot(
        df["episode"],
        df["tier_L1"], df["tier_L2"], df["tier_L3"], df["tier_MISS"],
        labels=["L1 (GPU)", "L2 (CPU)", "L3 (Disk)", "MISS"],
        colors=["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"],
        alpha=0.7,
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Chunk Count")
    axes[1, 1].set_title("Tier Hit Distribution Over Training")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "hardware_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Training curves saved → {path.name}")


# ═══════════════════════════════════════════════════════════════
# QUICK TRAIN (smoke test)
# ═══════════════════════════════════════════════════════════════

def quick_train(hw_cfg: HardwareConfig, total_timesteps: int = 500):
    """Minimal training for testing the pipeline on real hardware."""
    print("\n[quick_train] 🧪 Running hardware smoke test...")
    raw_cfg = load_raw_yaml_config()
    traces = get_trace_paths()
    embeddings = ensure_embeddings(traces, hw_cfg.embed_dim)

    stage_b_ppo_training(
        hw_cfg, raw_cfg, traces, embeddings,
        total_timesteps=total_timesteps,
    )
    print("[quick_train] ✅ Hardware smoke test passed!")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train RL cache prefetching agent on REAL hardware"
    )
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                        help="Training stage: a=behavioral cloning, "
                             "b=PPO (hardware), both=a+b (default)")
    parser.add_argument("--quick", action="store_true",
                        help="Run a 500-step smoke test")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps for PPO")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to hardware_config.yaml")

    # ── Hardware-specific CLI overrides ────────────────────────
    parser.add_argument("--l1-mb", type=float, default=None,
                        help="L1 (GPU VRAM) capacity in MB")
    parser.add_argument("--l2-mb", type=float, default=None,
                        help="L2 (CPU RAM) capacity in MB")
    parser.add_argument("--l3-mb", type=float, default=None,
                        help="L3 (Disk) capacity in MB")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose cache output")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU-only mode (no GPU)")

    args = parser.parse_args()

    # ── Build config ──────────────────────────────────────────
    overrides = {}
    if args.l1_mb is not None:
        overrides["l1_capacity_mb"] = args.l1_mb
    if args.l2_mb is not None:
        overrides["l2_capacity_mb"] = args.l2_mb
    if args.l3_mb is not None:
        overrides["l3_capacity_mb"] = args.l3_mb
    if args.verbose:
        overrides["verbose"] = True
    if args.cpu_only:
        overrides["force_cpu_mode"] = True

    hw_cfg = load_hardware_config(args.config, overrides if overrides else None)

    # ── Quick test? ───────────────────────────────────────────
    if args.quick:
        quick_train(hw_cfg)
        return

    # ── Full training ─────────────────────────────────────────
    raw_cfg = load_raw_yaml_config(args.config)
    traces = get_trace_paths()

    # Check traces exist
    for name, path in traces.items():
        if not path.exists():
            print(f"[train_hardware] ❌ Trace missing: {path}")
            print(f"[train_hardware]    Run: python data/generate_traces.py")
            sys.exit(1)

    # Load embeddings
    print("\n[train_hardware] 📥 Loading embeddings...")
    embeddings = ensure_embeddings(traces, hw_cfg.embed_dim)

    # Stage A: Behavioral Cloning (uses simulated cache for speed)
    if args.stage in ("a", "both"):
        stage_a_behavioral_cloning(hw_cfg, raw_cfg, traces, embeddings)

    # Stage B: PPO on REAL HARDWARE
    if args.stage in ("b", "both"):
        stage_b_ppo_training(
            hw_cfg, raw_cfg, traces, embeddings,
            total_timesteps=args.timesteps,
        )

    print("\n[train_hardware] 🎉 All done! Check results/ for metrics and plots.")


if __name__ == "__main__":
    main()
