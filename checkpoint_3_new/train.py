#!/usr/bin/env python3
"""
Training: Behavioral Cloning → PPO on Real Hardware
=====================================================

Two-stage training for the RL KV cache prefetching agent:

  Stage A - Behavioral Cloning (fast, uses simulated cache):
    Collects (observation, oracle_action) pairs from all 4 traces.
    Oracle = "which candidates will the NEXT query need?"
    Trains a small MLP with BCE loss. Saves bc_pretrained.pt.

  Stage B - PPO Fine-Tuning (REAL GPU/CPU/Disk hardware):
    Creates CachePrefetchEnv with HardwareCache.
    Loads BC-pretrained weights → trains with PPO from stable-baselines3.
    All cache operations happen on real hardware with measured latencies.
    Saves ppo_final.zip + training metrics + plots.

Usage:
    python train.py                     # Full training (Stage A + B)
    python train.py --quick             # 500-step smoke test
    python train.py --stage a           # Stage A only (behavioral cloning)
    python train.py --stage b           # Stage B only (PPO, assumes BC done)
    python train.py --l1-mb 24 --l2-mb 32   # Custom cache sizes
    python train.py --timesteps 10000   # Override timestep count
    python train.py --cpu-only          # Test without GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ═══════════════════════════════════════════════════════════════
# CONFIG LOADING
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml."""
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Apply CLI argument overrides to config dict."""
    if args.l1_mb is not None:
        print(f"[config] Override: l1_capacity_mb = {cfg.get('l1_capacity_mb')} → {args.l1_mb}")
        cfg["l1_capacity_mb"] = args.l1_mb
    if args.l2_mb is not None:
        print(f"[config] Override: l2_capacity_mb = {cfg.get('l2_capacity_mb')} → {args.l2_mb}")
        cfg["l2_capacity_mb"] = args.l2_mb
    if args.l3_mb is not None:
        print(f"[config] Override: l3_capacity_mb = {cfg.get('l3_capacity_mb')} → {args.l3_mb}")
        cfg["l3_capacity_mb"] = args.l3_mb
    if args.cpu_only:
        cfg["force_cpu_mode"] = True
    if args.timesteps is not None:
        cfg["total_timesteps"] = args.timesteps
    return cfg


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def get_trace_paths() -> dict:
    """Return {workload_name: trace_csv_path}."""
    return {
        "prefix":    DATA_DIR / "traces_prefix.csv",
        "rag":       DATA_DIR / "traces_rag.csv",
        "nocontext": DATA_DIR / "traces_nocontext.csv",
        "multiturn": DATA_DIR / "traces_multiturn.csv",
    }


def load_embeddings(embed_dim: int = 384) -> dict:
    """Load pre-computed embeddings for all workloads."""
    embeddings = {}
    for name in ["prefix", "rag", "nocontext", "multiturn"]:
        path = DATA_DIR / f"embeddings_{name}.npy"
        if path.exists():
            embs = np.load(str(path))
            print(f"  Loaded embeddings_{name}.npy: {embs.shape}")
            embeddings[name] = embs
        else:
            print(f"  [WARN] {path.name} not found. Using zero embeddings.")
            # Count queries in trace to know how many zero embeddings to create
            import pandas as pd
            trace_path = DATA_DIR / f"traces_{name}.csv"
            if trace_path.exists():
                n = len(pd.read_csv(trace_path))
                embeddings[name] = np.zeros((n, embed_dim), dtype=np.float32)
            else:
                print(f"  [WARN] Trace {trace_path.name} also not found. Skipping {name}.")
    return embeddings


# ═══════════════════════════════════════════════════════════════
# STAGE A: BEHAVIORAL CLONING
# ═══════════════════════════════════════════════════════════════

def stage_a_behavioral_cloning(cfg: dict) -> Optional[nn.Sequential]:
    """
    Pre-train a policy network by imitating oracle prefetch decisions.

    Uses SimulatedCache (fast, no GPU tensors needed) because oracle
    labels don't depend on real latencies — they're just "which chunks
    will the NEXT query need?"
    """
    import pandas as pd
    from environment import CachePrefetchEnv, SimulatedCache

    print("\n" + "=" * 64)
    print("  STAGE A: Behavioral Cloning (Oracle Imitation)")
    print("  Using: SimulatedCache (fast — oracle labels are the same")
    print("         regardless of real vs simulated latencies)")
    print("=" * 64)

    # Config values
    embed_dim = cfg.get("embed_dim", 384)
    max_candidates = cfg.get("max_candidates", 16)
    history_len = cfg.get("history_len", 5)
    chunk_size_bytes = cfg.get("chunk_size_bytes", 3_145_728)
    obs_dim = embed_dim + 3 + max_candidates

    traces = get_trace_paths()
    embeddings = load_embeddings(embed_dim)

    # Create simulated cache matching our config
    sim_cache_template = lambda: SimulatedCache(
        l1_cap=int(cfg.get("l1_capacity_mb", 48.0) * 1024 * 1024),
        l2_cap=int(cfg.get("l2_capacity_mb", 51.2) * 1024 * 1024),
        l3_cap=int(cfg.get("l3_capacity_mb", 5120.0) * 1024 * 1024),
        chunk_size=chunk_size_bytes,
        l1_lat=cfg.get("l1_hit_latency_ms", 0.1),
        l2_lat=cfg.get("l2_hit_latency_ms", 0.24),
        l3_lat=cfg.get("l3_hit_latency_ms", 6.0),
        cold_lat=cfg.get("cold_compute_per_chunk_ms", 30.8),
        prefetch_lat=cfg.get("prefetch_l3_to_l2_ms", 6.0),
    )

    # Collect (observation, oracle_action) pairs from all traces
    all_obs = []
    all_actions = []

    for name, trace_path in traces.items():
        if not trace_path.exists() or name not in embeddings:
            continue

        df = pd.read_csv(trace_path)
        embs = embeddings[name]

        sim_cache = sim_cache_template()
        env = CachePrefetchEnv(
            trace_path=trace_path,
            cache=sim_cache,
            query_embeddings=embs,
            embed_dim=embed_dim,
            max_candidates=max_candidates,
            history_len=history_len,
            chunk_size_bytes=chunk_size_bytes,
            alpha=cfg.get("alpha", 1.0),
            beta=cfg.get("beta", 0.01),
            gamma_reward=cfg.get("gamma_reward", 0.1),
            l1_hit_latency_ms=cfg.get("l1_hit_latency_ms", 0.1),
            l2_hit_latency_ms=cfg.get("l2_hit_latency_ms", 0.24),
            l3_hit_latency_ms=cfg.get("l3_hit_latency_ms", 6.0),
            cold_compute_per_chunk_ms=cfg.get("cold_compute_per_chunk_ms", 30.8),
        )

        obs, _ = env.reset()

        for step_idx in range(len(df) - 1):
            # Oracle: peek at what the NEXT query will need
            next_chunks = set(json.loads(df.iloc[step_idx + 1]["chunk_ids_needed"]))

            # Oracle action: prefetch candidates that match next query's needs
            oracle_action = np.zeros(max_candidates, dtype=np.float32)
            for i, cid in enumerate(env.candidate_ids):
                if cid in next_chunks:
                    oracle_action[i] = 1.0

            all_obs.append(obs.copy())
            all_actions.append(oracle_action.copy())

            # Step with oracle action to advance the environment
            obs, _, terminated, _, _ = env.step(oracle_action.astype(int))
            if terminated:
                break

        print(f"  {name}: collected {len(all_obs)} samples so far")

    if not all_obs:
        print("  [WARN] No training data generated. Skipping Stage A.")
        return None

    X = torch.tensor(np.array(all_obs), dtype=torch.float32)
    Y = torch.tensor(np.array(all_actions), dtype=torch.float32)
    print(f"\n  Training data: {X.shape[0]} samples, obs_dim={X.shape[1]}")

    # Build MLP matching PPO policy architecture
    net_arch = cfg.get("policy_net_arch", [64, 64])
    layers = []
    in_dim = obs_dim
    for hidden in net_arch:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, max_candidates))
    layers.append(nn.Sigmoid())
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("bc_lr", 0.001))
    criterion = nn.BCELoss()

    bc_epochs = cfg.get("bc_epochs", 20)
    batch_size = cfg.get("batch_size", 64)

    for epoch in range(bc_epochs):
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X[idx])
            loss = criterion(pred, Y[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{bc_epochs} — BCE loss: {avg_loss:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "bc_pretrained.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")

    return model


# ═══════════════════════════════════════════════════════════════
# STAGE B: PPO FINE-TUNING (REAL HARDWARE!)
# ═══════════════════════════════════════════════════════════════

def stage_b_ppo_training(cfg: dict):
    """
    Fine-tune the policy using PPO against REAL hardware cache.

    Every cache operation runs on real GPU/CPU/Disk with measured latencies.
    The agent learns to minimize real data movement costs through smart prefetching.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from cache import HardwareCache
    from environment import CachePrefetchEnv

    print("\n" + "=" * 64)
    print("  STAGE B: PPO Fine-Tuning (REAL HARDWARE)")
    print("  Using: HardwareCache with actual GPU/CPU/Disk")
    print("=" * 64)

    # Config values
    embed_dim = cfg.get("embed_dim", 384)
    max_candidates = cfg.get("max_candidates", 16)
    chunk_size_bytes = cfg.get("chunk_size_bytes", 3_145_728)
    total_timesteps = cfg.get("total_timesteps", 50_000)

    # Load embeddings
    embeddings = load_embeddings(embed_dim)

    # Use RAG trace for training (highest impact from prefetching)
    rag_trace = DATA_DIR / "traces_rag.csv"
    if not rag_trace.exists():
        print(f"  ERROR: {rag_trace} not found!")
        print(f"  Run: python generate_traces.py --skip-llm")
        sys.exit(1)

    rag_embs = embeddings.get("rag")
    if rag_embs is None:
        print(f"  ERROR: RAG embeddings not found!")
        sys.exit(1)

    # Create REAL hardware cache
    hw_cache = HardwareCache(
        chunk_size_bytes=chunk_size_bytes,
        l1_capacity_mb=cfg.get("l1_capacity_mb", 48.0),
        l2_capacity_mb=cfg.get("l2_capacity_mb", 51.2),
        l3_capacity_mb=cfg.get("l3_capacity_mb", 5120.0),
        l3_disk_dir=cfg.get("l3_disk_dir", "./data/cache_store"),
        force_cpu_mode=cfg.get("force_cpu_mode", False),
        cuda_warmup_iterations=cfg.get("cuda_warmup_iterations", 5),
        cold_compute_ms=cfg.get("cold_compute_per_chunk_ms", 30.8),
    )

    # Create environment
    env = CachePrefetchEnv(
        trace_path=rag_trace,
        cache=hw_cache,
        query_embeddings=rag_embs,
        embed_dim=embed_dim,
        max_candidates=max_candidates,
        history_len=cfg.get("history_len", 5),
        chunk_size_bytes=chunk_size_bytes,
        alpha=cfg.get("alpha", 1.0),
        beta=cfg.get("beta", 0.01),
        gamma_reward=cfg.get("gamma_reward", 0.1),
        l1_hit_latency_ms=cfg.get("l1_hit_latency_ms", 0.1),
        l2_hit_latency_ms=cfg.get("l2_hit_latency_ms", 0.24),
        l3_hit_latency_ms=cfg.get("l3_hit_latency_ms", 6.0),
        cold_compute_per_chunk_ms=cfg.get("cold_compute_per_chunk_ms", 30.8),
    )

    print(f"\n  Environment:")
    print(f"    Trace: traces_rag.csv ({env.n_queries} queries)")
    print(f"    L1 (GPU): {cfg.get('l1_capacity_mb', 48.0):.1f} MB "
          f"({hw_cache.l1_capacity_chunks} chunks)")
    print(f"    L2 (CPU): {cfg.get('l2_capacity_mb', 51.2):.1f} MB "
          f"({hw_cache.l2_capacity_chunks} chunks)")
    print(f"    Hardware: {'CUDA GPU' if hw_cache.use_cuda else 'CPU-only'}")

    # ── Training callback ─────────────────────────────────────────

    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_metrics = []
            self.episode_count = 0

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode_summary" in info:
                    s = info["episode_summary"]
                    self.episode_count += 1
                    self.episode_rewards.append(s["total_reward"])

                    tc = s.get("tier_counts", {})
                    tl = s.get("tier_latencies_ms", {})
                    total_accesses = sum(tc.values()) or 1

                    self.episode_metrics.append({
                        "episode": self.episode_count,
                        "total_reward": s["total_reward"],
                        "hit_rate_pct": s["hit_rate_pct"],
                        "prefetch_accuracy_pct": s["prefetch_accuracy_pct"],
                        "total_prefetches": s["total_prefetches"],
                        "useful_prefetches": s["useful_prefetches"],
                        "avg_measured_latency_ms": s.get("avg_measured_latency_ms", 0),
                        "avg_baseline_latency_ms": s.get("avg_baseline_latency_ms", 0),
                        "total_measured_latency_ms": s.get("total_measured_latency_ms", 0),
                        "total_baseline_latency_ms": s.get("total_baseline_latency_ms", 0),
                        "latency_reduction_pct": s.get("latency_reduction_pct", 0),
                        # TTFT metrics
                        "avg_ttft_ms": s.get("avg_ttft_ms", 0),
                        "median_ttft_ms": s.get("median_ttft_ms", 0),
                        "p95_ttft_ms": s.get("p95_ttft_ms", 0),
                        # Tier counts
                        "tier_L1": tc.get("L1", 0),
                        "tier_L2": tc.get("L2", 0),
                        "tier_L3": tc.get("L3", 0),
                        "tier_MISS": tc.get("MISS", 0),
                        # Tier percentages
                        "tier_L1_pct": round(tc.get("L1", 0) / total_accesses * 100, 1),
                        "tier_L2_pct": round(tc.get("L2", 0) / total_accesses * 100, 1),
                        "tier_L3_pct": round(tc.get("L3", 0) / total_accesses * 100, 1),
                        "tier_MISS_pct": round(tc.get("MISS", 0) / total_accesses * 100, 1),
                        # Tier latencies
                        "latency_L1_ms": tl.get("L1", 0),
                        "latency_L2_ms": tl.get("L2", 0),
                        "latency_L3_ms": tl.get("L3", 0),
                        "latency_MISS_ms": tl.get("MISS", 0),
                    })

                    if self.episode_count % 10 == 0:
                        recent = self.episode_rewards[-10:]
                        avg = sum(recent) / len(recent)
                        print(f"  Episode {self.episode_count:4d} | "
                              f"Avg reward(10): {avg:7.2f} | "
                              f"Hit rate: {s['hit_rate_pct']:5.1f}% | "
                              f"Prefetch acc: {s['prefetch_accuracy_pct']:5.1f}% | "
                              f"Avg TTFT: {s.get('avg_ttft_ms', 0):.2f}ms | "
                              f"Lat reduction: {s.get('latency_reduction_pct', 0):.1f}%")
            return True

    logger = RewardLogger()

    # ── PPO model ─────────────────────────────────────────────────

    net_arch = cfg.get("policy_net_arch", [64, 64])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps=min(cfg.get("n_steps", 2048), total_timesteps),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma_discount", 0.99),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        vf_coef=cfg.get("vf_coef", 0.5),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        policy_kwargs={"net_arch": net_arch},
        device="cpu",  # PPO MLP on CPU; GPU reserved for cache data
        verbose=0,
    )

    # ── Load BC pre-trained weights ───────────────────────────────
    # CRITICAL: Without this, Stage A is wasted — PPO starts from scratch!
    # Map BC Sequential layers → SB3's internal actor network structure:
    #   BC: Linear(403→64) → ReLU → Linear(64→64) → ReLU → Linear(64→16) → Sigmoid
    #   SB3: mlp_extractor.policy_net.0 = Linear(403→64)
    #        mlp_extractor.policy_net.2 = Linear(64→64)
    #        action_net = Linear(64→16)

    bc_path = MODELS_DIR / "bc_pretrained.pt"
    if bc_path.exists():
        try:
            bc_state = torch.load(bc_path, map_location="cpu", weights_only=True)
            ppo_policy = model.policy

            # BC layer indices → SB3 parameter names
            # BC Sequential: [0]=Linear, [1]=ReLU, [2]=Linear, [3]=ReLU, [4]=Linear, [5]=Sigmoid
            # SB3 policy_net: [0]=Linear, [1]=ReLU, [2]=Linear, [3]=ReLU
            # SB3 action_net: Linear (output layer)
            mapping = {
                "0.weight": "mlp_extractor.policy_net.0.weight",
                "0.bias":   "mlp_extractor.policy_net.0.bias",
                "2.weight": "mlp_extractor.policy_net.2.weight",
                "2.bias":   "mlp_extractor.policy_net.2.bias",
                "4.weight": "action_net.weight",
                "4.bias":   "action_net.bias",
            }

            loaded_count = 0
            for bc_key, sb3_key in mapping.items():
                if bc_key in bc_state:
                    param = dict(ppo_policy.named_parameters()).get(sb3_key)
                    if param is not None and param.shape == bc_state[bc_key].shape:
                        param.data.copy_(bc_state[bc_key])
                        loaded_count += 1

            print(f"  BC weights loaded: {loaded_count}/{len(mapping)} parameters transferred")
            if loaded_count < len(mapping):
                print(f"  [WARN] Some BC weights could not be transferred (shape mismatch?)")
        except Exception as e:
            print(f"  [WARN] Could not load BC weights: {e}")
            print(f"  PPO will start from random initialization.")
    else:
        print(f"  [INFO] No BC weights found at {bc_path}. Starting from scratch.")

    # ── Train ─────────────────────────────────────────────────────

    print(f"\n  Starting PPO: {total_timesteps} timesteps")
    print(f"  (This will be SLOWER than simulated training because")
    print(f"   every step involves REAL disk I/O and GPU transfers)\n")

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=logger)
    elapsed = time.time() - t_start

    print(f"\n  Training complete in {elapsed:.1f}s "
          f"({total_timesteps / elapsed:.0f} steps/sec)")

    # ── Save model ────────────────────────────────────────────────

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "ppo_final"
    model.save(str(save_path))
    print(f"  Model saved → {save_path}.zip")

    # ── Save training metrics ─────────────────────────────────────

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "training_rewards.json", "w") as f:
        json.dump(logger.episode_rewards, f, indent=2)

    if logger.episode_metrics:
        import pandas as pd
        pd.DataFrame(logger.episode_metrics).to_csv(
            RESULTS_DIR / "training_metrics.csv", index=False)
        print(f"  Training metrics → training_metrics.csv")

        # Print summary
        print(f"\n  Training Summary:")
        print(f"    Episodes:          {logger.episode_count}")
        print(f"    Final avg reward:  {np.mean(logger.episode_rewards[-10:]):.2f}")
        print(f"    Best reward:       {max(logger.episode_rewards):.2f}")
        print(f"    Final hit rate:    {logger.episode_metrics[-1]['hit_rate_pct']:.1f}%")
        print(f"    Final pf accuracy: {logger.episode_metrics[-1]['prefetch_accuracy_pct']:.1f}%")

    # ── Training plots ────────────────────────────────────────────

    _plot_training_curves(logger)


def _plot_training_curves(logger):
    """Generate training progress plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("  [WARN] matplotlib not available. Skipping plots.")
        return

    if not logger.episode_metrics:
        return

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(logger.episode_metrics)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # 1. Reward
    axes[0, 0].plot(df["episode"], df["total_reward"],
                    color="#2ecc71", alpha=0.4, linewidth=0.8)
    if len(df) >= 10:
        axes[0, 0].plot(df["episode"], df["total_reward"].rolling(10).mean(),
                        color="#27ae60", linewidth=2, label="Rolling avg (10)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("Training Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Hit rate
    axes[0, 1].plot(df["episode"], df["hit_rate_pct"],
                    color="#3498db", alpha=0.4, linewidth=0.8)
    if len(df) >= 10:
        axes[0, 1].plot(df["episode"], df["hit_rate_pct"].rolling(10).mean(),
                        color="#2980b9", linewidth=2, label="Rolling avg (10)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Hit Rate (%)")
    axes[0, 1].set_title("Cache Hit Rate (L1 + L2)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Latency
    axes[1, 0].plot(df["episode"], df["avg_measured_latency_ms"],
                    color="#e74c3c", alpha=0.4, linewidth=0.8, label="Measured")
    axes[1, 0].plot(df["episode"], df["avg_baseline_latency_ms"],
                    color="#95a5a6", alpha=0.4, linewidth=0.8,
                    linestyle="--", label="Baseline")
    if len(df) >= 10:
        axes[1, 0].plot(df["episode"],
                        df["avg_measured_latency_ms"].rolling(10).mean(),
                        color="#c0392b", linewidth=2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Avg Latency (ms)")
    axes[1, 0].set_title("Measured vs Baseline Latency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Tier distribution
    axes[1, 1].stackplot(
        df["episode"],
        df["tier_L1"], df["tier_L2"], df["tier_L3"], df["tier_MISS"],
        labels=["L1 (GPU)", "L2 (CPU)", "L3 (Disk)", "MISS"],
        colors=["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"],
        alpha=0.7,
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Chunk Count")
    axes[1, 1].set_title("Tier Hit Distribution")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Estimated TTFT
    if "avg_ttft_ms" in df.columns:
        axes[2, 0].plot(df["episode"], df["avg_ttft_ms"],
                        color="#9b59b6", alpha=0.4, linewidth=0.8)
        if len(df) >= 10:
            axes[2, 0].plot(df["episode"], df["avg_ttft_ms"].rolling(10).mean(),
                            color="#8e44ad", linewidth=2, label="Rolling avg (10)")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("Avg TTFT (ms)")
        axes[2, 0].set_title("Estimated Time To First Token")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # 6. Prefetch accuracy
    if "prefetch_accuracy_pct" in df.columns:
        axes[2, 1].plot(df["episode"], df["prefetch_accuracy_pct"],
                        color="#f39c12", alpha=0.4, linewidth=0.8)
        if len(df) >= 10:
            axes[2, 1].plot(df["episode"],
                            df["prefetch_accuracy_pct"].rolling(10).mean(),
                            color="#e67e22", linewidth=2, label="Rolling avg (10)")
        axes[2, 1].set_xlabel("Episode")
        axes[2, 1].set_ylabel("Prefetch Accuracy (%)")
        axes[2, 1].set_title("Prefetch Accuracy (useful / total)")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves → {path.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train RL cache prefetching agent"
    )
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                        help="a=behavioral cloning, b=PPO (hardware), both=a+b")
    parser.add_argument("--quick", action="store_true",
                        help="500-step smoke test")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--l1-mb", type=float, default=None,
                        help="Override L1 (GPU VRAM) capacity in MB")
    parser.add_argument("--l2-mb", type=float, default=None,
                        help="Override L2 (CPU RAM) capacity in MB")
    parser.add_argument("--l3-mb", type=float, default=None,
                        help="Override L3 (Disk) capacity in MB")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU-only mode (no GPU)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.quick:
        cfg["total_timesteps"] = 500
        cfg["bc_epochs"] = 3
        print("\n[quick mode] 500 timesteps, 3 BC epochs")

    # Check traces exist
    missing = False
    for name, path in get_trace_paths().items():
        if not path.exists():
            print(f"  ERROR: Missing trace: {path}")
            missing = True
    if missing:
        print("  Run: python generate_traces.py --skip-llm")
        sys.exit(1)

    # Stage A
    if args.stage in ("a", "both"):
        stage_a_behavioral_cloning(cfg)

    # Stage B
    if args.stage in ("b", "both"):
        stage_b_ppo_training(cfg)

    print("\n Done! Check results/ for metrics and plots.")


if __name__ == "__main__":
    main()
