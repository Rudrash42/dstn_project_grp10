#!/usr/bin/env python3
"""
Main training entrypoint for the RL cache prefetching agent.

Two-stage training:
  Stage A — Behavioral Cloning: pre-train from oracle labels
  Stage B — PPO Fine-tuning: train against the cache simulator

Usage:
    python train.py                  # full training
    python train.py --quick          # 500-step smoke test
    python train.py --stage a        # behavioral cloning only
    python train.py --stage b        # PPO only (requires pretrained weights)
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from env.tier_config import TierConfig
from env.cache_env import CacheEnv
from agent.state_encoder import StateEncoder

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_config() -> dict:
    """Load ppo_config.yaml."""
    cfg_path = PROJECT_ROOT / "configs" / "ppo_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_tier_config() -> TierConfig:
    """Create TierConfig from the YAML."""
    cfg_path = PROJECT_ROOT / "configs" / "ppo_config.yaml"
    return TierConfig.from_yaml(cfg_path)


def get_trace_paths() -> dict:
    """Return dict of {workload_name: trace_csv_path}."""
    data_dir = PROJECT_ROOT / "data"
    return {
        "prefix": data_dir / "traces_prefix.csv",
        "rag": data_dir / "traces_rag.csv",
        "nocontext": data_dir / "traces_nocontext.csv",
        "multiturn": data_dir / "traces_multiturn.csv",
    }


def ensure_embeddings(traces: dict, tier_cfg: TierConfig) -> dict:
    """
    Pre-compute or load embeddings for all traces.
    Returns dict of {workload: np.ndarray of shape (n, embed_dim)}.
    """
    data_dir = PROJECT_ROOT / "data"
    encoder = StateEncoder(embed_dim=tier_cfg.embed_dim)

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

def stage_a_behavioral_cloning(
    tier_cfg: TierConfig,
    raw_cfg: dict,
    traces: dict,
    embeddings: dict,
):
    """
    Pre-train a policy network by imitating oracle prefetch decisions.
    The oracle label is: "prefetch any candidate chunk that will be
    accessed by the *next* query."
    """
    print("\n" + "=" * 60)
    print("STAGE A: Behavioral Cloning (Oracle Imitation)")
    print("=" * 60)

    import pandas as pd

    # Collect (observation, oracle_action) pairs from all traces
    all_obs = []
    all_actions = []

    for name, trace_path in traces.items():
        df = pd.read_csv(trace_path)
        embs = embeddings[name]
        env = CacheEnv(trace_path, config=tier_cfg, query_embeddings=embs)

        obs, _ = env.reset()

        for step_idx in range(len(df) - 1):
            # Get next query's needed chunks (oracle knowledge)
            next_chunks = set(json.loads(df.iloc[step_idx + 1]["chunk_ids_needed"]))

            # Compute oracle action: prefetch candidates that match next query
            oracle_action = np.zeros(tier_cfg.max_candidate_chunks, dtype=np.float32)
            for i, cid in enumerate(env.candidate_ids):
                if cid in next_chunks:
                    oracle_action[i] = 1.0

            all_obs.append(obs.copy())
            all_actions.append(oracle_action.copy())

            # Step with oracle action to advance the env
            obs, _, terminated, _, _ = env.step(oracle_action.astype(int))
            if terminated:
                break

    if not all_obs:
        print("  [WARN] No training data generated. Skipping Stage A.")
        return None

    X = torch.tensor(np.array(all_obs), dtype=torch.float32)
    Y = torch.tensor(np.array(all_actions), dtype=torch.float32)

    print(f"  Training data: {X.shape[0]} samples, obs_dim={X.shape[1]}")

    # Simple MLP matching the PPO policy architecture
    net_arch = raw_cfg.get("policy_net_arch", [64, 64])
    layers = []
    in_dim = tier_cfg.obs_dim
    for hidden in net_arch:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, tier_cfg.max_candidate_chunks))
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
    save_path = PROJECT_ROOT / "models" / "bc_pretrained.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")

    return model


# ═══════════════════════════════════════════════════════════════
# STAGE B: PPO FINE-TUNING
# ═══════════════════════════════════════════════════════════════

def stage_b_ppo_training(
    tier_cfg: TierConfig,
    raw_cfg: dict,
    traces: dict,
    embeddings: dict,
    total_timesteps: Optional[int] = None,
):
    """
    Fine-tune the policy using PPO against the cache simulator.
    Trains on the RAG trace (highest impact) by default.
    """
    print("\n" + "=" * 60)
    print("STAGE B: PPO Fine-Tuning")
    print("=" * 60)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from agent.policy import make_policy_kwargs

    # Use RAG trace for PPO (most reward signal from prefetching)
    rag_trace = traces["rag"]
    rag_embs = embeddings["rag"]

    env = CacheEnv(rag_trace, config=tier_cfg, query_embeddings=rag_embs)

    # PPO hyperparameters from config
    if total_timesteps is None:
        total_timesteps = raw_cfg.get("total_timesteps", 50_000)

    net_arch = raw_cfg.get("policy_net_arch", [64, 64])

    class RewardLogger(BaseCallback):
        """Log episode rewards during training."""
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_count = 0

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode_summary" in info:
                    summary = info["episode_summary"]
                    self.episode_count += 1
                    self.episode_rewards.append(summary["total_reward"])
                    if self.episode_count % 10 == 0:
                        recent = self.episode_rewards[-10:]
                        avg = sum(recent) / len(recent)
                        hr = summary["hit_rate_pct"]
                        print(
                            f"  Episode {self.episode_count:4d} | "
                            f"Avg reward(10): {avg:7.2f} | "
                            f"Hit rate: {hr:5.1f}% | "
                            f"Prefetch acc: {summary['prefetch_accuracy_pct']:5.1f}%"
                        )
            return True

    logger = RewardLogger()

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
        device="cpu",  # MLP policies run faster on CPU; saves GPU for the LLM
        verbose=0,
    )

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=logger)
    elapsed = time.time() - t_start

    print(f"\n  Training complete in {elapsed:.1f}s ({total_timesteps} timesteps)")

    # Save model
    save_path = PROJECT_ROOT / "models" / "ppo_final"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"  Saved → {save_path}.zip")

    # Save training curves
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_rewards.json", "w") as f:
        json.dump(logger.episode_rewards, f)

    return model


# ═══════════════════════════════════════════════════════════════
# QUICK TRAIN (for smoke testing)
# ═══════════════════════════════════════════════════════════════

def quick_train(total_timesteps: int = 500):
    """Minimal training for testing the pipeline."""
    print("\n[quick_train] Running minimal smoke test...")
    tier_cfg = get_tier_config()
    raw_cfg = load_config()
    traces = get_trace_paths()
    embeddings = ensure_embeddings(traces, tier_cfg)
    stage_b_ppo_training(tier_cfg, raw_cfg, traces, embeddings,
                         total_timesteps=total_timesteps)
    print("[quick_train] Smoke test passed!")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train RL cache prefetching agent")
    parser.add_argument("--stage", choices=["a", "b", "both"], default="both",
                        help="Training stage: a=behavioral cloning, "
                             "b=PPO, both=a+b (default)")
    parser.add_argument("--quick", action="store_true",
                        help="Run a 500-step smoke test")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps for PPO")
    args = parser.parse_args()

    if args.quick:
        quick_train()
        return

    tier_cfg = get_tier_config()
    raw_cfg = load_config()
    traces = get_trace_paths()

    # Generate traces if they don't exist
    for name, path in traces.items():
        if not path.exists():
            print(f"[train] Traces missing. Run: python data/generate_traces.py")
            from data.generate_traces import main as gen_main
            gen_main()
            break

    # Pre-compute embeddings
    print("\n[train] Ensuring embeddings are ready...")
    embeddings = ensure_embeddings(traces, tier_cfg)

    if args.stage in ("a", "both"):
        stage_a_behavioral_cloning(tier_cfg, raw_cfg, traces, embeddings)

    if args.stage in ("b", "both"):
        stage_b_ppo_training(tier_cfg, raw_cfg, traces, embeddings,
                             total_timesteps=args.timesteps)

    print("\n[train] All done! 🎉")


if __name__ == "__main__":
    main()
