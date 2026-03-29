# RL-Based KV Cache Prefetching for LMCache

Replaces LMCache's reactive fetching with a proactive **PPO-trained RL agent** that predicts which KV-cache chunks will be needed and pre-migrates them from L3 (disk) → L2 (CPU) before they're requested.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate trace data
python data/generate_traces.py

# 3. Train the agent (full pipeline, ~10 min)
python train.py

# 4. Evaluate against baselines
python eval/evaluate.py

# 5. Generate plots
python eval/plot_results.py
```

## Smoke Test

```bash
python train.py --quick      # 500-step test (< 1 min)
```

## Architecture

```
Agent observes: [query embedding (384d)] + [cache stats (3d)] + [candidate scores (16d)]
Agent decides:  which of 16 L3 candidate chunks to prefetch → MultiBinary(16)
Reward:         R = α·time_saved − β·bytes_migrated − γ·unused_prefetches
```

## Training Pipeline

1. **Stage A** — Behavioral cloning from oracle labels (what *should* have been prefetched)
2. **Stage B** — PPO fine-tuning against the cache simulator (learns cost/benefit tradeoffs)

## Evaluation

Compares against:
- **No Cache** — every access is a cold miss (worst case)
- **LRU** — reactive LRU caching, no prefetching (current system)
- **Oracle** — perfect future knowledge (theoretical best)

## Directory Structure

```
data/               Trace datasets + embeddings
env/                Cache simulator + Gymnasium env + reward function
agent/              State encoder + policy network
eval/               Evaluation, baselines, ablation, plotting
configs/            PPO hyperparameters
models/             Saved checkpoints
results/            Metrics + plots (auto-generated)
```
