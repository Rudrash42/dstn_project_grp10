# RL-Based KV Cache Prefetching (Hardware-Only)

Proactive **PPO-trained RL agent** that predicts which KV-cache chunks will be needed and pre-migrates them from L3 (NVMe Disk) → L2 (CPU RAM) before they're requested. Uses **real GPU/CPU/Disk hardware** for training and evaluation — no vLLM or LMCache required.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate traces (calibrates real hardware + builds workload traces)
python data/generate_traces.py

# 3. Train the agent on real hardware
python train_hardware.py

# 4. Evaluate against baselines on real hardware
python eval/evaluate_hardware.py

# Or run the full pipeline in one go:
bash run_pipeline.sh
```

## Smoke Test

```bash
python train_hardware.py --quick      # 500-step hardware smoke test (< 5 min)
python data/generate_traces.py --cpu-only  # Generate traces without GPU
```

## Runtime Trace Standardization (LMCache Cold-Pass)

If you already generated runtime-collected traces under `data/runs/<timestamp>/`
with `chunk_event_source=lmcache_store_coldpass`, promote the latest run into the
canonical `data/traces_*.csv` files used by training/evaluation:

```bash
# Promote latest runtime run
python data/promote_runtime_traces.py

# Or pick a specific run id
python data/promote_runtime_traces.py --run-id 20260423T172836Z

# See available runs
python data/promote_runtime_traces.py --list
```

This keeps your pipeline on runtime-provenance traces instead of synthetic chunk
assignments whenever run artifacts are available.

## Architecture

```
Agent observes: [query embedding (384d)] + [cache stats (3d)] + [candidate scores (16d)]
Agent decides:  which of 16 L3 candidate chunks to prefetch → MultiBinary(16)
Reward:         R = α·time_saved − β·bytes_migrated − γ·unused_prefetches
Hardware:       L1 = GPU VRAM | L2 = CPU Pinned RAM | L3 = NVMe Disk
```

## Training Pipeline

1. **Stage A** — Behavioral cloning from oracle labels (simulated for speed)
2. **Stage B** — PPO fine-tuning on **real hardware cache** (actual GPU/CPU/Disk data movement)

## Evaluation

Compares against:
- **No Cache** — every access is a cold miss (worst case)
- **LRU** — reactive LRU caching, no prefetching (baseline)
- **Oracle** — perfect future knowledge (theoretical best)

All evaluations run on **real hardware** with measured latencies.

## Directory Structure

```
data/               Trace datasets + embeddings + calibration report
env/                Cache simulator + Gymnasium env + reward function
env/hardware/       REAL hardware cache (GPU VRAM / CPU RAM / NVMe Disk)
agent/              State encoder + policy network
eval/               Evaluation, baselines, plotting (hardware)
configs/            Hardware config + PPO hyperparameters
models/             Saved checkpoints
results/            Metrics + plots (auto-generated)
```
