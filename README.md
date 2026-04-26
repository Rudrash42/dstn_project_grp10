# Neuro-Tiering: RL-Based KV Cache Prefetching for LMCache
**DSTN Group 10** | BITS Pilani, Goa Campus

Replaces LMCache's reactive LRU caching with a **PPO-trained RL agent** that proactively migrates KV-cache chunks across a three-tier memory hierarchy (GPU VRAM → CPU RAM → NVMe SSD) before they are requested, reducing Time-to-First-Token (TTFT) latency.

---

## Project Layout

```
checkpoint_2/          Milestone 2 – LMCache stress tests & caching comparison plots
checkpoint_3/
  rl_cache_prefetch/   Main RL pipeline (train, evaluate, plot)
checkpoint_3_new/      Experimental model/results snapshots
checkpoint_4/          Stage 4 – extended action space (prefetch + eviction)
report/                IEEE 2-column conference paper (LaTeX + PDF)
presentations/         Slide decks
```

---

## Four-Stage Development Pipeline

| Stage | Description |
|-------|-------------|
| **1 – Simulation** | Software-only 3-tier cache simulator (no real hardware I/O) used for rapid hyperparameter search |
| **2 – Simulated Traces** | PPO agent trained via behavioural cloning + RL on synthetically generated workload traces |
| **3 – Real Hardware Traces** | Agent re-trained on traces collected from actual vLLM + LMCache runs on RTX 3050 hardware |
| **4 – Extended Action Space** | Action expanded to 32 bits: 16 for L3→L2 prefetch + 16 for explicit L2→L3 eviction |

---

## Prerequisites

```bash
# Python 3.10+, CUDA 12.1 (RTX 3050 tested)
python -m venv .venv && source .venv/bin/activate

pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121

cd checkpoint_3/rl_cache_prefetch
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running the Full Pipeline

All commands are run from `checkpoint_3/rl_cache_prefetch/`.

### Step 1 – Benchmark hardware latencies
```bash
python -m env.hardware.benchmark
# Output → results/hardware_benchmark_latencies.csv
```

### Step 2 – Train the agent
```bash
# Quick smoke test (< 2 min)
python train_hardware.py --quick

# Full training (~10–30 min on RTX 3050)
python train_hardware.py

# Custom cache sizes
python train_hardware.py --l1-mb 24 --l2-mb 32
```
Saves model to `models/ppo_hardware_final.zip`.

### Step 3 – Evaluate against baselines
```bash
python eval/evaluate_hardware.py
# Output → results/hardware_metrics.json
#          results/hardware_evaluation.csv
```

### Step 4 – Generate plots
```bash
python eval/plot_results.py
# Output → results/*.png
```

### One-shot convenience script
```bash
chmod +x run_pipeline.sh && ./run_pipeline.sh
```

---

## Key Results (Multi-Turn Workload, RTX 3050)

| Strategy | Avg Latency (ms) | Prefetch Accuracy | Speedup vs LRU |
|----------|-----------------|-------------------|----------------|
| No Cache | 981.6 | – | 0.108× |
| LRU      | 105.8 | – | 1.000× |
| Oracle   | 88.1  | – | 1.200× |
| **RL Agent** | **88.3** | **100%** | **1.198×** |

---

## Report

The full IEEE-format paper is at [`report/report.pdf`](report/report.pdf).  
To rebuild:
```bash
cd report
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
```

---

## Directory Structure (rl_cache_prefetch)

```
agent/          Policy network + state encoder
configs/        PPO & hardware config YAML files
data/           Trace CSVs + pre-computed query embeddings (.npy)
env/
  hardware/     Real GPU/CPU/Disk backends + benchmark script
eval/           Evaluation scripts, baselines (LRU, Oracle, No-Cache), plotting
models/         Saved model checkpoints
results/        Auto-generated metrics and plots
```

---

## Configuration (`configs/hardware_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l1_capacity_mb` | 3 | GPU VRAM budget for KV cache |
| `l2_capacity_mb` | 4.5 | CPU pinned RAM budget |
| `total_timesteps` | 50 000 | PPO training steps |
| `behavioral_cloning_epochs` | 20 | Stage A BC epochs |
| `alpha/beta/gamma_reward` | 1.0 / 0.01 / 0.1 | Reward function weights |
