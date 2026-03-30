# Real Hardware Cache Backend

> **Drop-in replacement for the simulated `CacheSimulator`.**
> Uses actual GPU VRAM, CPU RAM, and NVMe disk instead of fake numbers.

## What This Does

The original `cache_simulator.py` uses Python dictionaries and hardcoded latency
constants. The RL agent trains against these fake numbers.

This module replaces that with **real hardware**:

| Tier | Simulated (original) | Hardware (this module) |
|------|----------------------|------------------------|
| **L1** | `OrderedDict[int, int]` | `torch.cuda.FloatTensor` on GPU VRAM |
| **L2** | `OrderedDict[int, int]` | `torch.FloatTensor` in pinned CPU RAM |
| **L3** | `OrderedDict[int, int]` | Binary files on NVMe SSD |
| **Latency** | Hardcoded constants | Measured with CUDA events |
| **Eviction** | Same LRU logic | Same LRU logic + real data movement |
| **Prefetch** | Instant + fake cost | Real disk→CPU copy (measured) |

## Quick Start

### 1. Benchmark Your Hardware

```bash
# From rl_cache_prefetch/
python -m env.hardware.benchmark

# With custom config:
python -m env.hardware.benchmark --l1-mb 24 --l2-mb 32 --trials 50
```

This will:
- Measure real latencies for each tier
- Compare against simulated values
- Save results to `results/hardware_benchmark_*.csv`
- Generate plots in `results/plots/`

### 2. Use in Training

```python
from env.hardware import HardwareCacheEnv, HardwareConfig

# Configure (edit these to match YOUR hardware!)
config = HardwareConfig(
    l1_capacity_mb=48.0,    # GPU VRAM budget
    l2_capacity_mb=51.2,    # CPU RAM budget
    l3_capacity_mb=5120.0,  # Disk budget
    verbose=False,          # Quiet during training
)

# Create environment (same API as CacheEnv!)
env = HardwareCacheEnv(
    trace_path="data/traces_rag.csv",
    config=config,
    query_embeddings=embeddings,
)

# Use with stable-baselines3 PPO exactly like before
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=50000)
```

### 3. Load Config from YAML

```yaml
# hardware_config.yaml
l1_capacity_mb: 24.0
l2_capacity_mb: 32.0
l3_capacity_mb: 1024.0
chunk_size_bytes: 3145728
verbose: false
enable_operation_log: true
```

```python
config = HardwareConfig.from_yaml("hardware_config.yaml")
```

## Configuration Guide

### ⚠️ The Most Important Setting: Cache Pressure

If L1 (GPU) is too large, **all chunks fit in VRAM** and the RL agent
has nothing to learn. The key is to create enough "cache pressure" so
chunks spill to L2 and L3, giving the agent meaningful prefetch decisions.

**Rule of thumb:**
- Count how many chunks your workload uses (check trace CSVs)
- Set L1 to hold **~30-50%** of those chunks
- Set L2 to hold **another ~30%**
- L3 handles overflow

**Example for RAG workload (8 chunks per query):**
```python
config = HardwareConfig(
    chunk_size_bytes=3_145_728,  # 3 MB per chunk
    l1_capacity_mb=12.0,        # Fits 4 chunks (50% of 8)
    l2_capacity_mb=9.0,         # Fits 3 chunks
    l3_capacity_mb=1024.0,      # Overflow
)
```

### All Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size_bytes` | 3,145,728 | Size of each KV cache chunk (3 MB) |
| `chunk_size_tokens` | 256 | Tokens per chunk |
| `l1_capacity_mb` | 48.0 | GPU VRAM budget for cache |
| `l2_capacity_mb` | 51.2 | CPU pinned RAM budget |
| `l3_capacity_mb` | 5120.0 | Disk storage budget |
| `l3_disk_dir` | `./data/hardware_cache_store` | Where L3 files are stored |
| `force_cpu_mode` | False | Run without GPU (for testing) |
| `verbose` | True | Print detailed operation logs |
| `enable_operation_log` | True | Log ops for CSV export |
| `cuda_warmup_iterations` | 5 | GPU warmup before measuring |

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3050)
- **CPU**: Any modern CPU with enough RAM for L2
- **Disk**: NVMe SSD recommended for realistic L3 latencies
- **PyTorch**: >= 2.0 with CUDA support

### CPU-Only Fallback

If no CUDA GPU is available, the cache runs in CPU-only mode:
- L1 uses regular CPU tensors (not real GPU!)
- Latencies will NOT reflect real GPU performance
- Useful for debugging the logic on machines without GPUs

## File Structure

```
env/hardware/
├── __init__.py           # Exports HardwareCache, HardwareCacheEnv, HardwareConfig
├── hardware_config.py    # All configurable parameters
├── hardware_cache.py     # Real GPU/CPU/Disk cache implementation
├── hardware_cache_env.py # Gymnasium environment wrapper
├── benchmark.py          # Benchmarking + CSV + plots
└── README.md             # This file
```

## Output Files

After running benchmarks or training:

```
results/
├── hardware_benchmark_latencies.csv    # Per-trial latency measurements
├── hardware_benchmark_operations.csv   # Every cache operation logged
├── hardware_benchmark_episode.csv      # Per-step episode metrics
└── plots/
    ├── hardware_latency_comparison.png # Measured vs simulated latencies
    └── hardware_episode_timeline.png   # Reward/latency/tiers over time
```
