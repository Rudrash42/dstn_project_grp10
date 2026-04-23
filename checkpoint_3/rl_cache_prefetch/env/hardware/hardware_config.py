"""
Hardware Cache Configuration
=============================
ALL configurable knobs for the real hardware cache live here.
Edit these values to match YOUR hardware and experiment needs.

┌─────────────────────────────────────────────────────────────────┐
│  🔧  EASY CONFIGURATION GUIDE                                  │
│                                                                 │
│  • Want MORE cache pressure (harder for the RL agent)?          │
│    → REDUCE l1_capacity_mb and l2_capacity_mb                   │
│    → This forces chunks to spill to slower tiers                │
│                                                                 │
│  • Want LESS cache pressure (easier for the agent)?             │
│    → INCREASE l1_capacity_mb and l2_capacity_mb                 │
│    → More chunks fit in fast tiers                              │
│                                                                 │
│  • Chunk size controls granularity:                             │
│    → Bigger chunks = fewer fit in cache = more pressure         │
│    → Smaller chunks = more fit = less pressure                  │
│                                                                 │
│  • IMPORTANT: If L1 is huge, ALL chunks fit in GPU VRAM         │
│    and the agent has nothing to learn! Keep L1 small for        │
│    meaningful training.                                         │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """
    Configuration for the real hardware 3-tier cache.

    There are THREE tiers, just like in the simulated version:
        L1 = GPU VRAM   (fastest, smallest)  — the "hot" tier
        L2 = CPU RAM    (medium)              — the "warm" tier
        L3 = NVMe Disk  (slowest, largest)    — the "cold" tier

    The RL agent learns to prefetch chunks from L3 → L2 before they're
    needed, so when the LLM asks for them, they're already warm.

    ALL values below can be overridden via YAML config file or
    by editing the defaults directly.
    """

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  CHUNK GEOMETRY — How big is each KV cache chunk?        ║
    # ╚═══════════════════════════════════════════════════════════╝

    # Number of tokens per chunk (defines the granularity of cache management)
    chunk_size_tokens: int = 256

    # Bytes of KV cache data per token
    # Formula: 2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes
    # For Qwen2.5-0.5B: 2 × 24 × 2 × 64 × 2 = 12,288 bytes/token
    kv_bytes_per_token: int = 12288

    # Total bytes per chunk = chunk_size_tokens × kv_bytes_per_token
    # Default: 256 × 12,288 = 3,145,728 bytes (3 MB)
    chunk_size_bytes: int = 3_145_728

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  TIER CAPACITIES — How much storage per tier?             ║
    # ║                                                           ║
    # ║  ⚠️  CRITICAL: These determine cache pressure!            ║
    # ║  If L1 is too big for your workload, everything fits      ║
    # ║  in GPU and the RL agent learns nothing useful.           ║
    # ║                                                           ║
    # ║  Example: with 3 MB chunks:                               ║
    # ║    L1 = 48 MB  → fits ~15 chunks                         ║
    # ║    L2 = 51 MB  → fits ~16 chunks                         ║
    # ║    L3 = 5120 MB → fits ~1600 chunks                      ║
    # ╚═══════════════════════════════════════════════════════════╝

    # L1 (GPU VRAM) — RTX 3050 has 4096 MB total VRAM
    # Keep this MUCH smaller than total VRAM so PyTorch/CUDA
    # can use the rest for the RL model and overhead.
    l1_capacity_mb: float = 6.0

    # L2 (CPU pinned RAM) — Your laptop's system RAM
    # Pinned memory is locked in physical RAM (can't be swapped to disk)
    # so don't set this too high or you'll starve your OS.
    l2_capacity_mb: float = 9.0

    # L3 (NVMe disk) — Your SSD storage
    # This can be large since disk is cheap.
    l3_capacity_mb: float = 5120.0

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  DISK STORAGE — Where to save L3 chunks on disk           ║
    # ╚═══════════════════════════════════════════════════════════╝

    # Directory where L3 chunk files are stored
    # Each chunk becomes one file: chunk_0042.pt (about 3 MB each)
    l3_disk_dir: str = "./data/hardware_cache_store"

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  SIMULATED LATENCIES (for baseline/reward computation)    ║
    # ║                                                           ║
    # ║  These are the ESTIMATED latencies used to compute what   ║
    # ║  "would have happened" without prefetching (the baseline).║
    # ║  Actual access latencies are MEASURED on real hardware!   ║
    # ╚═══════════════════════════════════════════════════════════╝

    # L1 (GPU) hit latency — sub-millisecond, data already on GPU
    l1_hit_latency_ms: float = 0.1

    # L2 (CPU→GPU) hit latency — PCIe transfer of ~3 MB
    # RTX 3050 uses PCIe 4.0 x8 → ~12 GB/s → 3 MB in ~0.25 ms
    l2_hit_latency_ms: float = 0.25

    # L3 (Disk→CPU→GPU) hit latency — NVMe read + PCIe transfer
    # NVMe SSD ~3 GB/s → 3 MB in ~1 ms + PCIe ~0.25 ms ≈ 6 ms
    l3_hit_latency_ms: float = 6.0

    # Cold miss — must recompute KV cache from scratch
    # This is the SLOWEST operation: run the LLM forward pass
    cold_compute_per_chunk_ms: float = 37.0

    # Prefetch cost (L3 → L2) — reading from disk to CPU RAM
    prefetch_l3_to_l2_ms: float = 6.0

    # Prefetch cost (L2 → L1) — reading from CPU RAM to GPU VRAM
    prefetch_l2_to_l1_ms: float = 0.1

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  REWARD FUNCTION KNOBS                                    ║
    # ║                                                           ║
    # ║  R = α × time_saved - β × MB_migrated - γ × unused_count ║
    # ╚═══════════════════════════════════════════════════════════╝

    alpha: float = 1.0        # Reward weight for time saved (ms)
    beta: float = 0.01        # Penalty per MB of data migrated
    gamma_reward: float = 0.1  # Penalty per wasted (unused) prefetch

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  OBSERVATION SPACE — What the RL agent sees               ║
    # ╚═══════════════════════════════════════════════════════════╝

    # Dimension of query text embeddings (all-MiniLM-L6-v2 = 384)
    embed_dim: int = 384

    # How many L3/L2 candidate chunks to show the agent for prefetching
    max_candidate_chunks: int = 16

    # How many cached chunks to show the agent for eviction decisions
    max_eviction_candidates: int = 16

    # How many past accesses to track for recency scoring
    history_len: int = 5

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  HARDWARE BEHAVIOR FLAGS                                  ║
    # ╚═══════════════════════════════════════════════════════════╝

    # Use CUDA for GPU operations? Auto-detected if not set.
    # Set to False to run in CPU-only mode (useful for debugging
    # on machines without a GPU — L1 uses normal CPU tensors).
    force_cpu_mode: bool = False

    # Enable verbose print statements?
    # True = prints every insert/access/evict/prefetch operation
    # False = prints only summaries and errors
    verbose: bool = True

    # Log every operation to a list for CSV export?
    enable_operation_log: bool = True

    # Number of CUDA warmup operations before real measurements
    # (first few GPU ops are slow due to driver initialization)
    cuda_warmup_iterations: int = 5

    # ─── Derived properties ────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Total observation vector size: embedding + cache_stats + candidates + eviction."""
        return self.embed_dim + 3 + self.max_candidate_chunks + self.max_eviction_candidates

    @property
    def l1_capacity_bytes(self) -> int:
        """L1 capacity in bytes."""
        return int(self.l1_capacity_mb * 1024 * 1024)

    @property
    def l2_capacity_bytes(self) -> int:
        """L2 capacity in bytes."""
        return int(self.l2_capacity_mb * 1024 * 1024)

    @property
    def l3_capacity_bytes(self) -> int:
        """L3 capacity in bytes."""
        return int(self.l3_capacity_mb * 1024 * 1024)

    @property
    def l1_capacity_chunks(self) -> int:
        """How many chunks fit in L1."""
        return self.l1_capacity_bytes // self.chunk_size_bytes

    @property
    def l2_capacity_chunks(self) -> int:
        """How many chunks fit in L2."""
        return self.l2_capacity_bytes // self.chunk_size_bytes

    @property
    def l3_capacity_chunks(self) -> int:
        """How many chunks fit in L3."""
        return self.l3_capacity_bytes // self.chunk_size_bytes

    @property
    def chunk_num_floats(self) -> int:
        """Number of float32 elements per chunk (for tensor creation)."""
        return self.chunk_size_bytes // 4  # float32 = 4 bytes

    # ─── Summary ───────────────────────────────────────────────

    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("\n" + "=" * 64)
        print("  ⚙️  HARDWARE CACHE CONFIGURATION")
        print("=" * 64)

        print(f"\n  📦 Chunk Geometry:")
        print(f"     Tokens/chunk:     {self.chunk_size_tokens}")
        print(f"     Bytes/token (KV): {self.kv_bytes_per_token:,}")
        print(f"     Bytes/chunk:      {self.chunk_size_bytes:,} ({self.chunk_size_bytes / 1e6:.1f} MB)")
        print(f"     Floats/chunk:     {self.chunk_num_floats:,}")

        print(f"\n  🏗️  Tier Capacities:")
        print(f"     L1 (GPU VRAM):  {self.l1_capacity_mb:>8.1f} MB → fits {self.l1_capacity_chunks:>4d} chunks")
        print(f"     L2 (CPU RAM):   {self.l2_capacity_mb:>8.1f} MB → fits {self.l2_capacity_chunks:>4d} chunks")
        print(f"     L3 (NVMe Disk): {self.l3_capacity_mb:>8.1f} MB → fits {self.l3_capacity_chunks:>4d} chunks")

        print(f"\n  ⏱️  Baseline Latencies (for reward computation):")
        print(f"     L1 hit:          {self.l1_hit_latency_ms:>6.2f} ms")
        print(f"     L2 hit:          {self.l2_hit_latency_ms:>6.2f} ms")
        print(f"     L3 hit:          {self.l3_hit_latency_ms:>6.2f} ms")
        print(f"     Cold miss:       {self.cold_compute_per_chunk_ms:>6.2f} ms")
        print(f"     Prefetch L3→L2:  {self.prefetch_l3_to_l2_ms:>6.2f} ms")
        print(f"     Prefetch L2→L1:  {self.prefetch_l2_to_l1_ms:>6.2f} ms")

        print(f"\n  🎯 Reward Weights:")
        print(f"     α (time saved):     {self.alpha}")
        print(f"     β (migration cost): {self.beta}")
        print(f"     γ (unused prefetch):{self.gamma_reward}")

        print(f"\n  🔧 Flags:")
        print(f"     CPU-only mode:  {self.force_cpu_mode}")
        print(f"     Verbose:        {self.verbose}")
        print(f"     Operation log:  {self.enable_operation_log}")
        print(f"     L3 disk dir:    {self.l3_disk_dir}")

        print("=" * 64 + "\n")

    # ─── YAML loading ─────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HardwareConfig":
        """
        Load config from a YAML file. Any key not present in the
        YAML falls back to the default value above.

        Example YAML:
            l1_capacity_mb: 24.0
            l2_capacity_mb: 32.0
            chunk_size_tokens: 128
            verbose: true
        """
        path = Path(path)
        if not path.exists():
            print(f"[HardwareConfig] YAML not found at {path}, using defaults.")
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Map YAML keys directly to dataclass fields
        # (we intentionally use the SAME key names for simplicity)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        unknown = []

        for key, value in raw.items():
            if key in valid_fields:
                kwargs[key] = value
            else:
                unknown.append(key)

        if unknown:
            print(f"[HardwareConfig] ⚠️  Unknown YAML keys (ignored): {unknown}")

        cfg = cls(**kwargs)
        print(f"[HardwareConfig] Loaded from {path}")
        return cfg

    def to_yaml(self, path: str | Path):
        """Save current config to a YAML file for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunk_size_tokens": self.chunk_size_tokens,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "chunk_size_bytes": self.chunk_size_bytes,
            "l1_capacity_mb": self.l1_capacity_mb,
            "l2_capacity_mb": self.l2_capacity_mb,
            "l3_capacity_mb": self.l3_capacity_mb,
            "l3_disk_dir": self.l3_disk_dir,
            "l1_hit_latency_ms": self.l1_hit_latency_ms,
            "l2_hit_latency_ms": self.l2_hit_latency_ms,
            "l3_hit_latency_ms": self.l3_hit_latency_ms,
            "cold_compute_per_chunk_ms": self.cold_compute_per_chunk_ms,
            "prefetch_l3_to_l2_ms": self.prefetch_l3_to_l2_ms,
            "prefetch_l2_to_l1_ms": self.prefetch_l2_to_l1_ms,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma_reward": self.gamma_reward,
            "embed_dim": self.embed_dim,
            "max_candidate_chunks": self.max_candidate_chunks,
            "history_len": self.history_len,
            "force_cpu_mode": self.force_cpu_mode,
            "verbose": self.verbose,
            "enable_operation_log": self.enable_operation_log,
            "cuda_warmup_iterations": self.cuda_warmup_iterations,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"[HardwareConfig] Saved to {path}")
