"""
Real Hardware 3-Tier Cache (L1 GPU / L2 CPU / L3 Disk)
======================================================

This is the REAL version of cache_simulator.py.

Instead of pretending with fake numbers, this class ACTUALLY:
  • Stores KV cache chunks as PyTorch tensors in GPU VRAM (L1)
  • Stores KV cache chunks as pinned CPU tensors in RAM (L2)
  • Stores KV cache chunks as files on your NVMe SSD (L3)
  • MEASURES real transfer latencies with CUDA events / perf_counter

The API is identical to CacheSimulator — same method names, same
signatures, same return types. This means CacheEnv (or HardwareCacheEnv)
can use either one interchangeably.

How it works at a high level:
─────────────────────────────
When a chunk is "inserted" (new KV data computed by the LLM):
  1. A random tensor of the right size (3 MB) is created on GPU
  2. If GPU is full, the oldest chunk is EVICTED from GPU → CPU
  3. If CPU is full, the oldest chunk is EVICTED from CPU → Disk

When a chunk is "accessed" (LLM needs this data):
  • L1 hit:  Data is already on GPU → near-zero latency
  • L2 hit:  Data must be copied CPU → GPU (measured ~0.2-0.5 ms)
  • L3 hit:  Data must be read Disk → CPU → GPU (measured ~2-10 ms)
  • MISS:    Data doesn't exist, must be created (measured, simulates recompute)

When a chunk is "prefetched" (RL agent's proactive decision):
  • Moves a chunk from Disk (L3) → CPU RAM (L2)
  • Next time the LLM needs it, it'll be an L2 hit instead of L3 hit

Requires: PyTorch with CUDA support (for real GPU cache)
          Falls back to CPU-only mode if CUDA unavailable
"""

from __future__ import annotations

import collections
import os
import shutil
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .hardware_config import HardwareConfig


class HardwareCache:
    """
    3-Tier KV Cache using REAL hardware (GPU / CPU / Disk).

    Drop-in replacement for CacheSimulator with identical API.
    The only difference: latencies are MEASURED, not hardcoded.

    Usage:
        from env.hardware import HardwareCache, HardwareConfig

        config = HardwareConfig(l1_capacity_mb=24, verbose=True)
        cache = HardwareCache(config)

        cache.insert_chunks([0, 1, 2, 3])     # Put chunks in L1 (GPU)
        tier, latency = cache.access_chunk(1)  # → ("L1", 0.04 ms)
        cost = cache.prefetch(5)               # Move chunk 5: Disk → CPU
    """

    def __init__(self, config: Optional[HardwareConfig] = None):
        self.cfg = config or HardwareConfig()

        # ─── Detect hardware ──────────────────────────────────
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for HardwareCache. "
                "Install it: pip install torch"
            )

        # Determine device: GPU or CPU fallback
        self.use_cuda = torch.cuda.is_available() and not self.cfg.force_cpu_mode
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.cfg.verbose:
            if self.use_cuda:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem_mb = torch.cuda.get_device_properties(0).total_mem / (1024**2)
                print(f"\n[HardwareCache] 🎮 GPU detected: {gpu_name}")
                print(f"[HardwareCache]    Total VRAM: {gpu_mem_mb:.0f} MB")
                print(f"[HardwareCache]    L1 budget:  {self.cfg.l1_capacity_mb:.1f} MB "
                      f"({self.cfg.l1_capacity_mb / gpu_mem_mb * 100:.1f}% of VRAM)")
            else:
                print(f"\n[HardwareCache] ⚠️  No CUDA GPU available (or force_cpu_mode=True)")
                print(f"[HardwareCache]    Running in CPU-ONLY mode")
                print(f"[HardwareCache]    L1 will use regular CPU tensors (NOT real GPU!)")

        # ─── L1: GPU VRAM (OrderedDict: chunk_id → tensor on GPU) ─────
        #     OrderedDict preserves insertion order for LRU eviction.
        #     Most-recently-used chunk is at the END of the dict.
        self.l1: collections.OrderedDict[int, torch.Tensor] = collections.OrderedDict()
        self.l1_bytes: int = 0  # Current total bytes stored in L1

        # ─── L2: CPU Pinned RAM (OrderedDict: chunk_id → pinned tensor) ─
        #     "Pinned" memory is locked in physical RAM — the GPU can DMA
        #     directly from it, avoiding an extra copy through pageable RAM.
        self.l2: collections.OrderedDict[int, torch.Tensor] = collections.OrderedDict()
        self.l2_bytes: int = 0

        # ─── L3: NVMe Disk (OrderedDict: chunk_id → file path string) ──
        #     Each chunk is saved as a raw binary file (.bin) on disk.
        self.l3: collections.OrderedDict[int, str] = collections.OrderedDict()
        self.l3_bytes: int = 0

        # ─── Access history (for recency scoring) ─────────────────────
        self.access_history: List[int] = []

        # ─── Operation log (for CSV export) ───────────────────────────
        #     Every insert/access/evict/prefetch gets logged here with
        #     timestamps, tiers, latencies, and chunk IDs.
        self.operation_log: List[Dict[str, Any]] = []
        self._op_counter: int = 0

        # ─── Create disk directory for L3 ─────────────────────────────
        self._disk_dir = Path(self.cfg.l3_disk_dir)
        self._disk_dir.mkdir(parents=True, exist_ok=True)

        # ─── CUDA warmup ──────────────────────────────────────────────
        #     The first few GPU operations are slow because CUDA needs
        #     to initialize contexts, compile kernels, etc.
        #     We do a few dummy operations to "warm up" the GPU.
        if self.use_cuda:
            self._cuda_warmup()

        if self.cfg.verbose:
            self.cfg.print_summary()

    # ═══════════════════════════════════════════════════════════════
    # RESET — Clear everything
    # ═══════════════════════════════════════════════════════════════

    def reset(self):
        """
        Clear all tiers, free memory, delete disk files.
        Called at the start of each RL episode.
        """
        if self.cfg.verbose:
            print("[HardwareCache] 🔄 Resetting all tiers...")

        # Free GPU tensors
        for cid, tensor in self.l1.items():
            del tensor
        self.l1.clear()
        self.l1_bytes = 0

        # Free GPU memory back to CUDA
        if self.use_cuda:
            torch.cuda.empty_cache()

        # Free CPU pinned tensors
        for cid, tensor in self.l2.items():
            del tensor
        self.l2.clear()
        self.l2_bytes = 0

        # Delete disk files
        for cid, filepath in self.l3.items():
            try:
                os.remove(filepath)
            except FileNotFoundError:
                pass
        self.l3.clear()
        self.l3_bytes = 0

        # Reset history and logs
        self.access_history.clear()
        self.operation_log.clear()
        self._op_counter = 0

        if self.cfg.verbose:
            print("[HardwareCache] ✅ Reset complete. All tiers empty.\n")

    # ═══════════════════════════════════════════════════════════════
    # INTERNAL: Tensor creation & timing helpers
    # ═══════════════════════════════════════════════════════════════

    def _create_chunk_tensor(self, on_device: str = "cpu") -> torch.Tensor:
        """
        Create a tensor representing one KV cache chunk.

        The tensor has the RIGHT SIZE to match real KV cache data:
          chunk_size_bytes = 3,145,728 bytes = 786,432 float32 values

        We fill it with random data (the content doesn't matter for
        cache behavior — only size and location matter).

        Parameters
        ----------
        on_device : str
            Where to create the tensor: "cpu", "cpu_pinned", or "cuda"
        """
        num_floats = self.cfg.chunk_num_floats  # e.g., 786,432

        if on_device == "cuda" and self.use_cuda:
            # Create directly on GPU
            tensor = torch.randn(num_floats, dtype=torch.float32, device="cuda")
        elif on_device == "cpu_pinned":
            # Create in pinned (page-locked) CPU memory
            tensor = torch.randn(num_floats, dtype=torch.float32).pin_memory()
        else:
            # Regular CPU tensor
            tensor = torch.randn(num_floats, dtype=torch.float32)

        return tensor

    def _measure_time_ms(self):
        """
        Return a context-manager-like pair of functions for timing.
        Uses CUDA events for GPU ops, perf_counter for CPU/disk ops.

        Usage:
            start_fn, end_fn = self._measure_time_ms()
            start_fn()
            ... do something ...
            elapsed_ms = end_fn()
        """
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            def start():
                start_event.record()

            def end() -> float:
                end_event.record()
                torch.cuda.synchronize()
                return start_event.elapsed_time(end_event)  # milliseconds

            return start, end
        else:
            # CPU fallback: use perf_counter (microsecond resolution)
            container = {"t0": 0.0}

            def start():
                container["t0"] = time.perf_counter()

            def end() -> float:
                return (time.perf_counter() - container["t0"]) * 1000  # ms

            return start, end

    def _cuda_warmup(self):
        """
        Run dummy GPU operations to warm up CUDA.

        Why? The very first CUDA operation in a process triggers:
          - CUDA context initialization (~100-500 ms)
          - JIT kernel compilation
          - Memory allocator setup

        After warmup, subsequent operations show realistic latencies.
        """
        if self.cfg.verbose:
            print(f"[HardwareCache] 🔥 Warming up CUDA ({self.cfg.cuda_warmup_iterations} iterations)...")

        for i in range(self.cfg.cuda_warmup_iterations):
            # Create a small tensor on GPU, do a simple operation, delete it
            dummy = torch.randn(1000, device="cuda")
            _ = dummy * 2.0
            del dummy

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if self.cfg.verbose:
            print("[HardwareCache] 🔥 CUDA warmup complete.\n")

    def _log_operation(self, op_type: str, chunk_id: int,
                       tier: str, latency_ms: float, detail: str = ""):
        """Log an operation for later CSV export."""
        if not self.cfg.enable_operation_log:
            return

        self._op_counter += 1
        self.operation_log.append({
            "op_num": self._op_counter,
            "timestamp": time.time(),
            "operation": op_type,
            "chunk_id": chunk_id,
            "tier": tier,
            "latency_ms": round(latency_ms, 4),
            "detail": detail,
            "l1_count": len(self.l1),
            "l2_count": len(self.l2),
            "l3_count": len(self.l3),
            "l1_mb": round(self.l1_bytes / 1e6, 2),
            "l2_mb": round(self.l2_bytes / 1e6, 2),
            "l3_mb": round(self.l3_bytes / 1e6, 2),
        })

    # ═══════════════════════════════════════════════════════════════
    # DISK I/O: Save and load chunk tensors to/from disk
    # ═══════════════════════════════════════════════════════════════

    def _chunk_filepath(self, chunk_id: int) -> str:
        """Get the disk file path for a chunk."""
        return str(self._disk_dir / f"chunk_{chunk_id:06d}.bin")

    def _save_to_disk(self, chunk_id: int, tensor: torch.Tensor) -> float:
        """
        Save a tensor to disk as a raw binary file.
        Returns the time taken in milliseconds.

        We use raw binary (numpy tofile) instead of torch.save
        to avoid pickle overhead and get clean disk I/O timing.
        """
        filepath = self._chunk_filepath(chunk_id)

        t0 = time.perf_counter()
        # Convert to numpy and write raw bytes
        np_arr = tensor.detach().cpu().numpy()
        np_arr.tofile(filepath)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return elapsed_ms

    def _load_from_disk(self, chunk_id: int) -> Tuple[torch.Tensor, float]:
        """
        Load a tensor from disk.
        Returns (tensor_on_cpu, time_ms).
        """
        filepath = self._chunk_filepath(chunk_id)

        t0 = time.perf_counter()
        np_arr = np.fromfile(filepath, dtype=np.float32)
        tensor = torch.from_numpy(np_arr)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return tensor, elapsed_ms

    def _delete_from_disk(self, chunk_id: int):
        """Delete a chunk file from disk."""
        filepath = self._chunk_filepath(chunk_id)
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass

    # ═══════════════════════════════════════════════════════════════
    # EVICTION: LRU eviction from each tier (waterfall)
    # ═══════════════════════════════════════════════════════════════

    def _evict_lru_l1(self) -> Tuple[int, float]:
        """
        Evict the LEAST recently used chunk from L1 (GPU) → L2 (CPU).

        This is called when L1 is full and we need to make room.
        The evicted chunk's tensor is moved from GPU → pinned CPU memory.

        Returns (evicted_chunk_id, transfer_latency_ms)
        """
        if not self.l1:
            return -1, 0.0

        # Pop the OLDEST entry (first item in OrderedDict = LRU)
        cid, gpu_tensor = self.l1.popitem(last=False)
        self.l1_bytes -= self.cfg.chunk_size_bytes

        # Measure the GPU → CPU transfer
        start_fn, end_fn = self._measure_time_ms()
        start_fn()

        # Move tensor from GPU to pinned CPU memory
        if self.use_cuda:
            cpu_tensor = gpu_tensor.cpu().pin_memory()
        else:
            cpu_tensor = gpu_tensor.clone()

        # Free the GPU copy
        del gpu_tensor

        latency_ms = end_fn()

        # Insert into L2 (may trigger L2 → L3 eviction)
        self._insert_l2_tensor(cid, cpu_tensor)

        if self.cfg.verbose:
            print(f"  [EVICT] L1→L2  chunk={cid:<5d}  "
                  f"transfer={latency_ms:.3f}ms  "
                  f"L1: {len(self.l1)}/{self.cfg.l1_capacity_chunks} chunks")

        self._log_operation("evict_l1_to_l2", cid, "L1→L2", latency_ms)
        return cid, latency_ms

    def _evict_lru_l2(self) -> Tuple[int, float]:
        """
        Evict the LEAST recently used chunk from L2 (CPU) → L3 (Disk).

        The evicted chunk's tensor is saved to a file on NVMe SSD.

        Returns (evicted_chunk_id, save_latency_ms)
        """
        if not self.l2:
            return -1, 0.0

        # Pop the OLDEST entry (LRU)
        cid, cpu_tensor = self.l2.popitem(last=False)
        self.l2_bytes -= self.cfg.chunk_size_bytes

        # Measure the CPU → Disk save
        save_ms = self._save_to_disk(cid, cpu_tensor)

        # Free our reference to the CPU tensor
        del cpu_tensor

        # Record the file path in L3
        self.l3[cid] = self._chunk_filepath(cid)
        self.l3_bytes += self.cfg.chunk_size_bytes

        if self.cfg.verbose:
            print(f"  [EVICT] L2→L3  chunk={cid:<5d}  "
                  f"disk_write={save_ms:.3f}ms  "
                  f"L2: {len(self.l2)}/{self.cfg.l2_capacity_chunks} | "
                  f"L3: {len(self.l3)} chunks")

        self._log_operation("evict_l2_to_l3", cid, "L2→L3", save_ms)
        return cid, save_ms

    def _evict_lru_l3(self) -> Tuple[int, float]:
        """
        Evict the LEAST recently used chunk from L3 (Disk).

        This just deletes the file — the chunk is gone forever.

        Returns (evicted_chunk_id, delete_latency_ms)
        """
        if not self.l3:
            return -1, 0.0

        # Pop the OLDEST
        cid, filepath = self.l3.popitem(last=False)
        self.l3_bytes -= self.cfg.chunk_size_bytes

        # Delete the file
        t0 = time.perf_counter()
        self._delete_from_disk(cid)
        delete_ms = (time.perf_counter() - t0) * 1000

        if self.cfg.verbose:
            print(f"  [EVICT] L3 DEL chunk={cid:<5d}  "
                  f"delete={delete_ms:.3f}ms")

        self._log_operation("evict_l3_delete", cid, "L3", delete_ms)
        return cid, delete_ms

    # ═══════════════════════════════════════════════════════════════
    # INSERTION: Insert tensors into each tier
    # ═══════════════════════════════════════════════════════════════

    def _insert_l1_tensor(self, cid: int, tensor: torch.Tensor):
        """
        Insert a chunk tensor into L1 (GPU VRAM).
        If L1 is full, evicts LRU chunks to L2 first.

        The tensor is moved to GPU if it isn't already there.
        """
        chunk_bytes = self.cfg.chunk_size_bytes

        # Evict from L1 until there's room for one chunk
        while self.l1_bytes + chunk_bytes > self.cfg.l1_capacity_bytes and self.l1:
            self._evict_lru_l1()

        # Move tensor to GPU (if not already there)
        if self.use_cuda and tensor.device.type != "cuda":
            tensor = tensor.to("cuda", non_blocking=False)
        elif not self.use_cuda:
            tensor = tensor.clone()

        # Store in L1 OrderedDict (at the END = most recently used)
        self.l1[cid] = tensor
        self.l1_bytes += chunk_bytes

    def _insert_l2_tensor(self, cid: int, tensor: torch.Tensor):
        """
        Insert a chunk tensor into L2 (CPU pinned RAM).
        If L2 is full, evicts LRU chunks to L3 first.
        """
        # Skip if already in L2
        if cid in self.l2:
            self.l2.move_to_end(cid)
            return

        chunk_bytes = self.cfg.chunk_size_bytes

        # Evict from L2 until there's room
        while self.l2_bytes + chunk_bytes > self.cfg.l2_capacity_bytes and self.l2:
            self._evict_lru_l2()

        # Ensure tensor is on CPU and pinned
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if self.use_cuda and not tensor.is_pinned():
            tensor = tensor.pin_memory()

        self.l2[cid] = tensor
        self.l2_bytes += chunk_bytes

    def _insert_l3_filepath(self, cid: int, tensor: torch.Tensor):
        """
        Insert a chunk into L3 by saving tensor to disk.
        If L3 is full, evicts LRU chunks first.
        """
        # Skip if already in L3
        if cid in self.l3:
            self.l3.move_to_end(cid)
            return

        chunk_bytes = self.cfg.chunk_size_bytes

        # Evict from L3 until there's room
        while self.l3_bytes + chunk_bytes > self.cfg.l3_capacity_bytes and self.l3:
            self._evict_lru_l3()

        # Save to disk
        self._save_to_disk(cid, tensor)
        self.l3[cid] = self._chunk_filepath(cid)
        self.l3_bytes += chunk_bytes

    # ═══════════════════════════════════════════════════════════════
    # INSERT CHUNKS — Public API (matches CacheSimulator)
    # ═══════════════════════════════════════════════════════════════

    def insert_chunks(self, chunk_ids: List[int]):
        """
        Insert newly-computed KV cache chunks into L1 (GPU VRAM).

        This simulates what happens when the LLM computes KV cache for new tokens:
          1. The KV data is created on GPU (that's where the model runs)
          2. If GPU cache is full, old chunks waterfall: L1→L2→L3

        If a chunk already exists in any tier, it's just "touched"
        (moved to most-recently-used position) — no new data needed.

        Parameters
        ----------
        chunk_ids : list[int]
            IDs of chunks to insert (e.g., from the trace CSV).
        """
        for cid in chunk_ids:
            # ── Already in L1? Just touch it. ──
            if cid in self.l1:
                self.l1.move_to_end(cid)  # Mark as recently used
                continue

            # ── Already in L2? Promote to L1 (CPU → GPU). ──
            if cid in self.l2:
                tensor = self.l2.pop(cid)
                self.l2_bytes -= self.cfg.chunk_size_bytes
                self._insert_l1_tensor(cid, tensor)
                continue

            # ── Already in L3? Promote to L1 (Disk → GPU). ──
            if cid in self.l3:
                self.l3.pop(cid)
                self.l3_bytes -= self.cfg.chunk_size_bytes
                tensor, _ = self._load_from_disk(cid)
                self._delete_from_disk(cid)
                self._insert_l1_tensor(cid, tensor)
                continue

            # ── Brand new chunk: create on GPU (simulates LLM computation) ──
            tensor = self._create_chunk_tensor(on_device="cuda" if self.use_cuda else "cpu")
            self._insert_l1_tensor(cid, tensor)

            if self.cfg.verbose:
                print(f"  [INSERT] new chunk={cid:<5d} → L1  "
                      f"({len(self.l1)}/{self.cfg.l1_capacity_chunks} in L1)")

    # ═══════════════════════════════════════════════════════════════
    # ACCESS CHUNK — Read with REAL measured latency
    # ═══════════════════════════════════════════════════════════════

    def access_chunk(self, chunk_id: int) -> Tuple[str, float]:
        """
        Access (read) a chunk — simulates the LLM needing this KV data.

        UNLIKE the simulator, this method MEASURES real latency:
          • L1 hit:  Tensor is already on GPU. We do a trivial read
                     operation to measure GPU memory access time.
          • L2 hit:  Tensor is on CPU. We copy it to GPU via PCIe.
                     Latency = real PCIe transfer time for ~3 MB.
          • L3 hit:  Tensor is on disk. We load from NVMe → CPU → GPU.
                     Latency = real disk read + PCIe transfer.
          • MISS:    Chunk not in cache. We create a new tensor on GPU
                     to simulate KV recomputation from scratch.

        Returns
        -------
        (tier_name, latency_ms) : tuple
            tier_name: "L1", "L2", "L3", or "MISS"
            latency_ms: Measured wall-clock time in milliseconds
        """
        self.access_history.append(chunk_id)

        # ── L1 HIT: Already on GPU ──────────────────────────────
        if chunk_id in self.l1:
            start_fn, end_fn = self._measure_time_ms()
            start_fn()

            # Touch the tensor (a minimal read to simulate access)
            tensor = self.l1[chunk_id]
            _ = tensor.sum()  # Force a real GPU operation
            self.l1.move_to_end(chunk_id)  # Update LRU

            latency_ms = end_fn()

            self._log_operation("access", chunk_id, "L1", latency_ms, "GPU VRAM hit")
            return "L1", latency_ms

        # ── L2 HIT: On CPU, must transfer to GPU ────────────────
        if chunk_id in self.l2:
            start_fn, end_fn = self._measure_time_ms()
            start_fn()

            # Get the CPU tensor and copy to GPU
            cpu_tensor = self.l2.pop(chunk_id)
            self.l2_bytes -= self.cfg.chunk_size_bytes

            # Real PCIe transfer: CPU pinned memory → GPU VRAM
            if self.use_cuda:
                gpu_tensor = cpu_tensor.to("cuda", non_blocking=False)
            else:
                gpu_tensor = cpu_tensor.clone()

            # Insert into L1 (may trigger L1 eviction)
            self._insert_l1_tensor(chunk_id, gpu_tensor)

            latency_ms = end_fn()

            if self.cfg.verbose:
                print(f"  [ACCESS] L2 hit  chunk={chunk_id:<5d}  "
                      f"CPU→GPU={latency_ms:.3f}ms")

            self._log_operation("access", chunk_id, "L2", latency_ms, "CPU→GPU transfer")
            return "L2", latency_ms

        # ── L3 HIT: On disk, must load to CPU then GPU ──────────
        if chunk_id in self.l3:
            start_fn, end_fn = self._measure_time_ms()
            start_fn()

            # Remove from L3 tracking
            filepath = self.l3.pop(chunk_id)
            self.l3_bytes -= self.cfg.chunk_size_bytes

            # Real NVMe read: Disk → CPU
            cpu_tensor, _ = self._load_from_disk(chunk_id)

            # Real PCIe transfer: CPU → GPU
            if self.use_cuda:
                gpu_tensor = cpu_tensor.to("cuda", non_blocking=False)
            else:
                gpu_tensor = cpu_tensor.clone()

            # Clean up disk file
            self._delete_from_disk(chunk_id)

            # Insert into L1
            self._insert_l1_tensor(chunk_id, gpu_tensor)

            latency_ms = end_fn()

            if self.cfg.verbose:
                print(f"  [ACCESS] L3 hit  chunk={chunk_id:<5d}  "
                      f"Disk→GPU={latency_ms:.3f}ms")

            self._log_operation("access", chunk_id, "L3", latency_ms, "Disk→CPU→GPU")
            return "L3", latency_ms

        # ── COLD MISS: Chunk not in cache at all ────────────────
        start_fn, end_fn = self._measure_time_ms()
        start_fn()

        # Simulate KV recomputation: create new tensor on GPU
        tensor = self._create_chunk_tensor(on_device="cuda" if self.use_cuda else "cpu")
        self._insert_l1_tensor(chunk_id, tensor)

        latency_ms = end_fn()

        # Add the LLM compute time penalty because generating random data 
        # is vastly faster than actual LLM forward pass computation.
        if self.cfg.cold_compute_per_chunk_ms > 0:
            latency_ms += self.cfg.cold_compute_per_chunk_ms

        if self.cfg.verbose:
            print(f"  [ACCESS] MISS   chunk={chunk_id:<5d}  "
                  f"cold_compute={latency_ms:.3f}ms")

        self._log_operation("access", chunk_id, "MISS", latency_ms, "Cold recompute")
        return "MISS", latency_ms

    def access_chunks(self, chunk_ids: List[int]) -> Tuple[float, dict]:
        """
        Access multiple chunks. Returns total latency and per-tier counts.

        This is the batch version of access_chunk(), matching
        CacheSimulator's API exactly.
        """
        total_ms = 0.0
        tier_counts = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}

        for cid in chunk_ids:
            tier, lat = self.access_chunk(cid)
            total_ms += lat
            tier_counts[tier] += 1

        return total_ms, tier_counts

    # ═══════════════════════════════════════════════════════════════
    # PREFETCH — Proactive migration L3 → L2 (the RL agent's action)
    # ═══════════════════════════════════════════════════════════════

    def prefetch(self, chunk_id: int) -> float:
        """
        Proactively move a chunk from L3 (Disk) → L2 (CPU RAM).

        This is the RL agent's main action! The agent predicts which
        chunks will be needed soon and prefetches them to L2 so that
        when the LLM actually needs them, they're a fast L2 hit
        instead of a slow L3 hit.

        Returns
        -------
        float
            Cost of the prefetch in milliseconds (0 if chunk wasn't in L3).
        """
        # Already in L1? No need to prefetch.
        if chunk_id in self.l1:
            return 0.0

        # In L2? Move it to L1.
        if chunk_id in self.l2:
            t0 = time.perf_counter()

            # Get the CPU tensor and copy to GPU
            cpu_tensor = self.l2.pop(chunk_id)
            self.l2_bytes -= self.cfg.chunk_size_bytes

            if self.use_cuda:
                gpu_tensor = cpu_tensor.to("cuda", non_blocking=False)
            else:
                gpu_tensor = cpu_tensor.clone()

            self._insert_l1_tensor(chunk_id, gpu_tensor)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if self.cfg.verbose:
                print(f"  [PREFETCH] L2→L1  chunk={chunk_id:<5d}  "
                      f"cost={elapsed_ms:.3f}ms")

            self._log_operation("prefetch", chunk_id, "L2→L1", elapsed_ms)
            return elapsed_ms

        # In L3? Move it to L2.
        if chunk_id in self.l3:
            t0 = time.perf_counter()

            # Remove from L3
            filepath = self.l3.pop(chunk_id)
            self.l3_bytes -= self.cfg.chunk_size_bytes

            # Read from disk → CPU
            cpu_tensor, _ = self._load_from_disk(chunk_id)

            # Pin the memory for fast future GPU transfer
            if self.use_cuda:
                cpu_tensor = cpu_tensor.pin_memory()

            # Delete the disk file (data is now in CPU RAM)
            self._delete_from_disk(chunk_id)

            # Insert into L2
            self._insert_l2_tensor(chunk_id, cpu_tensor)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if self.cfg.verbose:
                print(f"  [PREFETCH] L3→L2  chunk={chunk_id:<5d}  "
                      f"cost={elapsed_ms:.3f}ms  "
                      f"L2: {len(self.l2)}/{self.cfg.l2_capacity_chunks} chunks")

            self._log_operation("prefetch", chunk_id, "L3→L2", elapsed_ms)
            return elapsed_ms

        # Not in cache at all — can't prefetch what doesn't exist
        return 0.0

    # ═══════════════════════════════════════════════════════════════
    # OBSERVATION HELPERS (match CacheSimulator API exactly)
    # ═══════════════════════════════════════════════════════════════

    def get_l3_candidates(self, k: int) -> List[int]:
        """
        Return the top-k chunks in L3, ordered by most-recently-used first.
        These are the candidates the RL agent can choose to prefetch.
        """
        all_ids = list(self.l3.keys())
        return list(reversed(all_ids[-k:]))

    def get_all_candidates(self, k: int) -> List[int]:
        """
        Return top-k candidate chunks from L2+L3 combined.
        Prioritizes L3 chunks (they benefit more from prefetch).
        """
        l3_ids = list(reversed(list(self.l3.keys())))
        l2_ids = list(reversed(list(self.l2.keys())))
        combined = l3_ids + l2_ids
        return combined[:k]

    def get_stats(self) -> Tuple[float, float, float]:
        """
        Returns (l1_usage_fraction, l2_usage_mb, l3_usage_mb).

        These are REAL memory usage stats (not simulated).
        """
        l1_frac = (
            self.l1_bytes / self.cfg.l1_capacity_bytes
            if self.cfg.l1_capacity_bytes > 0 else 0
        )
        l2_mb = self.l2_bytes / (1024 * 1024)
        l3_mb = self.l3_bytes / (1024 * 1024)
        return l1_frac, l2_mb, l3_mb

    def get_recent_accesses(self, n: int) -> List[int]:
        """Return the last n chunk IDs accessed."""
        return self.access_history[-n:]

    def chunk_in_cache(self, chunk_id: int) -> Optional[str]:
        """Check which tier a chunk is in, or None if not cached."""
        if chunk_id in self.l1:
            return "L1"
        if chunk_id in self.l2:
            return "L2"
        if chunk_id in self.l3:
            return "L3"
        return None

    # ═══════════════════════════════════════════════════════════════
    # REPORTING & EXPORT
    # ═══════════════════════════════════════════════════════════════

    def get_operation_log_df(self):
        """
        Export the operation log as a pandas DataFrame.

        Each row contains:
          op_num, timestamp, operation, chunk_id, tier, latency_ms,
          detail, l1_count, l2_count, l3_count, l1_mb, l2_mb, l3_mb

        Usage:
            df = cache.get_operation_log_df()
            df.to_csv("hardware_cache_ops.csv", index=False)
        """
        import pandas as pd
        return pd.DataFrame(self.operation_log)

    def save_operation_log(self, filepath: str | Path):
        """Save the operation log to a CSV file."""
        df = self.get_operation_log_df()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"[HardwareCache] 📊 Operation log saved: {filepath} ({len(df)} ops)")

    def print_stats_summary(self):
        """Print a nice summary of current cache state."""
        l1_frac, l2_mb, l3_mb = self.get_stats()

        print(f"\n{'─' * 50}")
        print(f"  📊 Hardware Cache Status")
        print(f"{'─' * 50}")
        print(f"  L1 (GPU):  {len(self.l1):>4d} chunks  "
              f"{self.l1_bytes / 1e6:>8.1f} MB  "
              f"({l1_frac * 100:.1f}% full)")
        print(f"  L2 (CPU):  {len(self.l2):>4d} chunks  "
              f"{self.l2_bytes / 1e6:>8.1f} MB  "
              f"({self.l2_bytes / max(self.cfg.l2_capacity_bytes, 1) * 100:.1f}% full)")
        print(f"  L3 (Disk): {len(self.l3):>4d} chunks  "
              f"{self.l3_bytes / 1e6:>8.1f} MB  "
              f"({self.l3_bytes / max(self.cfg.l3_capacity_bytes, 1) * 100:.1f}% full)")
        print(f"  Total:     {len(self.l1) + len(self.l2) + len(self.l3)} chunks")
        print(f"  Accesses:  {len(self.access_history)}")

        if self.use_cuda:
            gpu_alloc = torch.cuda.memory_allocated() / 1e6
            gpu_reserved = torch.cuda.memory_reserved() / 1e6
            print(f"\n  🎮 GPU Memory:")
            print(f"     Allocated: {gpu_alloc:.1f} MB")
            print(f"     Reserved:  {gpu_reserved:.1f} MB")

        print(f"{'─' * 50}\n")

    def __del__(self):
        """Cleanup: try to free resources when object is garbage collected."""
        try:
            # Clean up disk files
            if hasattr(self, '_disk_dir') and self._disk_dir.exists():
                for f in self._disk_dir.glob("chunk_*.bin"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
