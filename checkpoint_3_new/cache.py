"""
3-Tier Hardware Cache: GPU VRAM / CPU Pinned RAM / NVMe Disk
==============================================================

Manages KV cache chunks as REAL PyTorch tensors across three storage tiers.
Every operation involves actual data movement and is timed with hardware timers.

Tiers:
  L1 (GPU):  torch.cuda.FloatTensor on VRAM — fastest (~0.1ms access)
  L2 (CPU):  torch.FloatTensor in pinned RAM — medium (~0.3ms, PCIe transfer)
  L3 (Disk): Serialized .pt files on NVMe SSD — slow (~6ms, disk read)

Eviction follows an LRU waterfall: L1 → L2 → L3 → delete.
Prefetch moves chunks from L3 → L2 (disk to CPU RAM).
"""

from __future__ import annotations

import collections
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class HardwareCache:
    """
    Real 3-tier KV cache using GPU/CPU/Disk.

    Each chunk is a PyTorch tensor sized to match a real KV cache chunk.
    Operations are timed with torch.cuda.synchronize() for accurate GPU timing.
    """

    def __init__(
        self,
        chunk_size_bytes: int = 3_145_728,
        l1_capacity_mb: float = 9.0,
        l2_capacity_mb: float = 15.0,
        l3_capacity_mb: float = 1024.0,
        l3_disk_dir: str = "./data/cache_store",
        force_cpu_mode: bool = False,
        cuda_warmup_iterations: int = 5,
        cold_compute_ms: float = 30.8,
    ):
        self.chunk_size_bytes = chunk_size_bytes
        self.l1_capacity_bytes = int(l1_capacity_mb * 1024 * 1024)
        self.l2_capacity_bytes = int(l2_capacity_mb * 1024 * 1024)
        self.l3_capacity_bytes = int(l3_capacity_mb * 1024 * 1024)
        self.l3_disk_dir = Path(l3_disk_dir)

        # Cold miss: inject realistic delay to simulate LLM forward pass
        # Without this, cold miss only measures tensor alloc (~0.05ms),
        # not actual compute cost (~30ms per chunk)
        self.cold_compute_ms = cold_compute_ms

        # Tensor shape: chunk_size_bytes / 4 bytes per float32 element
        self.tensor_elements = chunk_size_bytes // 4

        # Determine device
        self.use_cuda = torch.cuda.is_available() and not force_cpu_mode
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        # LRU ordered dicts — most recently used at the END
        # Values are the actual tensors (on their respective device/path)
        self.l1: collections.OrderedDict[int, torch.Tensor] = collections.OrderedDict()
        self.l2: collections.OrderedDict[int, torch.Tensor] = collections.OrderedDict()
        self.l3: collections.OrderedDict[int, str] = collections.OrderedDict()  # chunk_id → file path

        # Byte tracking
        self.l1_bytes = 0
        self.l2_bytes = 0
        self.l3_bytes = 0

        # Access history for observation building
        self.access_history: List[int] = []

        # Occupancy snapshots (for plotting cache utilization over time)
        self.occupancy_log: List[Dict[str, float]] = []

        # Per-access operation log (for detailed CSV output)
        self.operation_log: List[Dict] = []

        # Capacity in chunks (for convenience)
        self.l1_capacity_chunks = self.l1_capacity_bytes // chunk_size_bytes
        self.l2_capacity_chunks = self.l2_capacity_bytes // chunk_size_bytes

        # Setup disk directory
        self.l3_disk_dir.mkdir(parents=True, exist_ok=True)

        # CUDA warmup — first few GPU operations are slow due to driver init
        if self.use_cuda and cuda_warmup_iterations > 0:
            self._cuda_warmup(cuda_warmup_iterations)

    def _cuda_warmup(self, n: int):
        """Run dummy GPU operations to warm up CUDA runtime."""
        for _ in range(n):
            t = torch.randn(self.tensor_elements, device=self.device)
            _ = t.cpu()
            del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _sync_gpu(self):
        """Synchronize GPU to get accurate timing."""
        if self.use_cuda:
            torch.cuda.synchronize()

    def _create_chunk_tensor(self) -> torch.Tensor:
        """Create a new chunk tensor (simulates KV cache computation output)."""
        return torch.randn(self.tensor_elements, device=self.device)

    def _tensor_to_cpu_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to pinned CPU memory (for fast GPU transfer later)."""
        cpu_tensor = torch.empty(self.tensor_elements, pin_memory=self.use_cuda)
        cpu_tensor.copy_(tensor.cpu() if tensor.is_cuda else tensor)
        return cpu_tensor

    def _tensor_to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor from CPU to GPU."""
        if self.use_cuda:
            return tensor.to(self.device, non_blocking=True)
        return tensor.clone()

    def _save_tensor_to_disk(self, chunk_id: int, tensor: torch.Tensor) -> str:
        """Save a tensor to disk (L3 storage). Returns file path."""
        path = str(self.l3_disk_dir / f"chunk_{chunk_id}.pt")
        # Always save CPU version to disk
        cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor
        torch.save(cpu_tensor, path)
        return path

    def _load_tensor_from_disk(self, path: str) -> torch.Tensor:
        """Load a tensor from disk (L3 → CPU)."""
        return torch.load(path, map_location="cpu", weights_only=True)

    # ─── Eviction ─────────────────────────────────────────────────

    def _evict_l1_to_l2(self):
        """Evict LRU chunk from L1 → L2. If L2 full, cascade to L3."""
        if not self.l1:
            return
        chunk_id, tensor = self.l1.popitem(last=False)  # Remove oldest
        self.l1_bytes -= self.chunk_size_bytes

        # Demote to L2 (move tensor to CPU pinned memory)
        cpu_tensor = self._tensor_to_cpu_pinned(tensor)
        del tensor  # Free GPU memory

        self._insert_l2(chunk_id, cpu_tensor)

    def _evict_l2_to_l3(self):
        """Evict LRU chunk from L2 → L3 (CPU → Disk)."""
        if not self.l2:
            return
        chunk_id, tensor = self.l2.popitem(last=False)
        self.l2_bytes -= self.chunk_size_bytes

        # Save to disk
        path = self._save_tensor_to_disk(chunk_id, tensor)
        del tensor  # Free CPU memory

        self._insert_l3(chunk_id, path)

    def _evict_l3(self):
        """Evict LRU chunk from L3 (delete from disk)."""
        if not self.l3:
            return
        chunk_id, path = self.l3.popitem(last=False)
        self.l3_bytes -= self.chunk_size_bytes
        # Delete file
        try:
            os.remove(path)
        except OSError:
            pass

    # ─── Insertion ────────────────────────────────────────────────

    def _insert_l1(self, chunk_id: int, tensor: torch.Tensor):
        """Insert a chunk into L1 (GPU). Evicts LRU if full."""
        while self.l1_bytes + self.chunk_size_bytes > self.l1_capacity_bytes and self.l1:
            self._evict_l1_to_l2()

        gpu_tensor = self._tensor_to_gpu(tensor) if not tensor.is_cuda else tensor
        self.l1[chunk_id] = gpu_tensor
        self.l1_bytes += self.chunk_size_bytes

    def _insert_l2(self, chunk_id: int, tensor: torch.Tensor):
        """Insert a chunk into L2 (CPU pinned). Evicts LRU if full."""
        if chunk_id in self.l2:
            self.l2.move_to_end(chunk_id)
            return

        while self.l2_bytes + self.chunk_size_bytes > self.l2_capacity_bytes and self.l2:
            self._evict_l2_to_l3()

        cpu_tensor = self._tensor_to_cpu_pinned(tensor) if tensor.is_cuda else tensor
        self.l2[chunk_id] = cpu_tensor
        self.l2_bytes += self.chunk_size_bytes

    def _insert_l3(self, chunk_id: int, path: str):
        """Insert a chunk into L3 (Disk). Evicts LRU if full."""
        if chunk_id in self.l3:
            self.l3.move_to_end(chunk_id)
            return

        while self.l3_bytes + self.chunk_size_bytes > self.l3_capacity_bytes and self.l3:
            self._evict_l3()

        self.l3[chunk_id] = path
        self.l3_bytes += self.chunk_size_bytes

    # ─── Access (the main operation) ──────────────────────────────

    def access_chunk(self, chunk_id: int) -> Tuple[str, float]:
        """
        Access a single chunk. Returns (tier_name, latency_ms).

        If chunk is in a cache tier, it's promoted to L1 (and timed).
        If not cached at all, it's a cold miss — tensor allocation + injected
        compute delay (cold_compute_ms) to simulate real LLM forward pass cost.
        """
        self.access_history.append(chunk_id)
        self._sync_gpu()

        # ── L1 HIT (GPU) ──
        if chunk_id in self.l1:
            t0 = time.perf_counter()
            self.l1.move_to_end(chunk_id)  # Touch for LRU
            # Access the tensor (force a read to measure real timing)
            _ = self.l1[chunk_id].sum()
            self._sync_gpu()
            latency_ms = (time.perf_counter() - t0) * 1000
            self._log_operation(chunk_id, "L1", latency_ms)
            return "L1", latency_ms

        # ── L2 HIT (CPU → GPU) ──
        if chunk_id in self.l2:
            t0 = time.perf_counter()
            cpu_tensor = self.l2.pop(chunk_id)
            self.l2_bytes -= self.chunk_size_bytes
            # Transfer CPU → GPU (this is the real PCIe transfer!)
            self._insert_l1(chunk_id, cpu_tensor)
            del cpu_tensor
            self._sync_gpu()
            latency_ms = (time.perf_counter() - t0) * 1000
            self._log_operation(chunk_id, "L2", latency_ms)
            return "L2", latency_ms

        # ── L3 HIT (Disk → CPU → GPU) ──
        if chunk_id in self.l3:
            t0 = time.perf_counter()
            path = self.l3.pop(chunk_id)
            self.l3_bytes -= self.chunk_size_bytes
            # Read from disk, then transfer to GPU
            cpu_tensor = self._load_tensor_from_disk(path)
            self._insert_l1(chunk_id, cpu_tensor)
            del cpu_tensor
            # Clean up disk file
            try:
                os.remove(path)
            except OSError:
                pass
            self._sync_gpu()
            latency_ms = (time.perf_counter() - t0) * 1000
            self._log_operation(chunk_id, "L3", latency_ms)
            return "L3", latency_ms

        # ── MISS (create tensor + injected compute delay) ──
        # Real cost = tensor alloc + simulated LLM forward pass
        t0 = time.perf_counter()
        new_tensor = self._create_chunk_tensor()
        self._sync_gpu()
        # Insert into L1 (may cause evictions)
        self._insert_l1(chunk_id, new_tensor)
        self._sync_gpu()
        hw_latency_ms = (time.perf_counter() - t0) * 1000
        # Inject realistic compute cost: cold miss should be EXPENSIVE
        # so the agent learns to avoid misses through prefetching
        latency_ms = max(hw_latency_ms, self.cold_compute_ms)
        self._log_operation(chunk_id, "MISS", latency_ms)
        return "MISS", latency_ms

    def access_chunks(self, chunk_ids: List[int]) -> Tuple[float, Dict[str, int]]:
        """
        Access multiple chunks. Returns (total_latency_ms, tier_count_dict).
        """
        total_ms = 0.0
        tier_counts = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        for cid in chunk_ids:
            tier, lat = self.access_chunk(cid)
            total_ms += lat
            tier_counts[tier] += 1
        return total_ms, tier_counts

    def access_chunks_detailed(
        self, chunk_ids: List[int],
    ) -> Tuple[float, Dict[str, int], List[Dict]]:
        """
        Access multiple chunks with per-chunk detail.
        Returns (total_latency_ms, tier_counts, per_chunk_details).
        Each detail: {"chunk_id": int, "tier": str, "latency_ms": float}
        """
        total_ms = 0.0
        tier_counts = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        details = []
        for cid in chunk_ids:
            tier, lat = self.access_chunk(cid)
            total_ms += lat
            tier_counts[tier] += 1
            details.append({"chunk_id": cid, "tier": tier, "latency_ms": lat})
        return total_ms, tier_counts, details

    # ─── Prefetch (L3 → L2) ──────────────────────────────────────

    def prefetch(self, chunk_id: int) -> float:
        """
        Prefetch a chunk from L3 (disk) → L2 (CPU pinned RAM).
        Returns the transfer cost in ms (0 if chunk not in L3).
        """
        # Already in L1 or L2 — nothing to do
        if chunk_id in self.l1 or chunk_id in self.l2:
            return 0.0

        # Must be in L3 to prefetch
        if chunk_id not in self.l3:
            return 0.0

        self._sync_gpu()
        t0 = time.perf_counter()

        path = self.l3.pop(chunk_id)
        self.l3_bytes -= self.chunk_size_bytes

        # Read from disk to CPU pinned memory
        cpu_tensor = self._load_tensor_from_disk(path)
        pinned = self._tensor_to_cpu_pinned(cpu_tensor)
        del cpu_tensor

        self._insert_l2(chunk_id, pinned)

        # Clean up disk file
        try:
            os.remove(path)
        except OSError:
            pass

        latency_ms = (time.perf_counter() - t0) * 1000
        return latency_ms

    # ─── Observation helpers ──────────────────────────────────────

    def get_candidates(self, k: int) -> List[int]:
        """
        Return top-k candidate chunks from L2+L3 for prefetch decisions.
        Prioritizes L3 (they benefit more from prefetch: disk → CPU).
        Most recent first.
        """
        l3_ids = list(reversed(list(self.l3.keys())))
        l2_ids = list(reversed(list(self.l2.keys())))
        combined = l3_ids + l2_ids
        return combined[:k]

    def get_stats(self) -> Tuple[float, float, float]:
        """
        Returns (l1_usage_frac, l2_usage_frac, l3_usage_frac).
        All values in [0, 1].
        """
        l1_frac = self.l1_bytes / self.l1_capacity_bytes if self.l1_capacity_bytes > 0 else 0
        l2_frac = self.l2_bytes / self.l2_capacity_bytes if self.l2_capacity_bytes > 0 else 0
        l3_frac = self.l3_bytes / self.l3_capacity_bytes if self.l3_capacity_bytes > 0 else 0
        return l1_frac, l2_frac, l3_frac

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

    # ─── Logging / Metrics ────────────────────────────────────────

    def _log_operation(self, chunk_id: int, tier: str, latency_ms: float):
        """Log an access operation for detailed analysis."""
        self.operation_log.append({
            "chunk_id": chunk_id,
            "tier": tier,
            "latency_ms": latency_ms,
            "l1_count": len(self.l1),
            "l2_count": len(self.l2),
            "l3_count": len(self.l3),
        })

    def snapshot_occupancy(self, step: int = -1) -> Dict[str, float]:
        """Record a cache occupancy snapshot for metrics."""
        snap = {
            "step": step,
            "l1_chunks": len(self.l1),
            "l2_chunks": len(self.l2),
            "l3_chunks": len(self.l3),
            "l1_mb": round(self.l1_bytes / (1024 * 1024), 2),
            "l2_mb": round(self.l2_bytes / (1024 * 1024), 2),
            "l3_mb": round(self.l3_bytes / (1024 * 1024), 2),
            "l1_pct": round(self.l1_bytes / self.l1_capacity_bytes * 100, 1) if self.l1_capacity_bytes > 0 else 0,
            "l2_pct": round(self.l2_bytes / self.l2_capacity_bytes * 100, 1) if self.l2_capacity_bytes > 0 else 0,
        }
        self.occupancy_log.append(snap)
        return snap

    def get_operation_log(self) -> List[Dict]:
        """Return the full operation log."""
        return self.operation_log

    def get_occupancy_log(self) -> List[Dict[str, float]]:
        """Return all occupancy snapshots."""
        return self.occupancy_log

    # ─── Reset / Cleanup ─────────────────────────────────────────

    def reset(self):
        """Clear all tiers, free all memory, delete disk files."""
        # Free GPU tensors
        for tensor in self.l1.values():
            del tensor
        self.l1.clear()
        self.l1_bytes = 0

        # Free CPU tensors
        for tensor in self.l2.values():
            del tensor
        self.l2.clear()
        self.l2_bytes = 0

        # Delete disk files
        for path in self.l3.values():
            try:
                os.remove(path)
            except OSError:
                pass
        self.l3.clear()
        self.l3_bytes = 0

        self.access_history.clear()
        self.occupancy_log.clear()
        self.operation_log.clear()

        if self.use_cuda:
            torch.cuda.empty_cache()

    def cleanup_disk(self):
        """Remove the entire L3 disk directory."""
        if self.l3_disk_dir.exists():
            shutil.rmtree(self.l3_disk_dir)

    @property
    def total_chunks_cached(self) -> int:
        return len(self.l1) + len(self.l2) + len(self.l3)

    def __repr__(self) -> str:
        return (
            f"HardwareCache("
            f"L1={len(self.l1)}/{self.l1_capacity_chunks} chunks on "
            f"{'GPU' if self.use_cuda else 'CPU'}, "
            f"L2={len(self.l2)}/{self.l2_capacity_chunks} chunks in CPU RAM, "
            f"L3={len(self.l3)} chunks on disk)"
        )
