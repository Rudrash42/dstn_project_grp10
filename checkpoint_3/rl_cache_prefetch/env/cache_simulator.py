"""
3-Tier Cache Simulator (L1 GPU / L2 CPU / L3 Disk)
===================================================
Lightweight Python-only simulation of LMCache's cache hierarchy.
No GPU required — operates on chunk IDs and tracks which tier holds each chunk.
"""

from __future__ import annotations

import collections
from typing import Tuple, List, Optional

from .tier_config import TierConfig


class CacheSimulator:
    """Simulates LMCache's L1 (GPU) / L2 (CPU) / L3 (Disk) hierarchy."""

    def __init__(self, config: Optional[TierConfig] = None):
        self.cfg = config or TierConfig()

        # OrderedDict for LRU: most-recently-used at the END.
        # Each value = chunk_size_bytes (constant, but stored for flexibility)
        self.l1: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.l2: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.l3: collections.OrderedDict[int, int] = collections.OrderedDict()

        # Track total bytes in each tier
        self.l1_bytes = 0
        self.l2_bytes = 0
        self.l3_bytes = 0

        # Access history: list of recently-accessed chunk IDs
        self.access_history: List[int] = []

    # ─── Reset ────────────────────────────────────────────────

    def reset(self):
        """Clear all tiers and history."""
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        self.l1_bytes = 0
        self.l2_bytes = 0
        self.l3_bytes = 0
        self.access_history.clear()

    # ─── Core operations ──────────────────────────────────────

    def _evict_lru(self, tier: collections.OrderedDict,
                   tier_capacity: int, tier_bytes: int,
                   chunk_size: int) -> Tuple[int, List[int]]:
        """
        Evict LRU entries from *tier* until there's room for one chunk.
        Returns (new_tier_bytes, list_of_evicted_chunk_ids).
        """
        evicted = []
        while tier_bytes + chunk_size > tier_capacity and tier:
            oldest_id, oldest_size = tier.popitem(last=False)
            tier_bytes -= oldest_size
            evicted.append(oldest_id)
        return tier_bytes, evicted

    def insert_chunks(self, chunk_ids: List[int]):
        """
        Insert newly-computed chunks into L1.  Overflow evicts LRU from L1
        into L2, and from L2 into L3 (mirroring LMCache's waterfall).
        Chunks already present in any tier are skipped (touch only).
        """
        for cid in chunk_ids:
            size = self.cfg.chunk_size_bytes

            # Already in some tier? Just touch it.
            if cid in self.l1:
                self.l1.move_to_end(cid)
                continue
            if cid in self.l2:
                # Promote L2 → L1
                del self.l2[cid]
                self.l2_bytes -= size
                self._insert_l1(cid, size)
                continue
            if cid in self.l3:
                # Promote L3 → L1
                del self.l3[cid]
                self.l3_bytes -= size
                self._insert_l1(cid, size)
                continue

            # Brand new chunk → L1
            self._insert_l1(cid, size)

    def _insert_l1(self, cid: int, size: int):
        """Insert a chunk into L1, evicting LRU to L2 if needed."""
        # Evict from L1 if full
        self.l1_bytes, evicted = self._evict_lru(
            self.l1, self.cfg.l1_capacity_bytes, self.l1_bytes, size
        )
        # Evicted L1 chunks go to L2
        for eid in evicted:
            self._insert_l2(eid, size)

        self.l1[cid] = size
        self.l1_bytes += size

    def _insert_l2(self, cid: int, size: int):
        """Insert a chunk into L2, evicting LRU to L3 if needed."""
        if cid in self.l2:
            self.l2.move_to_end(cid)
            return

        self.l2_bytes, evicted = self._evict_lru(
            self.l2, self.cfg.l2_capacity_bytes, self.l2_bytes, size
        )
        for eid in evicted:
            self._insert_l3(eid, size)

        self.l2[cid] = size
        self.l2_bytes += size

    def _insert_l3(self, cid: int, size: int):
        """Insert a chunk into L3, evicting LRU if full."""
        if cid in self.l3:
            self.l3.move_to_end(cid)
            return

        self.l3_bytes, _ = self._evict_lru(
            self.l3, self.cfg.l3_capacity_bytes, self.l3_bytes, size
        )
        self.l3[cid] = size
        self.l3_bytes += size

    # ─── Access (read) ────────────────────────────────────────

    def access_chunk(self, chunk_id: int) -> Tuple[str, float]:
        """
        Simulate reading a chunk.  Returns (tier_name, latency_ms).
        If the chunk is in a tier, it's a hit.  Otherwise, cold miss.
        """
        self.access_history.append(chunk_id)

        if chunk_id in self.l1:
            self.l1.move_to_end(chunk_id)
            return "L1", self.cfg.l1_hit_latency_ms

        if chunk_id in self.l2:
            latency = self.cfg.l2_hit_latency_ms
            # Promote to L1
            size = self.l2.pop(chunk_id)
            self.l2_bytes -= size
            self._insert_l1(chunk_id, size)
            return "L2", latency

        if chunk_id in self.l3:
            latency = self.cfg.l3_hit_latency_ms
            # Promote to L1
            size = self.l3.pop(chunk_id)
            self.l3_bytes -= size
            self._insert_l1(chunk_id, size)
            return "L3", latency

        # Cold miss — chunk must be computed from scratch
        latency = self.cfg.cold_compute_per_chunk_ms
        self.insert_chunks([chunk_id])
        return "MISS", latency

    def access_chunks(self, chunk_ids: List[int]) -> Tuple[float, dict]:
        """
        Access multiple chunks.  Returns total latency and per-tier counts.
        """
        total_ms = 0.0
        tier_counts = {"L1": 0, "L2": 0, "L3": 0, "MISS": 0}
        for cid in chunk_ids:
            tier, lat = self.access_chunk(cid)
            total_ms += lat
            tier_counts[tier] += 1
        return total_ms, tier_counts

    # ─── Prefetch (proactive migration) ───────────────────────

    def prefetch(self, chunk_id: int) -> float:
        """
        Move a chunk from L3 → L2.  Returns the migration cost in ms.
        If the chunk is not in L3 (or already in L1/L2), returns 0.
        """
        if chunk_id in self.l1 or chunk_id in self.l2:
            return 0.0  # already warm

        if chunk_id in self.l3:
            size = self.l3.pop(chunk_id)
            self.l3_bytes -= size
            self._insert_l2(chunk_id, size)
            return self.cfg.prefetch_l3_to_l2_ms

        return 0.0  # not in cache at all, can't prefetch

    # ─── Observation helpers ──────────────────────────────────

    def get_l3_candidates(self, k: int) -> List[int]:
        """
        Return the top-k chunks in L3, ordered by most-recently-used first
        (i.e., the ones most likely to be needed again).
        """
        # OrderedDict end = most recent, so reverse
        all_ids = list(self.l3.keys())
        return list(reversed(all_ids[-k:]))

    def get_all_candidates(self, k: int) -> List[int]:
        """
        Return top-k candidate chunks from L2+L3 combined (for prefetch
        decisions). Prioritises L3 chunks (they benefit more from prefetch).
        """
        l3_ids = list(reversed(list(self.l3.keys())))
        l2_ids = list(reversed(list(self.l2.keys())))
        combined = l3_ids + l2_ids
        return combined[:k]

    def get_stats(self) -> Tuple[float, float, float]:
        """
        Returns (l1_usage_fraction, l2_usage_mb, l3_usage_mb).
        """
        l1_frac = self.l1_bytes / self.cfg.l1_capacity_bytes if self.cfg.l1_capacity_bytes > 0 else 0
        l2_mb = self.l2_bytes / (1024 * 1024)
        l3_mb = self.l3_bytes / (1024 * 1024)
        return l1_frac, l2_mb, l3_mb

    def get_recent_accesses(self, n: int) -> List[int]:
        """Return the last *n* chunk IDs accessed."""
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
