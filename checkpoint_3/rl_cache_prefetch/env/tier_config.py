"""
Tier configuration — capacities, latencies, and KV geometry.
Loaded from ppo_config.yaml or used with defaults.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TierConfig:
    """All cache-tier parameters in one place."""

    # ── KV geometry (Qwen2.5-0.5B) ────────────────────────────
    chunk_size_tokens: int = 256
    kv_bytes_per_token: int = 12288        # 2×24L×2H×64d×2B
    chunk_size_bytes: int = 3_145_728      # 256 tok × 12288 B

    # ── Tier capacities (bytes) ────────────────────────────────
    l1_capacity_bytes: int = 6 * 1024 * 1024         # 6 MB  (fits ~2 chunks)
    l2_capacity_bytes: int = 9 * 1024 * 1024         # 9 MB  (fits ~3 chunks)
    l3_capacity_bytes: int = 5120 * 1024 * 1024      # 5 GB

    # ── Latencies (milliseconds) ──────────────────────────────
    l1_hit_latency_ms: float = 0.1
    l2_hit_latency_ms: float = 0.25
    l3_hit_latency_ms: float = 6.0
    cold_compute_per_chunk_ms: float = 30.0
    prefetch_l3_to_l2_ms: float = 6.0
    prefetch_l2_to_l1_ms: float = 0.1

    # ── Reward knobs ──────────────────────────────────────────
    alpha: float = 1.0       # weight for time saved
    beta: float = 0.01       # penalty per MB migrated
    gamma_reward: float = 0.1  # penalty per unused prefetch

    # ── Observation ───────────────────────────────────────────
    embed_dim: int = 384
    max_candidate_chunks: int = 16
    max_eviction_candidates: int = 16
    history_len: int = 5

    @property
    def obs_dim(self) -> int:
        return self.embed_dim + 3 + self.max_candidate_chunks + self.max_eviction_candidates

    @property
    def l1_capacity_chunks(self) -> int:
        return self.l1_capacity_bytes // self.chunk_size_bytes

    @property
    def l2_capacity_chunks(self) -> int:
        return self.l2_capacity_bytes // self.chunk_size_bytes

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TierConfig":
        """Load config from a YAML file, falling back to defaults."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Map YAML keys to dataclass fields
        mapping = {
            "chunk_size_tokens": "chunk_size_tokens",
            "kv_bytes_per_token": "kv_bytes_per_token",
            "chunk_size_bytes": "chunk_size_bytes",
            "l1_hit_latency_ms": "l1_hit_latency_ms",
            "l2_hit_latency_ms": "l2_hit_latency_ms",
            "l3_hit_latency_ms": "l3_hit_latency_ms",
            "cold_compute_per_chunk_ms": "cold_compute_per_chunk_ms",
            "prefetch_l3_to_l2_ms": "prefetch_l3_to_l2_ms",
            "prefetch_l2_to_l1_ms": "prefetch_l2_to_l1_ms",
            "alpha": "alpha",
            "beta": "beta",
            "gamma_reward": "gamma_reward",
            "embed_dim": "embed_dim",
            "max_candidate_chunks": "max_candidate_chunks",
            "max_eviction_candidates": "max_eviction_candidates",
            "history_len": "history_len",
        }

        kwargs = {}
        for yaml_key, field_name in mapping.items():
            if yaml_key in raw:
                kwargs[field_name] = raw[yaml_key]

        # Handle capacity in MB → bytes
        if "l1_capacity_mb" in raw:
            kwargs["l1_capacity_bytes"] = int(raw["l1_capacity_mb"] * 1024 * 1024)
        if "l2_capacity_mb" in raw:
            kwargs["l2_capacity_bytes"] = int(raw["l2_capacity_mb"] * 1024 * 1024)
        if "l3_capacity_mb" in raw:
            kwargs["l3_capacity_bytes"] = int(raw["l3_capacity_mb"] * 1024 * 1024)

        return cls(**kwargs)
