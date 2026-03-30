"""
hardware/ — Real Hardware Cache Backend
========================================
Drop-in replacement for the simulated CacheSimulator.
Uses actual GPU VRAM (L1), CPU pinned RAM (L2), and NVMe disk (L3).

Usage:
    from env.hardware import HardwareCache, HardwareCacheEnv, HardwareConfig
"""

from .hardware_config import HardwareConfig
from .hardware_cache import HardwareCache
from .hardware_cache_env import HardwareCacheEnv
