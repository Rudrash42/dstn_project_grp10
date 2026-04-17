"""
MultiTraceEnv — Gymnasium wrapper for multi-workload PPO training.

Wraps N pre-built CacheEnv instances.  On every reset() it picks one
at random, ensuring the policy is trained on diverse access patterns
(prefix-sharing, RAG, multi-turn) rather than over-fitting one trace.

This is a drop-in replacement anywhere a single CacheEnv is used.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np

from .cache_env import CacheEnv


class MultiTraceEnv(gym.Env):
    """
    Randomly selects one of several CacheEnv instances on each reset().

    Parameters
    ----------
    envs : list[CacheEnv]
        Pre-constructed environments, one per trace / workload.
        All must share the same observation_space and action_space.

    Example
    -------
    >>> envs = [CacheEnv(path, cfg, embs) for path, embs in zip(traces, embeddings)]
    >>> train_env = MultiTraceEnv(envs)
    >>> model = PPO("MlpPolicy", train_env, ...)
    """

    metadata = {"render_modes": []}

    def __init__(self, envs: List[CacheEnv]):
        super().__init__()
        if not envs:
            raise ValueError("MultiTraceEnv requires at least one environment.")

        self._envs = envs
        self._current: CacheEnv = envs[0]

        # All child envs must share the same spaces
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    # ── Gymnasium interface ──────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Pick a random trace for the next episode and reset it."""
        self._current = random.choice(self._envs)
        return self._current.reset(seed=seed, options=options)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return self._current.step(action)

    def render(self) -> Any:
        return self._current.render()

    def close(self):
        for env in self._envs:
            env.close()
