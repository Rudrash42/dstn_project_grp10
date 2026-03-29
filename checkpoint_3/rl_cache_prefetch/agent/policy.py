"""
Custom policy network for stable-baselines3 PPO.
Small MLP: obs(403) → 64 → 64 → action(16).
"""

from __future__ import annotations

from stable_baselines3.common.policies import ActorCriticPolicy
from typing import List


def make_policy_kwargs(net_arch: List[int] = None) -> dict:
    """
    Create policy_kwargs dict for stable-baselines3 PPO.

    Parameters
    ----------
    net_arch : list[int]
        Hidden layer sizes for shared network.
        Default: [64, 64]

    Returns
    -------
    dict
        kwargs to pass to PPO(..., policy_kwargs=...)
    """
    if net_arch is None:
        net_arch = [64, 64]

    return {
        "net_arch": net_arch,
    }
