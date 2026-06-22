"""Reusable, policy-agnostic RL harness for the FlyEnv body (N-RL).

This package wraps the existing CMA-ES-era :class:`cellular_gaits.env.FlyEnv`
as a Gymnasium environment so a PPO loop (and, later, a FlyWire connectome
sub-circuit) can train the same body on the same tasks. It is **purely
additive**: nothing under ``cellular_gaits`` outside this package is touched, so
v1 / chemo / loom / closed-loop / nav (CMA-ES) behaviours stay bit-exact.

Layers
------
- :mod:`cellular_gaits.rl.env_base` — ``EmbodiedRLEnv``: the policy-agnostic
  Gymnasium wrapper with four pluggable hooks (observation builder,
  action->actuator map, reward fn, domain-randomization sampler).
- :mod:`cellular_gaits.rl.nav_env` — ``NavRLEnv`` / ``NavRLConfig``: the
  obstacle-navigation task built on top of ``EmbodiedRLEnv``.

The policy (``NCAPolicy``) and the PPO loop are *separate* modules (sessions
1b / 1c); this package is the env layer only and knows nothing about them.
"""

from __future__ import annotations

from .env_base import EmbodiedRLEnv
from .nav_env import NavRLConfig, NavRLEnv, make_nav_env

__all__ = [
    "EmbodiedRLEnv",
    "NavRLConfig",
    "NavRLEnv",
    "make_nav_env",
]
