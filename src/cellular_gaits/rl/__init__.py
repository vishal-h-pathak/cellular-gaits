"""Reusable, policy-agnostic RL harness for the embodied NCA controller.

This package is built across parallel N-RL sessions:
  * ``policies.py`` (this slice, N-RL-1b): ``NCAPolicy`` — the NCA as a
    state-threaded Gaussian PPO policy.
  * ``env_base.py`` / ``nav_env.py`` (N-RL-1a): the Gymnasium env + nav task.
  * ``ppo.py`` (N-RL-1c): the PPO loop.

MERGE NOTE: this ``__init__`` was created by the policy session because the
``rl/`` package did not exist yet. When the env/ppo sessions land, fold their
exports in here (re-export ``EmbodiedRLEnv``, the PPO entry point, etc.) rather
than overwriting — keep the ``NCAPolicy`` export.
"""

from __future__ import annotations

from .policies import NCAPolicy

__all__ = ["NCAPolicy"]
