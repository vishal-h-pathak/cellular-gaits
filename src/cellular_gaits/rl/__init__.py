"""cellular_gaits.rl — the policy-/env-agnostic RL harness.

This package is the reusable PPO stack for the project (the navigation task today,
the FlyWire connectome sub-circuit later). It is split across wave-1 sessions:

  - ``ppo.py``      — the PPO training loop (this session, ``feat/n-rl-ppo``).
  - ``env_base.py`` / ``nav_env.py`` — the Gymnasium env wrapper + nav task (env session).
  - ``policies.py`` — the ``NCAPolicy`` Gaussian policy (policy session).

This ``__init__`` is intentionally minimal so the three branches merge without
conflict. If another wave-1 branch already shipped an ``__init__.py`` with more
content, prefer theirs and re-add the ``ppo`` re-export below by hand.
"""

from cellular_gaits.rl.ppo import PPOConfig, train  # noqa: F401

__all__ = ["PPOConfig", "train"]
