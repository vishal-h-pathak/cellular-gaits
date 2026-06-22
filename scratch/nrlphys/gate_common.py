"""Shared deterministic harness for the N-RL-PHYS physics gates.

A single deterministic open-loop policy is used everywhere so:
  * the no-obstacle A/B (Gate 4) is a true byte-exact comparison (same targets,
    same seed, same n_steps), and
  * the blocking / tunneling gates drive the fly with a *reproducible* forward
    gait (no trained controller needed).

The policy is a fixed sinusoidal tripod-ish pattern whose per-actuator
amplitudes/phases come from a seeded RNG, so it is identical across runs and
across the pre-/post-edit env.
"""

from __future__ import annotations

import numpy as np

from cellular_gaits.env import CONTROL_DT_S, N_ACTUATORS

POLICY_SEED = 20260622
GAIT_FREQ_HZ = 8.0  # leg cycle frequency


def make_policy(amp: float = 0.6, seed: int = POLICY_SEED):
    """Return a deterministic open-loop policy ``policy(t) -> (42,) in [-1,1]``.

    Per-actuator phase offsets are seeded; amplitude is shared. The pattern is a
    plain forward-walking oscillation — enough to make the fly walk roughly +x.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=N_ACTUATORS)
    # Bias a subset of actuators to give a forward push (deterministic mask).
    bias = rng.uniform(-0.15, 0.15, size=N_ACTUATORS)

    def policy(t: int) -> np.ndarray:
        ph = 2.0 * np.pi * GAIT_FREQ_HZ * (t * CONTROL_DT_S)
        out = amp * np.sin(ph + phases) + bias
        return np.clip(out, -1.0, 1.0)

    return policy
