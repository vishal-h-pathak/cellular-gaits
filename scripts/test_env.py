"""Smoke test for the env wrapper with two dummy controllers.

Confirms:
    1. FlyEnv constructs and reset() returns a sensible thorax position.
    2. rollout(zeros) runs end-to-end with a finite fitness.
    3. rollout(sinusoid) produces a *different* fitness — i.e., the
       actuators visibly respond to the controller signal (not stuck on
       a constant pose).

No mp4 is rendered here (saves time). The presence of a non-zero
delta in fitness between the two controllers is the verification.

Run from repo root:

    uv run python scripts/test_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cellular_gaits.env import (
    CONTROL_DT_S,
    DEFAULT_N_STEPS,
    N_ACTUATORS,
    FlyEnv,
)


def zeros_policy(t: int) -> np.ndarray:
    return np.zeros(N_ACTUATORS, dtype=np.float64)


def sinusoid_policy_factory(seed: int = 0) -> "callable":
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=N_ACTUATORS)
    freq_hz = 4.0
    amp = 0.4

    def policy(t: int) -> np.ndarray:
        time_s = t * CONTROL_DT_S
        return amp * np.sin(2.0 * np.pi * freq_hz * time_s + phase)

    return policy


def main() -> None:
    env = FlyEnv()
    obs0 = env.reset()
    print(f"reset: thorax={obs0['thorax_xyz']}, time={obs0['time']:.3f}s")
    print(f"adaptive z_threshold = {env._z_threshold:.5f}")

    fit_zeros, traj_zeros = env.rollout(zeros_policy, n_steps=DEFAULT_N_STEPS)
    print(
        f"zeros:    fitness={fit_zeros:.4f}, "
        f"forward_dx={traj_zeros['forward_dx']:.4f}, "
        f"n_below={traj_zeros['n_below']}/{traj_zeros['n_steps']}"
    )

    sinu = sinusoid_policy_factory(seed=0)
    fit_sin, traj_sin = env.rollout(sinu, n_steps=DEFAULT_N_STEPS)
    print(
        f"sinusoid: fitness={fit_sin:.4f}, "
        f"forward_dx={traj_sin['forward_dx']:.4f}, "
        f"n_below={traj_sin['n_below']}/{traj_sin['n_steps']}"
    )

    assert np.isfinite(fit_zeros) and np.isfinite(fit_sin)
    assert traj_zeros["thorax_xyz"].shape == (DEFAULT_N_STEPS + 1, 3)
    assert traj_sin["joint_targets"].shape == (DEFAULT_N_STEPS, N_ACTUATORS)
    if abs(fit_sin - fit_zeros) < 1e-6:
        print("WARN: zeros and sinusoid produced identical fitness — actuators may not be responding.")
    else:
        print(f"OK: actuators respond to controller signal (delta={fit_sin - fit_zeros:.4f}).")


if __name__ == "__main__":
    main()
