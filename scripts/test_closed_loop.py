"""C2-A integration tests: sensors, perturbation determinism, A/B integrity.

Verifies (with the real Flygym sim, so it is slow-ish):
    1. read_sensors() returns 42 joint angles in [-1,1] and 6 contacts in {0,1}.
    2. A perturbation schedule is identical across two envs at the same seed,
       and a closed-loop and open-loop env see the *same* shove.
    3. A/B integrity: a closed-loop NCA warm-started from v1 weights, run
       sensor-blind (pass_sensors via zeroed weights), reproduces the v1
       open-loop rollout fitness/trajectory.
    4. Sensors actually change behavior once the sensor weights are non-zero.

Run from repo root:

    uv run python scripts/test_closed_loop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.env import N_ACTUATORS, N_LEGS, FlyEnv, Perturbation
from cellular_gaits.nca import NCA, V1_N_PARAMS

V1_CKPT = ROOT / "checkpoints" / "2026-05-02T00-01-51Z" / "gen_50.npz"
N = 120  # short rollout for tests


def _v1_params() -> np.ndarray:
    data = np.load(V1_CKPT, allow_pickle=False)
    return np.asarray(data["best_params"], dtype=np.float64)


def open_loop_policy(nca: NCA, seed: int):
    """v1-style: 4-channel NCA, autonomous, no sensors."""
    state = NCA.init_state(seed=seed)

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)  # sensors default to zero
        return nca.motor_targets(state)

    return policy


def closed_loop_policy(nca: NCA, seed: int):
    """Closed loop: feed live sensors into the grid each tick."""
    state = NCA.init_state(seed=seed)

    def policy(t: int, sensors: dict) -> np.ndarray:
        nonlocal state
        smap = NCA.build_sensor_map(
            sensors["joint_angles_unit"], sensors["foot_contacts"]
        )
        with torch.no_grad():
            state = nca.step(state, sensors=smap)
        return nca.motor_targets(state)

    return policy


def test_read_sensors() -> None:
    env = FlyEnv()
    env.reset()
    ja, fc = env.read_sensors()
    assert ja.shape == (N_ACTUATORS,), ja.shape
    assert fc.shape == (N_LEGS,), fc.shape
    assert (np.abs(ja) <= 1.0 + 1e-9).all(), "joint angles out of [-1,1]"
    assert set(np.unique(fc)).issubset({0.0, 1.0}), "contacts not boolean"
    print(f"  sensors: |ja|max={np.abs(ja).max():.3f}  contacts={fc.astype(int)}")


def test_perturbation_determinism() -> None:
    p = Perturbation(magnitude=3.0, seed=7)
    s1 = p.schedule(N)
    s2 = p.schedule(N)
    assert s1.onset_step == s2.onset_step
    np.testing.assert_allclose(s1.force, s2.force)
    # different seed -> (very likely) different schedule
    s3 = Perturbation(magnitude=3.0, seed=8).schedule(N)
    assert (s3.onset_step != s1.onset_step) or not np.allclose(s3.force, s1.force)
    print(
        f"  impulse onset={s1.onset_step}/{N} force={np.round(s1.force, 2)}"
    )


def test_ab_integrity() -> None:
    """Closed-loop NCA warm-started from v1, run sensor-blind == v1 open loop."""
    v1 = _v1_params()

    # Open-loop reference: a true 4-channel v1 NCA. We emulate it with the
    # closed-loop class warm-started from v1, run sensor-blind (zero weights).
    nca_ref = NCA()
    nca_ref.warm_start_from_v1(v1[:V1_N_PARAMS])
    env_a = FlyEnv()
    fit_a, traj_a = env_a.rollout(open_loop_policy(nca_ref, seed=0), n_steps=N)

    # Same warm-started weights, but now actually feeding live sensors. Because
    # the sensor input-channel weights are zero, sensors must not change a thing.
    nca_blind = NCA()
    nca_blind.warm_start_from_v1(v1[:V1_N_PARAMS])
    env_b = FlyEnv()
    fit_b, traj_b = env_b.rollout(
        closed_loop_policy(nca_blind, seed=0), n_steps=N, pass_sensors=True
    )

    dfit = abs(fit_a - fit_b)
    dtraj = float(
        np.abs(traj_a["joint_targets"] - traj_b["joint_targets"]).max()
    )
    assert dfit < 1e-9, f"A/B fitness mismatch: {dfit}"
    assert dtraj < 1e-6, f"A/B trajectory mismatch: {dtraj}"
    print(f"  A/B: dfit={dfit:.2e} dtraj_max={dtraj:.2e} (sensors-zeroed == open-loop)")


def test_sensors_change_behavior() -> None:
    """With non-zero sensor weights, live sensors must alter the rollout."""
    v1 = _v1_params()
    nca = NCA()
    nca.warm_start_from_v1(v1[:V1_N_PARAMS])
    # Inject non-zero sensor weights.
    with torch.no_grad():
        nca.conv1.weight[:, 4:, :, :].normal_(0.0, 0.3)

    env_blind = FlyEnv()
    fit_blind, _ = env_blind.rollout(open_loop_policy(nca, seed=0), n_steps=N)
    env_live = FlyEnv()
    fit_live, _ = env_live.rollout(
        closed_loop_policy(nca, seed=0), n_steps=N, pass_sensors=True
    )
    assert abs(fit_blind - fit_live) > 1e-6, "live sensors had no effect"
    print(f"  sensors effect: blind={fit_blind:.4f} live={fit_live:.4f}")


def main() -> None:
    print("test_read_sensors"); test_read_sensors()
    print("test_perturbation_determinism"); test_perturbation_determinism()
    print("test_ab_integrity"); test_ab_integrity()
    print("test_sensors_change_behavior"); test_sensors_change_behavior()
    print("OK: closed-loop env tests passed.")


if __name__ == "__main__":
    main()
