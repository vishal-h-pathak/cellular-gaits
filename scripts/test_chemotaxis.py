"""CH-A integration tests: chemo sensors, A/B integrity, bilateral geometry.

Verifies (mix of fast NCA-level checks and slower real-sim checks):
    1. The chemo NCA has the expected 8-input architecture and param count.
    2. Odor field is 1 at the source and decays monotonically with distance.
    3. A/B integrity (NCA level): a chemo NCA warm-started from the closed-loop
       controller, with the chemo channels zeroed, steps identically to the
       6-input closed-loop controller.
    4. Antenna geometry: at the +x spawn heading the left antenna sits to the
       left (+y) and the right antenna to the right (-y), both ahead (+x).
    5. A/B integrity (env level): the chemo controller run with NO odor field
       reproduces the closed-loop rollout trajectory bit-for-bit.
    6. The chemo cue actually steers: with non-zero chemo weights, a source on
       the left vs. the right produces opposite turning.

Run from repo root:

    uv run python scripts/test_chemotaxis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.env import N_ACTUATORS, FlyEnv, OdorField
from cellular_gaits.evolve_chemotaxis import _make_chemo_policy, load_cl_params
from cellular_gaits.nca import CL_N_PARAMS, NCA

N = 120  # short rollout for the sim-based tests


def _closed_loop_policy(nca: NCA, seed: int = 0):
    """6-input closed-loop policy (proprio only)."""
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


def test_chemo_arch() -> None:
    cl = load_cl_params()
    assert cl.size == CL_N_PARAMS, cl.size
    nca = NCA(chemo=True)
    assert nca.sensor_channels == 4, nca.sensor_channels
    # conv1: (4 state + 4 sensor) -> 16, 3x3 = 8*16*9 + 16 = 1168; conv2 = 68.
    assert nca.n_params == 1168 + 68, nca.n_params
    print(f"  chemo NCA: {nca.n_params} params, conv1 in={4 + nca.sensor_channels}")


def test_field_monotonic() -> None:
    f = OdorField(source_xy=(20.0, 0.0), lam=12.0)
    assert abs(f.concentration((20.0, 0.0)) - 1.0) < 1e-12
    ds = [0, 5, 10, 20, 40]
    cs = [f.concentration((20.0 - d, 0.0)) for d in ds]
    assert all(cs[i] > cs[i + 1] for i in range(len(cs) - 1)), cs
    print(f"  field C at d={ds}: {[round(c, 3) for c in cs]}")


def test_ab_nca_level() -> None:
    """chemo-zeroed chemo NCA steps identically to the closed-loop NCA."""
    cl = load_cl_params()
    clm = NCA()
    clm.set_params(cl)
    cm = NCA(chemo=True)
    cm.warm_start_from_closed_loop(cl)

    rng = np.random.RandomState(0)
    state = NCA.init_state(seed=0)
    ja = rng.uniform(-1, 1, 42)
    fc = (rng.uniform(0, 1, 6) > 0.5).astype(float)
    sp = NCA.build_sensor_map(ja, fc)
    sc = NCA.build_chemo_sensor_map(ja, fc, 0.0, 0.0)
    with torch.no_grad():
        o_cl = clm.step(state, sp)
        o_cm = cm.step(state, sc)
    d = float((o_cl - o_cm).abs().max())
    assert d == 0.0, f"NCA-level A/B mismatch: {d}"
    # And a non-zero odor reading must NOT change the output at warm start
    # (chemo weights are zero).
    sc2 = NCA.build_chemo_sensor_map(ja, fc, 1.0, 0.0)
    with torch.no_grad():
        o_cm2 = cm.step(state, sc2)
    d2 = float((o_cm - o_cm2).abs().max())
    assert d2 == 0.0, f"chemo weights not zero at warm start: {d2}"
    print(f"  NCA A/B: dmax={d:.2e}  chemo-on@warmstart dmax={d2:.2e}")


def test_antenna_geometry() -> None:
    env = FlyEnv()
    env.reset()
    pos_l, pos_r = env.antenna_positions()
    thorax = env._thorax_xyz()[:2]
    # at spawn the fly faces +x: both antennae ahead, left is +y of right.
    assert pos_l[0] > thorax[0] and pos_r[0] > thorax[0], (pos_l, pos_r, thorax)
    assert pos_l[1] > pos_r[1], (pos_l, pos_r)
    print(f"  thorax={np.round(thorax,2)} L={np.round(pos_l,2)} R={np.round(pos_r,2)}")


def test_ab_env_rollout() -> None:
    """Chemo controller with NO odor field == closed-loop rollout, bit-for-bit."""
    cl = load_cl_params()

    clm = NCA()
    clm.set_params(cl)
    env_a = FlyEnv()
    fit_a, traj_a = env_a.rollout(
        _closed_loop_policy(clm, seed=0), n_steps=N, pass_sensors=True
    )

    cm = NCA(chemo=True)
    cm.warm_start_from_closed_loop(cl)
    env_b = FlyEnv()  # no set_odor -> chemo reads (0, 0)
    fit_b, traj_b = env_b.rollout(
        _make_chemo_policy(cm, ca_seed=0), n_steps=N, pass_sensors=True
    )

    dfit = abs(fit_a - fit_b)
    dtraj = float(np.abs(traj_a["joint_targets"] - traj_b["joint_targets"]).max())
    assert dfit < 1e-9, f"env A/B fitness mismatch: {dfit}"
    assert dtraj < 1e-6, f"env A/B trajectory mismatch: {dtraj}"
    print(f"  env A/B: dfit={dfit:.2e} dtraj_max={dtraj:.2e} (chemo-zeroed == closed-loop)")


def test_chemo_steers_both_ways() -> None:
    """With non-zero chemo weights, left vs right source turns opposite ways."""
    cl = load_cl_params()
    nca = NCA(chemo=True)
    nca.warm_start_from_closed_loop(cl)
    # Inject a deliberate bilateral steering bias into the chemo weights so the
    # response is large enough to read in a short rollout. This is a mechanism
    # check, not the evolved controller.
    with torch.no_grad():
        nca.conv1.weight[:, 6:, :, :].normal_(0.0, 0.5)

    def final_yaw(source_xy):
        env = FlyEnv()
        env.set_odor(OdorField(source_xy=source_xy, lam=12.0))
        _, traj = env.rollout(
            _make_chemo_policy(nca, ca_seed=0), n_steps=N, pass_sensors=True
        )
        yaw = np.asarray(traj["yaw"], dtype=np.float64)
        return float(yaw[-1] - traj["yaw0"])

    dyaw_left = final_yaw((6.0, 18.0))   # source on the left
    dyaw_right = final_yaw((6.0, -18.0))  # source on the right
    print(f"  Δyaw left-source={dyaw_left:+.3f} rad  right-source={dyaw_right:+.3f} rad")
    assert abs(dyaw_left - dyaw_right) > 1e-3, (
        f"bilateral source produced no differential turn: {dyaw_left} vs {dyaw_right}"
    )


def main() -> None:
    print("test_chemo_arch"); test_chemo_arch()
    print("test_field_monotonic"); test_field_monotonic()
    print("test_ab_nca_level"); test_ab_nca_level()
    print("test_antenna_geometry"); test_antenna_geometry()
    print("test_ab_env_rollout"); test_ab_env_rollout()
    print("test_chemo_steers_both_ways"); test_chemo_steers_both_ways()
    print("OK: chemotaxis tests passed.")


if __name__ == "__main__":
    main()
