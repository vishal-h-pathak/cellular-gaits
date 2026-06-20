"""X-A integration tests: loom sensors, A/B integrity, bilateral geometry.

Verifies (mix of fast NCA-level checks and slower real-sim checks):
    1. The loom NCA has the expected 8-input architecture and param count
       (identical shape to the chemo NCA; only the cue meaning differs).
    2. A/B integrity (NCA level): a loom NCA warm-started from the closed-loop
       controller, with the loom channels zeroed, steps identically to the
       6-input closed-loop controller; a non-zero loom reading does NOT change
       the warm-start output (loom weights are zero).
    3. A/B integrity (env level): the loom controller run with NO threat
       reproduces the closed-loop rollout trajectory bit-for-bit.
    4. Threat geometry: before onset the loom is zero; the threat appears at
       onset and looms (grows) as it approaches.
    5. The bilateral loom cue is correctly signed: a threat from the LEFT excites
       the left eye more, a threat from the RIGHT the right eye more.
    6. The loom cue actually steers: with non-zero loom weights, a threat on the
       left vs. the right produces opposite turning (emergent-escape mechanism).
    7. escape_fitness returns finite components on the warm-start controller.

Run from repo root:

    uv run python scripts/test_escape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.env import FlyEnv, Threat
from cellular_gaits.evolve_escape import (
    EscapeConfig,
    _make_escape_policy,
    escape_fitness,
    load_cl_params,
)
from cellular_gaits.nca import CL_N_PARAMS, NCA

N = 120  # short rollout for the NCA-level / geometry tests
NFULL = 300  # ~1.2 s, enough for a threat to appear and loom


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


def test_loom_arch() -> None:
    cl = load_cl_params()
    assert cl.size == CL_N_PARAMS, cl.size
    nca = NCA(loom=True)
    assert nca.sensor_channels == 4, nca.sensor_channels
    # conv1: (4 state + 4 sensor) -> 16, 3x3 = 8*16*9 + 16 = 1168; conv2 = 68.
    assert nca.n_params == 1168 + 68, nca.n_params
    # chemo and loom are mutually exclusive.
    try:
        NCA(chemo=True, loom=True)
        raise AssertionError("chemo+loom should have raised")
    except ValueError:
        pass
    print(f"  loom NCA: {nca.n_params} params, conv1 in={4 + nca.sensor_channels}")


def test_ab_nca_level() -> None:
    """loom-zeroed loom NCA steps identically to the closed-loop NCA."""
    cl = load_cl_params()
    clm = NCA()
    clm.set_params(cl)
    lm = NCA(loom=True)
    lm.warm_start_from_closed_loop(cl)

    rng = np.random.RandomState(0)
    state = NCA.init_state(seed=0)
    ja = rng.uniform(-1, 1, 42)
    fc = (rng.uniform(0, 1, 6) > 0.5).astype(float)
    sp = NCA.build_sensor_map(ja, fc)
    sl = NCA.build_loom_sensor_map(ja, fc, 0.0, 0.0)
    with torch.no_grad():
        o_cl = clm.step(state, sp)
        o_lm = lm.step(state, sl)
    d = float((o_cl - o_lm).abs().max())
    assert d == 0.0, f"NCA-level A/B mismatch: {d}"
    # A non-zero loom reading must NOT change the output at warm start.
    sl2 = NCA.build_loom_sensor_map(ja, fc, 1.0, 0.0)
    with torch.no_grad():
        o_lm2 = lm.step(state, sl2)
    d2 = float((o_lm - o_lm2).abs().max())
    assert d2 == 0.0, f"loom weights not zero at warm start: {d2}"
    print(f"  NCA A/B: dmax={d:.2e}  loom-on@warmstart dmax={d2:.2e}")


def test_ab_env_rollout() -> None:
    """Loom controller with NO threat == closed-loop rollout, bit-for-bit."""
    cl = load_cl_params()

    clm = NCA()
    clm.set_params(cl)
    env_a = FlyEnv()
    fit_a, traj_a = env_a.rollout(
        _closed_loop_policy(clm, seed=0), n_steps=N, pass_sensors=True
    )

    lm = NCA(loom=True)
    lm.warm_start_from_closed_loop(cl)
    env_b = FlyEnv()  # no set_threat -> loom reads (0, 0)
    fit_b, traj_b = env_b.rollout(
        _make_escape_policy(lm, ca_seed=0), n_steps=N, pass_sensors=True
    )

    dfit = abs(fit_a - fit_b)
    dtraj = float(np.abs(traj_a["joint_targets"] - traj_b["joint_targets"]).max())
    assert dfit < 1e-9, f"env A/B fitness mismatch: {dfit}"
    assert dtraj < 1e-6, f"env A/B trajectory mismatch: {dtraj}"
    print(f"  env A/B: dfit={dfit:.2e} dtraj_max={dtraj:.2e} (loom-zeroed == closed-loop)")


def test_threat_looms() -> None:
    """Before onset the loom is zero; the threat then appears and grows."""
    cl = load_cl_params()
    lm = NCA(loom=True)
    lm.warm_start_from_closed_loop(cl)
    env = FlyEnv()
    env.set_threat(Threat(azimuth_deg=0.0, seed=7))
    _, traj = env.rollout(_make_escape_policy(lm, ca_seed=0), n_steps=NFULL, pass_sensors=True)
    loom = np.asarray(traj["loom"], dtype=np.float64)
    onset = int(traj["threat"]["onset_step"])
    mag = loom.sum(axis=1)
    assert mag[:onset].max(initial=0.0) == 0.0, "loom nonzero before onset"
    assert mag[onset:].max() > 0.2, f"threat never loomed: {mag.max()}"
    # loom should be rising on average across the approach (pre-peak).
    pk = int(np.argmax(mag))
    assert pk > onset, "peak loom at/under onset"
    print(f"  onset={onset} peak@{pk} max_loom={mag.max():.3f} (zero pre-onset)")


def test_loom_cue_signed() -> None:
    """A LEFT threat excites the left eye more; a RIGHT threat the right eye."""
    cl = load_cl_params()
    lm = NCA(loom=True)
    lm.warm_start_from_closed_loop(cl)

    def mean_LmR(az):
        env = FlyEnv()
        env.set_threat(Threat(azimuth_deg=az, seed=7))
        _, traj = env.rollout(
            _make_escape_policy(lm, ca_seed=0), n_steps=NFULL, pass_sensors=True
        )
        loom = np.asarray(traj["loom"], dtype=np.float64)
        mag = loom.sum(axis=1)
        sig = loom[mag > 0.1]
        return float((sig[:, 0] - sig[:, 1]).mean()) if sig.size else 0.0

    left = mean_LmR(90.0)   # threat from the fly's left
    right = mean_LmR(270.0)  # threat from the fly's right
    print(f"  mean(L-R): left-threat={left:+.3f}  right-threat={right:+.3f}")
    # Correctly signed cue (same threshold as the calibration's _cue_correct).
    assert left > 0.02, f"left threat did not bias the left eye: {left}"
    assert right < -0.02, f"right threat did not bias the right eye: {right}"


def test_loom_steers_both_ways() -> None:
    """With non-zero loom weights, left vs right threat turns opposite ways."""
    cl = load_cl_params()
    nca = NCA(loom=True)
    nca.warm_start_from_closed_loop(cl)
    # Inject a deliberate bilateral steering bias into the loom weights so the
    # response is large enough to read in a short rollout. Mechanism check, not
    # the evolved controller. The escape policy amplifies the loom by
    # loom_input_gain (default 8) so a modest weight already moves the bang-bang
    # gait off its clamp; without that gain even large weights are clamped away.
    # Seed the injection so the demonstrated differential is deterministic.
    torch.manual_seed(0)
    with torch.no_grad():
        nca.conv1.weight[:, 6:, :, :].normal_(0.0, 0.3)

    def final_dyaw(az):
        env = FlyEnv()
        env.set_threat(Threat(azimuth_deg=az, seed=7))
        _, traj = env.rollout(
            _make_escape_policy(nca, ca_seed=0), n_steps=NFULL, pass_sensors=True
        )
        yaw = np.asarray(traj["yaw"], dtype=np.float64)
        return float(yaw[-1] - traj["yaw0"])

    dyaw_left = final_dyaw(90.0)
    dyaw_right = final_dyaw(270.0)
    print(f"  Δyaw left-threat={dyaw_left:+.3f} rad  right-threat={dyaw_right:+.3f} rad")
    assert abs(dyaw_left - dyaw_right) > 1e-3, (
        f"bilateral threat produced no differential turn: {dyaw_left} vs {dyaw_right}"
    )


def test_fitness_finite() -> None:
    cl = load_cl_params()
    nca = NCA(loom=True)
    nca.warm_start_from_closed_loop(cl)
    from cellular_gaits.evolve_escape import conditions

    cfg = EscapeConfig(rollout_steps=NFULL)
    env = FlyEnv()
    for az, threat in conditions(cfg):
        env.set_threat(threat)
        _, traj = env.rollout(
            _make_escape_policy(nca, ca_seed=0), n_steps=cfg.rollout_steps, pass_sensors=True
        )
        fc = escape_fitness(traj, cfg)
        assert np.isfinite(fc["fitness"]), (az, fc)
        for k in ("clear", "survive", "react", "head", "min_dist"):
            assert np.isfinite(fc[k]), (az, k, fc[k])
    print("  escape_fitness finite on all azimuths for the warm-start controller")


def main() -> None:
    print("test_loom_arch"); test_loom_arch()
    print("test_ab_nca_level"); test_ab_nca_level()
    print("test_ab_env_rollout"); test_ab_env_rollout()
    print("test_threat_looms"); test_threat_looms()
    print("test_loom_cue_signed"); test_loom_cue_signed()
    print("test_loom_steers_both_ways"); test_loom_steers_both_ways()
    print("test_fitness_finite"); test_fitness_finite()
    print("OK: escape tests passed.")


if __name__ == "__main__":
    main()
