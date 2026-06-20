"""Standalone unit test for the NCA module — no Flygym, no CMA-ES.

Verifies:
    1. Constructor produces <=1000 params (spec target).
    2. init_state shape is (1, 4, 8, 8) and is deterministic given a seed.
    3. step() preserves shape and clamps output to [-1, 1].
    4. flatten_params -> set_params -> flatten_params is identity.
    5. motor_targets returns 42 values from channel-0 of the 7x6 sub-grid.
    6. Two NCAs with identical params produce identical step outputs.

Run from repo root:

    uv run python scripts/test_nca.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.nca import (
    CHANNELS,
    GRID_H,
    GRID_W,
    MOTOR_COLS,
    MOTOR_ROWS,
    N_LEGS,
    N_MOTORS,
    NCA,
    SENSOR_SHAPE,
    STATE_SHAPE,
    V1_N_PARAMS,
)


def _zero_sensors():
    return torch.zeros(*SENSOR_SHAPE)


def test_param_count() -> int:
    nca = NCA()
    n = nca.n_params
    assert n <= 1000, f"NCA has {n} params, exceeds 1000"
    return n


def test_init_state_shape_and_determinism() -> None:
    s1 = NCA.init_state(seed=0)
    s2 = NCA.init_state(seed=0)
    s3 = NCA.init_state(seed=1)
    assert tuple(s1.shape) == STATE_SHAPE
    assert torch.equal(s1, s2), "init_state(seed=0) is not deterministic"
    assert not torch.equal(s1, s3), "init_state ignored seed change"
    assert (s1.abs() <= 0.1 + 1e-6).all()


def test_step_shape_and_clamp() -> None:
    nca = NCA()
    s = NCA.init_state(seed=0)
    out = nca.step(s)
    assert tuple(out.shape) == STATE_SHAPE
    assert (out >= -1.0).all() and (out <= 1.0).all()


def test_param_roundtrip() -> None:
    nca = NCA()
    v0 = nca.flatten_params()
    rng = np.random.default_rng(0)
    v1 = rng.normal(0, 0.5, size=v0.shape).astype(np.float64)
    nca.set_params(v1)
    v2 = nca.flatten_params()
    np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-6)


def test_motor_targets_shape_and_source() -> None:
    nca = NCA()
    s = NCA.init_state(seed=0)
    m = nca.motor_targets(s)
    assert m.shape == (N_MOTORS,)
    expected = (
        s[0, 0, :MOTOR_ROWS, :MOTOR_COLS]
        .reshape(-1)
        .numpy()
        .astype(np.float64)
    )
    np.testing.assert_allclose(m, expected)


def test_two_ncas_with_same_params_match() -> None:
    a = NCA()
    b = NCA()
    b.set_params(a.flatten_params())
    s = NCA.init_state(seed=0)
    assert torch.allclose(a.step(s), b.step(s), atol=1e-7)


def test_gain_default_is_identity() -> None:
    """gain=1.0 must reproduce the pre-gain rule bit-for-bit.

    The pre-gain rule was ``tanh(conv1(s))``; with the knob it is
    ``tanh(gain * conv1(s))``. At gain=1.0 the multiply is exact, so a
    multi-tick autonomous rollout must match the reference rule exactly.
    """
    nca = NCA(gain=1.0)
    rng = np.random.default_rng(7)
    nca.set_params(rng.normal(0, 0.5, size=nca.n_params))
    assert nca.gain == 1.0

    state = NCA.init_state(seed=0)
    ref = state.clone()
    z = _zero_sensors()
    for _ in range(50):
        state = nca.step(state)
        # reference = the old formula, no gain term at all; sensors zeroed
        ref_in = torch.cat([ref, z], dim=1)
        ref = torch.clamp(nca.conv2(torch.tanh(nca.conv1(ref_in))), -1.0, 1.0)
        assert torch.equal(state, ref), "gain=1.0 diverged from pre-gain rule"


def test_gain_changes_dynamics() -> None:
    """A gain other than 1.0 must actually move the trajectory."""
    rng = np.random.default_rng(7)
    params = rng.normal(0, 0.5, size=NCA().n_params)
    s0 = NCA.init_state(seed=0)

    native = NCA(gain=1.0)
    native.set_params(params)
    hot = NCA(gain=3.0)
    hot.set_params(params)

    a, b = s0.clone(), s0.clone()
    for _ in range(20):
        a, b = native.step(a), hot.step(b)
    assert not torch.allclose(a, b, atol=1e-4), "gain=3.0 had no effect"


def test_zero_sensors_match_no_sensors() -> None:
    """Passing an all-zero sensor map equals passing none (A/B integrity)."""
    nca = NCA()
    rng = np.random.default_rng(3)
    nca.set_params(rng.normal(0, 0.5, size=nca.n_params))
    s = NCA.init_state(seed=0)
    a = nca.step(s)
    b = nca.step(s, sensors=_zero_sensors())
    assert torch.equal(a, b), "zero sensor map changed the output"


def test_default_init_ignores_sensors() -> None:
    """A default-constructed NCA (zero sensor weights) must be sensor-invariant."""
    nca = NCA()
    rng = np.random.default_rng(11)
    nca.set_params(rng.normal(0, 0.5, size=nca.n_params))
    # Re-zero sensor weights (set_params overwrote them); emulate fresh-from-v1.
    nca.warm_start_from_v1(nca.flatten_params()[: V1_N_PARAMS])
    s = NCA.init_state(seed=0)
    live = NCA.build_sensor_map(
        joint_angles_unit=rng.uniform(-1, 1, size=N_MOTORS),
        foot_contacts=rng.integers(0, 2, size=N_LEGS),
    )
    a = nca.step(s, sensors=None)
    b = nca.step(s, sensors=live)
    assert torch.equal(a, b), "zero-weight sensors leaked into the output"


def test_warm_start_reproduces_v1() -> None:
    """A closed-loop NCA warm-started from v1 weights, run sensor-blind over a
    multi-tick rollout, matches a 4-channel v1 rule exactly."""
    rng = np.random.default_rng(5)
    v1 = rng.normal(0, 0.5, size=V1_N_PARAMS).astype(np.float64)

    nca = NCA()
    nca.warm_start_from_v1(v1)

    # Reference 4-channel rule with the same v1 weights.
    c1w_n = 16 * CHANNELS * 9
    w1 = torch.from_numpy(v1[:c1w_n].reshape(16, CHANNELS, 3, 3).astype(np.float32))
    b1 = torch.from_numpy(v1[c1w_n : c1w_n + 16].astype(np.float32))
    off = c1w_n + 16
    w2 = torch.from_numpy(v1[off : off + CHANNELS * 16].reshape(CHANNELS, 16, 1, 1).astype(np.float32))
    b2 = torch.from_numpy(v1[off + CHANNELS * 16 :].astype(np.float32))

    import torch.nn.functional as F

    state = NCA.init_state(seed=0)
    ref = state.clone()
    for _ in range(60):
        state = nca.step(state)  # sensors blind
        h = torch.tanh(F.conv2d(ref, w1, b1, padding=1))
        ref = torch.clamp(F.conv2d(h, w2, b2), -1.0, 1.0)
        assert torch.equal(state, ref), "warm-start diverged from v1 rule"


def test_build_sensor_map_placement() -> None:
    ja = np.linspace(-1, 1, N_MOTORS)
    fc = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    m = NCA.build_sensor_map(ja, fc)
    assert tuple(m.shape) == SENSOR_SHAPE
    # joint angles in 7x6 block of channel 0
    np.testing.assert_allclose(
        m[0, 0, :MOTOR_ROWS, :MOTOR_COLS].reshape(-1).numpy(), ja, atol=1e-6
    )
    # contacts in bottom row of channel 1
    np.testing.assert_allclose(m[0, 1, 7, :N_LEGS].numpy(), fc, atol=1e-6)
    # channel 0 outside the motor block is zero
    assert float(m[0, 0, 7, :].abs().sum()) == 0.0


def main() -> None:
    n = test_param_count()
    test_init_state_shape_and_determinism()
    test_step_shape_and_clamp()
    test_param_roundtrip()
    test_motor_targets_shape_and_source()
    test_two_ncas_with_same_params_match()
    test_gain_default_is_identity()
    test_gain_changes_dynamics()
    test_zero_sensors_match_no_sensors()
    test_default_init_ignores_sensors()
    test_warm_start_reproduces_v1()
    test_build_sensor_map_placement()
    print(
        f"OK: NCA tests passed. params={n}, state={STATE_SHAPE}, "
        f"motor_grid={MOTOR_ROWS}x{MOTOR_COLS}={N_MOTORS}, "
        f"channels={CHANNELS}, grid={GRID_H}x{GRID_W}"
    )


if __name__ == "__main__":
    main()
