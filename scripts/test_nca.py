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
    N_MOTORS,
    NCA,
    STATE_SHAPE,
)


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


def main() -> None:
    n = test_param_count()
    test_init_state_shape_and_determinism()
    test_step_shape_and_clamp()
    test_param_roundtrip()
    test_motor_targets_shape_and_source()
    test_two_ncas_with_same_params_match()
    print(
        f"OK: NCA tests passed. params={n}, state={STATE_SHAPE}, "
        f"motor_grid={MOTOR_ROWS}x{MOTOR_COLS}={N_MOTORS}, "
        f"channels={CHANNELS}, grid={GRID_H}x{GRID_W}"
    )


if __name__ == "__main__":
    main()
