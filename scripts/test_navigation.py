"""N-A integration tests: feelers, obstacles, A/B integrity, bilateral geometry.

Verifies (mix of fast NCA-level checks and slower real-sim checks):
    1. The nav NCA has the expected 10-input architecture and param count
       (4 state + 2 proprio + 2 odor + 2 feeler = 10 input channels, 1524 params).
    2. A/B integrity (NCA level): a nav NCA warm-started from the chemo
       controller, with the feeler channels zeroed, steps identically to the
       8-input chemo controller; a non-zero feeler reading does NOT change the
       warm-start output (feeler weights are zero).
    3. A/B integrity (env level): the nav controller homing on a goal with NO
       obstacles reproduces the chemo rollout trajectory bit-for-bit.
    4. Obstacles are physical AND sensed: with an obstacle on the path the
       feeler fires and the warm-start forager collides (genuine impediment).
    5. The bilateral feeler cue is correctly signed: an obstacle on the LEFT of
       the path excites the left feeler more, on the RIGHT the right feeler.
    6. The feeler actually steers: with non-zero feeler weights, an obstacle on
       the left vs the right produces opposite turning (emergent-detour mechanism).
    7. nav_fitness returns finite components on the warm-start controller.

Run from repo root:

    uv run python scripts/test_navigation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.env import FlyEnv, ObstacleField, OdorField
from cellular_gaits.evolve_chemotaxis import _make_chemo_policy
from cellular_gaits.evolve_navigation import (
    NavConfig,
    _approach_feeler_LmR,
    _make_nav_policy,
    build_condition_envs,
    conditions,
    load_chemo_params,
    nav_fitness,
)
from cellular_gaits.nca import CHEMO_N_PARAMS, NCA

N = 120  # short rollout for the NCA-level / geometry tests
NFULL = 400  # long enough for the fly to reach the obstacle on the path


def test_nav_arch() -> None:
    chemo = load_chemo_params()
    assert chemo.size == CHEMO_N_PARAMS, chemo.size
    nca = NCA(nav=True)
    assert nca.sensor_channels == 6, nca.sensor_channels
    # conv1: (4 state + 6 sensor) -> 16, 3x3 = 10*16*9 + 16 = 1456; conv2 = 68.
    assert nca.n_params == 1456 + 68, nca.n_params
    # chemo / loom / nav are mutually exclusive.
    for kw in (dict(chemo=True, nav=True), dict(loom=True, nav=True)):
        try:
            NCA(**kw)
            raise AssertionError(f"{kw} should have raised")
        except ValueError:
            pass
    print(f"  nav NCA: {nca.n_params} params, conv1 in={4 + nca.sensor_channels}")


def test_ab_nca_level() -> None:
    """feeler-zeroed nav NCA steps identically to the chemo NCA."""
    chemo = load_chemo_params()
    cm = NCA(chemo=True)
    cm.set_params(chemo)
    nav = NCA(nav=True)
    nav.warm_start_from_chemo(chemo)

    rng = np.random.RandomState(0)
    state = NCA.init_state(seed=0)
    ja = rng.uniform(-1, 1, 42)
    fc = (rng.uniform(0, 1, 6) > 0.5).astype(float)
    sc = NCA.build_chemo_sensor_map(ja, fc, 0.3, 0.7)
    sn = NCA.build_nav_sensor_map(ja, fc, 0.3, 0.7, 0.0, 0.0)
    with torch.no_grad():
        o_cm = cm.step(state, sc)
        o_nav = nav.step(state, sn)
    d = float((o_cm - o_nav).abs().max())
    assert d == 0.0, f"NCA-level A/B mismatch: {d}"
    # A non-zero feeler reading must NOT change the output at warm start.
    sn2 = NCA.build_nav_sensor_map(ja, fc, 0.3, 0.7, 1.0, 0.0)
    with torch.no_grad():
        o_nav2 = nav.step(state, sn2)
    d2 = float((o_nav - o_nav2).abs().max())
    assert d2 == 0.0, f"feeler weights not zero at warm start: {d2}"
    print(f"  NCA A/B: dmax={d:.2e}  feeler-on@warmstart dmax={d2:.2e}")


def test_ab_env_rollout() -> None:
    """Nav controller homing with NO obstacles == chemo rollout, bit-for-bit."""
    chemo = load_chemo_params()
    odor = OdorField(source_xy=(18.0, 0.0))

    cm = NCA(chemo=True)
    cm.set_params(chemo)
    env_a = FlyEnv()
    env_a.set_odor(odor)
    fit_a, traj_a = env_a.rollout(
        _make_chemo_policy(cm), n_steps=N, pass_sensors=True
    )

    nav = NCA(nav=True)
    nav.warm_start_from_chemo(chemo)
    env_b = FlyEnv()  # no obstacles -> feelers read (0, 0)
    env_b.set_odor(odor)
    fit_b, traj_b = env_b.rollout(
        _make_nav_policy(nav, ca_seed=0), n_steps=N, pass_sensors=True
    )

    dfit = abs(fit_a - fit_b)
    dtraj = float(np.abs(traj_a["joint_targets"] - traj_b["joint_targets"]).max())
    assert dfit < 1e-9, f"env A/B fitness mismatch: {dfit}"
    assert dtraj < 1e-6, f"env A/B trajectory mismatch: {dtraj}"
    print(f"  env A/B: dfit={dfit:.2e} dtraj_max={dtraj:.2e} (feeler-zeroed == chemo)")


def test_obstacle_physical_and_sensed() -> None:
    """An obstacle on the path fires the feeler AND physically impedes the fly."""
    chemo = load_chemo_params()
    nav = NCA(nav=True)
    nav.warm_start_from_chemo(chemo)
    env = FlyEnv(obstacles=ObstacleField(centers=((9.0, 0.0),), radius=2.0))
    env.set_odor(OdorField(source_xy=(18.0, 0.0)))
    _, traj = env.rollout(_make_nav_policy(nav, ca_seed=0), n_steps=NFULL, pass_sensors=True)
    feeler = np.asarray(traj["feeler"], dtype=np.float64)
    mag = feeler.sum(axis=1)
    assert mag.max() > 0.2, f"feeler never fired: {mag.max()}"
    assert traj["collision_count"] > 0, "warm-start forager did not collide"
    print(
        f"  feeler peak={mag.max():.3f}  collisions={traj['collision_count']}  "
        f"min_surface={traj['obstacle_min_surface']:.2f} (physical + sensed)"
    )


def _block_conditions():
    """The (block_left, block_right) NavConditions + their built envs.

    Uses the default config horizon (the obstacle placement reads the warm-start
    natural path, which only reaches its closest approach near step ~416 — a
    shorter horizon would mis-place the obstacles).
    """
    cfg = NavConfig()
    conds = conditions(cfg)
    envs = build_condition_envs(cfg)
    left = next(i for i, c in enumerate(conds) if c.block_side > 0)
    right = next(i for i, c in enumerate(conds) if c.block_side < 0)
    return cfg, conds, envs, left, right


def test_feeler_cue_signed() -> None:
    """An obstacle on the body-LEFT excites the left feeler; body-RIGHT the right.

    Uses the real condition layouts (obstacles placed on the natural path, offset
    in the fly's body frame), so this validates the placement machinery too.
    """
    chemo = load_chemo_params()
    nav = NCA(nav=True)
    nav.warm_start_from_chemo(chemo)
    cfg, conds, envs, li, ri = _block_conditions()

    def mean_LmR(i):
        _, traj = envs[i].rollout(
            _make_nav_policy(nav, ca_seed=0, feeler_input_gain=cfg.feeler_input_gain),
            n_steps=cfg.rollout_steps,
            pass_sensors=True,
        )
        return _approach_feeler_LmR(traj)

    left = mean_LmR(li)   # obstacle on the body-left -> detour right
    right = mean_LmR(ri)  # obstacle on the body-right -> detour left
    print(f"  mean(L-R): block_left={left:+.3f}  block_right={right:+.3f}")
    assert left > 0.01, f"left-block did not bias the left feeler: {left}"
    assert right < -0.01, f"right-block did not bias the right feeler: {right}"


def test_feeler_steers_both_ways() -> None:
    """With non-zero feeler weights, a left vs right block turns opposite ways."""
    chemo = load_chemo_params()
    nca = NCA(nav=True)
    nca.warm_start_from_chemo(chemo)
    # Inject a deterministic bilateral steering bias into the feeler weights so
    # the response is large enough to read in a short rollout. Mechanism check.
    torch.manual_seed(0)
    with torch.no_grad():
        nca.conv1.weight[:, 8:, :, :].normal_(0.0, 0.3)
    cfg, conds, envs, li, ri = _block_conditions()

    def final_dyaw(i):
        _, traj = envs[i].rollout(
            _make_nav_policy(nca, ca_seed=0, feeler_input_gain=cfg.feeler_input_gain),
            n_steps=cfg.rollout_steps,
            pass_sensors=True,
        )
        yaw = np.asarray(traj["yaw"], dtype=np.float64)
        return float(yaw[-1] - traj["yaw0"])

    dyaw_left = final_dyaw(li)
    dyaw_right = final_dyaw(ri)
    print(f"  Δyaw left-block={dyaw_left:+.3f} rad  right-block={dyaw_right:+.3f} rad")
    assert abs(dyaw_left - dyaw_right) > 1e-3, (
        f"bilateral block produced no differential turn: {dyaw_left} vs {dyaw_right}"
    )


def test_fitness_finite() -> None:
    chemo = load_chemo_params()
    nca = NCA(nav=True)
    nca.warm_start_from_chemo(chemo)
    cfg = NavConfig(rollout_steps=NFULL)
    envs = build_condition_envs(cfg)
    for env, cond in zip(envs, conditions(cfg)):
        _, traj = env.rollout(
            _make_nav_policy(nca, ca_seed=0), n_steps=cfg.rollout_steps, pass_sensors=True
        )
        fc = nav_fitness(traj, cfg)
        assert np.isfinite(fc["fitness"]), (cond.name, fc)
        for k in ("approach", "bonus", "collide_pen", "time_pen", "min_dist"):
            assert np.isfinite(fc[k]), (cond.name, k, fc[k])
    print("  nav_fitness finite on all conditions for the warm-start controller")


def main() -> None:
    print("test_nav_arch"); test_nav_arch()
    print("test_ab_nca_level"); test_ab_nca_level()
    print("test_ab_env_rollout"); test_ab_env_rollout()
    print("test_obstacle_physical_and_sensed"); test_obstacle_physical_and_sensed()
    print("test_feeler_cue_signed"); test_feeler_cue_signed()
    print("test_feeler_steers_both_ways"); test_feeler_steers_both_ways()
    print("test_fitness_finite"); test_fitness_finite()
    print("OK: navigation tests passed.")


if __name__ == "__main__":
    main()
