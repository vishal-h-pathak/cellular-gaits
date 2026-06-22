"""Gates 1 & 2 (realistic gait) + collision_count distribution.

Uses the actual warm-start forager (chemo controller warm-started into the nav
NCA) — the real bang-bang gait the RL run will start from — to:
  * measure the gait's top translational speed (sets the safety margin for the
    tunneling sweep),
  * Gate 1: drive it onto an on-path obstacle and confirm it physically stops /
    deflects and never ends beyond the post, with real fly<->obstacle contacts,
  * report the real-contact collision_count distribution (plow-straight-in vs a
    clean/near-miss pass) so wave-2 can re-tune w_collide against real numbers.
"""

from __future__ import annotations

import numpy as np

from cellular_gaits.env import CONTROL_DT_S, FlyEnv, ObstacleField, OdorField
from cellular_gaits.evolve_navigation import (
    NavConfig,
    _make_nav_policy,
    build_condition_envs,
    conditions,
    load_chemo_params,
    nav_fitness,
)
from cellular_gaits.nca import NCA


def _forager():
    nav = NCA(nav=True)
    nav.warm_start_from_chemo(load_chemo_params())
    return nav


def gait_top_speed(n_steps: int = 1000) -> dict:
    """Peak / mean translational speed of the warm-start forager homing freely."""
    nav = _forager()
    env = FlyEnv()
    env.set_odor(OdorField(source_xy=(18.0, 0.0)))
    _, traj = env.rollout(_make_nav_policy(nav, ca_seed=0), n_steps=n_steps,
                          pass_sensors=True)
    xy = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    step_disp = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = step_disp / CONTROL_DT_S  # units/s per control step
    return {"peak": float(speed.max()), "mean": float(speed.mean()),
            "p95": float(np.percentile(speed, 95))}


def gate1_blocking() -> dict:
    """Forager driven onto an on-path post.

    The post is a finite-width cylinder and the forager is a homing controller, so
    a *blocked* fly deflects and navigates AROUND it. The correct no-tunnel test is
    therefore PENETRATION: the thorax center must never enter the cylinder
    interior (min surface-distance >= 0). 'x beyond the post' just means it went
    around the finite post, which is fine.
    """
    nav = _forager()
    obstacle_x, radius = 9.0, 2.0
    env = FlyEnv(obstacles=ObstacleField(centers=((obstacle_x, 0.0),), radius=radius))
    env.set_odor(OdorField(source_xy=(18.0, 0.0)))
    _, traj = env.rollout(_make_nav_policy(nav, ca_seed=0), n_steps=400,
                          pass_sensors=True)
    xy = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    d_center = np.linalg.norm(xy - np.array([obstacle_x, 0.0]), axis=1)
    min_penetration = float((d_center - radius).min())  # >=0 => never inside
    return {
        "obstacle_x": obstacle_x, "radius": radius,
        "min_surface_dist": float(traj["obstacle_min_surface"]),
        "min_penetration": min_penetration,
        "thorax_ever_inside_post": bool(min_penetration < 0.0),
        "collision_count": int(traj["collision_count"]),
        "finite": bool(np.isfinite(xy).all()),
        "blocked": bool(min_penetration >= 0.0 and traj["collision_count"] > 0),
    }


def collision_distribution() -> dict:
    """Real-contact collision_count across the trained block conditions + a
    near-miss reference, to bracket the plow-in vs clean-pass range."""
    nav = _forager()
    cfg = NavConfig()
    envs = build_condition_envs(cfg)
    conds = conditions(cfg)
    rows = []
    for env, cond in zip(envs, conds):
        _, traj = env.rollout(
            _make_nav_policy(nav, ca_seed=0, feeler_input_gain=cfg.feeler_input_gain),
            n_steps=cfg.rollout_steps, pass_sensors=True)
        fc = nav_fitness(traj, cfg)
        rows.append({"name": cond.name, "collisions": int(fc["collision_count"]),
                     "reached": bool(fc["reached"]),
                     "min_surface": float(traj["obstacle_min_surface"])})
    # clean reference: no obstacle at all -> exactly zero real contacts.
    env = FlyEnv()
    env.set_odor(OdorField(source_xy=(18.0, 0.0)))
    _, traj = env.rollout(_make_nav_policy(nav, ca_seed=0), n_steps=1000,
                          pass_sensors=True)
    clean = int(traj["collision_count"])
    return {"plow_in_rows": rows, "clean_noobstacle": clean}


def main():
    print("=" * 78)
    g = gait_top_speed()
    print(f"GAIT TOP SPEED (warm-start forager, free homing): "
          f"peak={g['peak']:.2f}  p95={g['p95']:.2f}  mean={g['mean']:.2f} units/s")

    print("=" * 78)
    print("GATE 1 — blocking is real (warm-start forager onto on-path post)")
    b = gate1_blocking()
    for k, v in b.items():
        print(f"  {k:22} = {v}")
    print("  (forager deflects and navigates AROUND the finite post; the test is "
          "non-penetration, not x-beyond-post)")

    print("=" * 78)
    print("COLLISION_COUNT DISTRIBUTION (real contacts)")
    d = collision_distribution()
    cols = [r["collisions"] for r in d["plow_in_rows"]]
    for r in d["plow_in_rows"]:
        print(f"  {r['name']:>22}  collisions={r['collisions']:>4}  "
              f"reached={r['reached']!s:>5}  min_surface={r['min_surface']:.2f}")
    print(f"  plow-straight-in range = [{min(cols)}, {max(cols)}]  "
          f"clean no-obstacle pass = {d['clean_noobstacle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
