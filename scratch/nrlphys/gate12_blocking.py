"""Gates 1 & 2 — blocking is real, and no tunneling.

Drive the fly straight into an on-path obstacle with a simple forward actuation
(a constant forward force on the thorax — no trained controller) and check:

  Gate 1  the fly physically stops/deflects and does NOT pass through; MuJoCo
          shows fly<->obstacle contacts; final thorax x is short of / around the
          post, never beyond the far surface.
  Gate 2  sweep approach FORCE (=> a range of approach speeds) and obstacle
          RADIUS; the thorax never ends on the far side without a registered
          contact. Report the worst case tried.

The drive is applied at the physics rate via xfrc_applied (legs held at the
neutral keyframe), stepping PHYSICS_PER_CONTROL ticks per control step exactly
like FlyEnv.step, so the contact runs at the real physics/control rates.
"""

from __future__ import annotations

import numpy as np

from cellular_gaits.env import PHYSICS_PER_CONTROL, FlyEnv, ObstacleField

OBSTACLE_X = 8.0  # post center ahead of the fly (spawn x ~ 0)


def drive_into_post(
    force: float,
    radius: float,
    n_control: int = 400,
    obstacle_x: float = OBSTACLE_X,
):
    """Push the thorax +x with a constant force into a post at (obstacle_x, 0)."""
    env = FlyEnv(obstacles=ObstacleField(centers=((obstacle_x, 0.0),), radius=radius))
    env.reset()
    tid = env._thorax_body_id
    surface_x = obstacle_x - radius  # near surface
    far_x = obstacle_x + radius

    xs = [float(env._thorax_xyz()[0])]
    contact_steps = 0
    speeds = []
    prev_x = xs[0]
    first_contact_x = None
    for _ in range(n_control):
        env.sim.mj_data.xfrc_applied[tid, :3] = (force, 0.0, 0.0)
        for _ in range(PHYSICS_PER_CONTROL):
            env.sim.step()
        x = float(env._thorax_xyz()[0])
        # approach speed over the control step (units/s); only count pre-contact.
        in_contact = env._obstacle_contact_active()
        if not in_contact and first_contact_x is None and x < surface_x:
            speeds.append((x - prev_x) / (PHYSICS_PER_CONTROL * 1e-4))
        if in_contact:
            contact_steps += 1
            if first_contact_x is None:
                first_contact_x = x
        xs.append(x)
        prev_x = x

    xs = np.asarray(xs)
    final_x = float(xs[-1])
    peak_x = float(xs.max())
    peak_speed = float(max(speeds)) if speeds else 0.0
    blocked = peak_x <= far_x  # thorax never reached the far side
    tunneled = (peak_x > far_x) and (contact_steps == 0)
    return {
        "force": force,
        "radius": radius,
        "surface_x": surface_x,
        "far_x": far_x,
        "peak_x": peak_x,
        "final_x": final_x,
        "peak_speed": peak_speed,
        "contact_steps": contact_steps,
        "blocked": bool(blocked),
        "tunneled": bool(tunneled),
    }


def main():
    print("=" * 78)
    print("GATE 1 — blocking is real (force=15, radius=2.0, drive straight in)")
    r = drive_into_post(force=15.0, radius=2.0)
    print(f"  near surface x = {r['surface_x']:.2f}, far surface x = {r['far_x']:.2f}")
    print(f"  peak thorax x  = {r['peak_x']:.3f}  (must stay <= far surface)")
    print(f"  final thorax x = {r['final_x']:.3f}")
    print(f"  peak approach speed (pre-contact) = {r['peak_speed']:.2f} units/s")
    print(f"  control steps in real fly<->obstacle contact = {r['contact_steps']}")
    print(f"  BLOCKED (never beyond far surface) = {r['blocked']}")
    print(f"  contacts registered = {r['contact_steps'] > 0}")

    print("=" * 78)
    print("GATE 2 — no tunneling: sweep force (=> speed) x radius")
    forces = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    radii = [1.0, 1.5, 2.0, 3.0]
    rows = []
    worst = None
    print(f"  {'force':>7} {'radius':>7} {'peak_v':>8} {'peak_x':>8} "
          f"{'far_x':>7} {'contacts':>9} {'blocked':>8} {'tunneled':>9}")
    for rad in radii:
        for f in forces:
            res = drive_into_post(force=f, radius=rad, n_control=500)
            rows.append(res)
            print(f"  {f:7.0f} {rad:7.2f} {res['peak_speed']:8.2f} "
                  f"{res['peak_x']:8.3f} {res['far_x']:7.2f} "
                  f"{res['contact_steps']:9d} {str(res['blocked']):>8} "
                  f"{str(res['tunneled']):>9}")
            if worst is None or res["peak_speed"] > worst["peak_speed"]:
                worst = res
    any_tunnel = any(r["tunneled"] for r in rows)
    all_blocked = all(r["blocked"] for r in rows)
    print("-" * 78)
    print(f"  ANY tunneled = {any_tunnel}   ALL blocked = {all_blocked}")
    print(f"  worst case tried: force={worst['force']:.0f} radius={worst['radius']:.2f} "
          f"peak_speed={worst['peak_speed']:.2f} u/s -> blocked={worst['blocked']} "
          f"tunneled={worst['tunneled']} contacts={worst['contact_steps']}")
    return 0 if (not any_tunnel and all_blocked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
