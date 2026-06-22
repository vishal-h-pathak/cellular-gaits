"""Gate 2 — no tunneling, controlled head-on approach speed.

Cleanly separates *closing speed* from *driving force*: impose a forward thorax
velocity v on the freejoint each physics tick UNTIL the first fly<->obstacle
contact, then RELEASE (the contact constraint alone governs the collision — we
never override velocity during contact). Sweep v (from the realistic gait top
speed up to far beyond it) and obstacle radius, and check:

  * non-penetration: the thorax center never enters the cylinder interior
    (min surface-distance stays >= ~0),
  * head-on stop: the thorax does not emerge on the far side along the centreline,
  * clean stop: bounded rebound, no NaN/instability,
  * report the worst case and the tunneling-onset speed if any.
"""

from __future__ import annotations

import numpy as np

from cellular_gaits.env import PHYSICS_PER_CONTROL, FlyEnv, ObstacleField

# freejoint translational DoFs are qvel[0:3] (dofadr 0). See probe in session.
VX_DOF = 0


def coast_into_post(v: float, radius: float, obstacle_x: float = 6.0,
                    n_control: int = 200) -> dict:
    env = FlyEnv(obstacles=ObstacleField(centers=((obstacle_x, 0.0),), radius=radius))
    env.reset()
    data = env.sim.mj_data
    far_x = obstacle_x + radius
    released = False
    xs, dpen = [], []
    nan = False
    for _ in range(n_control):
        for _ in range(PHYSICS_PER_CONTROL):
            if not released:
                data.qvel[VX_DOF] = v  # hold closing speed until first contact
            env.sim.step()
            if env._obstacle_contact_active():
                released = True  # let the constraint govern from here
        p = env._thorax_xyz()
        if not np.isfinite(p).all():
            nan = True
            break
        xs.append(float(p[0]))
        dpen.append(float(np.hypot(p[0] - obstacle_x, p[1]) - radius))
    xs = np.asarray(xs)
    dpen = np.asarray(dpen)
    if xs.size == 0:
        return {"v": v, "radius": radius, "nan": True, "blocked": False,
                "tunneled": True, "min_pen": float("nan"), "peak_x": float("nan"),
                "final_x": float("nan"), "rebound": float("nan"),
                "contacted": False}
    peak_x = float(xs.max())
    final_x = float(xs[-1])
    min_pen = float(dpen.min())  # <0 => thorax inside the post
    rebound = float(peak_x - final_x)  # how far it backed off after peak
    contacted = bool(released)
    # head-on: emerging beyond the far surface near the centreline == tunneled.
    tunneled = bool(peak_x > far_x or min_pen < -0.05 or nan)
    blocked = bool((not tunneled) and contacted and (min_pen >= -0.05))
    return {"v": v, "radius": radius, "nan": nan, "contacted": contacted,
            "min_pen": min_pen, "peak_x": peak_x, "final_x": final_x,
            "far_x": far_x, "rebound": rebound, "blocked": blocked,
            "tunneled": tunneled}


def main():
    speeds = [10.0, 30.0, 75.0, 120.0, 200.0, 400.0, 800.0, 1600.0]
    radii = [1.0, 2.0, 3.0]
    print("Gate 2 — controlled head-on coast (gait top speed ~118 u/s peak)")
    print(f"  {'v(u/s)':>8} {'radius':>7} {'min_pen':>8} {'peak_x':>8} {'far_x':>7} "
          f"{'rebound':>8} {'contact':>8} {'blocked':>8} {'tunneled':>9} {'nan':>5}")
    rows = []
    for rad in radii:
        for v in speeds:
            r = coast_into_post(v, rad)
            rows.append(r)
            print(f"  {v:8.0f} {rad:7.2f} {r['min_pen']:8.3f} {r['peak_x']:8.3f} "
                  f"{r.get('far_x', float('nan')):7.2f} {r['rebound']:8.3f} "
                  f"{str(r['contacted']):>8} {str(r['blocked']):>8} "
                  f"{str(r['tunneled']):>9} {str(r['nan']):>5}")
    # tunneling onset = lowest speed that tunneled at any radius
    tun = [r["v"] for r in rows if r["tunneled"]]
    realistic = [r for r in rows if r["v"] <= 120.0]
    print("-" * 78)
    print(f"  realistic envelope (v<=120 u/s, >= gait top speed): "
          f"all blocked = {all(r['blocked'] for r in realistic)}, "
          f"any tunneled = {any(r['tunneled'] for r in realistic)}")
    print(f"  tunneling-onset speed (any radius) = "
          f"{min(tun) if tun else 'none up to 1600 u/s'}")
    ok = all(r["blocked"] for r in realistic) and not any(r["tunneled"] for r in realistic)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
