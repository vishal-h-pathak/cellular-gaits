"""Gate 3 — cost is bounded by the solver cap (obstacles-only).

A fly pinned against a solid post generates a large contact set. MuJoCo's default
Newton solver does a dense Cholesky per iteration; the obstacle env caps it to the
CG solver (no Cholesky). This measures a worst-case PINNED rollout under each
solver, confirms the cap is applied ONLY when obstacles are present, and reports
per-reset vs per-step cost so wave-2 knows the physical-task throughput ceiling.

The pin is produced by a sustained forward thorax force pressing the fly into the
post (many leg/body geoms in contact), held for the whole rollout.
"""

from __future__ import annotations

import time

import mujoco as mj
import numpy as np

from cellular_gaits.env import (
    OBSTACLE_SOLVER,
    OBSTACLE_SOLVER_ITERATIONS,
    OBSTACLE_SOLVER_LS_ITERATIONS,
    PHYSICS_PER_CONTROL,
    FlyEnv,
    ObstacleField,
)

NEWTON = int(mj.mjtSolver.mjSOL_NEWTON)  # uncapped default
PRESS_FORCE = 6.0
OBSTACLE_X = 3.0  # close, so the press pins quickly
N_CONTROL = 300


def pinned_rollout(uncapped: bool) -> dict:
    env = FlyEnv(obstacles=ObstacleField(centers=((OBSTACLE_X, 0.0),), radius=2.0))
    opt = env.sim.mj_model.opt
    if uncapped:
        opt.solver = NEWTON
        opt.iterations = 100
        opt.ls_iterations = 50
    solver = int(opt.solver)
    iters = int(opt.iterations)

    t0 = time.perf_counter()
    env.reset()
    reset_s = time.perf_counter() - t0

    tid = env._thorax_body_id
    contact_steps = 0
    ncon_peak = 0
    t0 = time.perf_counter()
    for _ in range(N_CONTROL):
        env.sim.mj_data.xfrc_applied[tid, :3] = (PRESS_FORCE, 0.0, 0.0)
        for _ in range(PHYSICS_PER_CONTROL):
            env.sim.step()
        ncon_peak = max(ncon_peak, int(env.sim.mj_data.ncon))
        if env._obstacle_contact_active():
            contact_steps += 1
    step_s = time.perf_counter() - t0
    finite = bool(np.isfinite(env._thorax_xyz()).all())
    return {
        "uncapped": uncapped, "solver": solver, "iterations": iters,
        "reset_s": reset_s, "step_s": step_s,
        "per_step_ms": 1000.0 * step_s / N_CONTROL,
        "contact_steps": contact_steps, "ncon_peak": ncon_peak, "finite": finite,
    }


def main():
    print("=" * 78)
    print("Solver cap is OBSTACLES-ONLY:")
    e0 = FlyEnv()  # no obstacles
    e1 = FlyEnv(obstacles=ObstacleField(centers=((3.0, 0.0),), radius=2.0))
    print(f"  no-obstacle env  solver={int(e0.sim.mj_model.opt.solver)} "
          f"iters={int(e0.sim.mj_model.opt.iterations)}  (MuJoCo/FlyGym default)")
    print(f"  obstacle env     solver={int(e1.sim.mj_model.opt.solver)} "
          f"iters={int(e1.sim.mj_model.opt.iterations)} "
          f"ls={int(e1.sim.mj_model.opt.ls_iterations)}  "
          f"(expect CG={OBSTACLE_SOLVER}, {OBSTACLE_SOLVER_ITERATIONS}/"
          f"{OBSTACLE_SOLVER_LS_ITERATIONS})")

    print("=" * 78)
    print(f"PINNED rollout ({N_CONTROL} control steps, sustained press into post):")
    capped = pinned_rollout(uncapped=False)
    uncapped = pinned_rollout(uncapped=True)
    for tag, r in [("CAPPED (CG)", capped), ("UNCAPPED (Newton)", uncapped)]:
        print(f"  {tag:18} solver={r['solver']} iters={r['iterations']:>3}  "
              f"reset={r['reset_s']:.3f}s  rollout={r['step_s']:.3f}s  "
              f"per_step={r['per_step_ms']:.3f}ms  ncon_peak={r['ncon_peak']:>3}  "
              f"in_contact={r['contact_steps']}/{N_CONTROL}  finite={r['finite']}")
    speedup = uncapped["step_s"] / capped["step_s"] if capped["step_s"] else float("nan")
    print(f"  uncapped/capped rollout-time ratio = {speedup:.2f}x")
    secs_per_rollout_4s = capped["per_step_ms"] * 1000 / 1000.0  # informational
    print(f"  per-reset = {capped['reset_s']*1000:.1f} ms   "
          f"per-step (capped) = {capped['per_step_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
