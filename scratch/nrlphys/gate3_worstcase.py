"""Gate 3 worst-case probe — can a pinned fly generate a LARGE contact set?

The CG cap only pays off when Newton's dense Cholesky is stressed by a big
contact set. This tries to maximise the fly<->obstacle contact set: a tight
"cage" of posts around the spawn plus random leg flailing (an RL-exploration-like
candidate), and measures ncon and capped(CG) vs uncapped(Newton) per-step cost.
"""

from __future__ import annotations

import time

import mujoco as mj
import numpy as np

from cellular_gaits.env import (
    N_ACTUATORS,
    PHYSICS_PER_CONTROL,
    FlyEnv,
    ObstacleField,
)

NEWTON = int(mj.mjtSolver.mjSOL_NEWTON)
N_CONTROL = 200

# A tight ring of posts around the spawn (0,0): the fly is boxed in.
CAGE = tuple(
    (float(2.6 * np.cos(a)), float(2.6 * np.sin(a)))
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
)


def run(uncapped: bool, flail: bool) -> dict:
    env = FlyEnv(obstacles=ObstacleField(centers=CAGE, radius=1.2))
    opt = env.sim.mj_model.opt
    if uncapped:
        opt.solver = NEWTON
        opt.iterations = 100
        opt.ls_iterations = 50
    env.reset()
    tid = env._thorax_body_id
    rng = np.random.default_rng(0)
    ncon_peak = 0
    contact_steps = 0
    t0 = time.perf_counter()
    for t in range(N_CONTROL):
        if flail:
            ctrl = rng.uniform(-1, 1, N_ACTUATORS) * 3.14
            env.sim.set_actuator_inputs(env.sim and "nmf", None, ctrl) if False else None
            # drive actuators directly (bypass env.step bookkeeping)
            from flygym.compose import ActuatorType
            env.sim.set_actuator_inputs("nmf", ActuatorType.POSITION, ctrl)
        # squeeze the fly outward into a post
        env.sim.mj_data.xfrc_applied[tid, :3] = (4.0, 2.0, 0.0)
        for _ in range(PHYSICS_PER_CONTROL):
            env.sim.step()
        ncon_peak = max(ncon_peak, int(env.sim.mj_data.ncon))
        if env._obstacle_contact_active():
            contact_steps += 1
    dt = time.perf_counter() - t0
    return {"solver": int(opt.solver), "per_step_ms": 1000 * dt / N_CONTROL,
            "ncon_peak": ncon_peak, "contact_steps": contact_steps,
            "finite": bool(np.isfinite(env._thorax_xyz()).all())}


def main():
    for flail in (False, True):
        cap = run(uncapped=False, flail=flail)
        unc = run(uncapped=True, flail=flail)
        ratio = unc["per_step_ms"] / cap["per_step_ms"]
        print(f"flail={flail}")
        print(f"  CAPPED(CG)     ncon_peak={cap['ncon_peak']:>3} "
              f"per_step={cap['per_step_ms']:.3f}ms contacts={cap['contact_steps']} "
              f"finite={cap['finite']}")
        print(f"  UNCAPPED(NEWT) ncon_peak={unc['ncon_peak']:>3} "
              f"per_step={unc['per_step_ms']:.3f}ms contacts={unc['contact_steps']} "
              f"finite={unc['finite']}")
        print(f"  uncapped/capped per-step ratio = {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
