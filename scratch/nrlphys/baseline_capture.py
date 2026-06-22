"""Gate-4 baseline: capture a no-obstacle rollout trajectory BEFORE the edit.

Run this on the pre-edit env to record the reference trajectory; rerun
``gate4_ab.py`` on the post-edit env and assert max|Δ| == 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import make_policy  # noqa: E402

from cellular_gaits.env import DEFAULT_N_STEPS, FlyEnv  # noqa: E402

OUT = Path(__file__).resolve().parent / "baseline_noobstacle.npz"


def run_noobstacle():
    env = FlyEnv()  # no obstacles -> the byte-exact path
    policy = make_policy()
    fitness, traj = env.rollout(policy, n_steps=DEFAULT_N_STEPS)
    return fitness, traj


if __name__ == "__main__":
    fitness, traj = run_noobstacle()
    np.savez(
        OUT,
        thorax_xyz=traj["thorax_xyz"],
        joint_targets=traj["joint_targets"],
        yaw=traj["yaw"],
        thorax_z=traj["thorax_z"],
        fitness=np.array([fitness], dtype=np.float64),
        forward_dx=np.array([traj["forward_dx"]], dtype=np.float64),
        n_below=np.array([traj["n_below"]], dtype=np.int64),
    )
    print(f"baseline saved -> {OUT}")
    print(f"  fitness={fitness:.6f} forward_dx={traj['forward_dx']:.6f} "
          f"n_below={traj['n_below']} n_steps={traj['n_steps']}")
    print(f"  thorax_xyz shape={traj['thorax_xyz'].shape} "
          f"final={traj['thorax_xyz'][-1]}")
