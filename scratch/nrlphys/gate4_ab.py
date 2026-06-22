"""Gate 4 — no-obstacle A/B: post-edit no-obstacle rollout == pre-edit baseline.

Asserts max|Δ| == 0 on trajectory, targets, yaw, z, and scalar fitness, proving
the obstacle change did not perturb the no-obstacle (chemo/loom/closed-loop/v1)
physics path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_capture import run_noobstacle  # noqa: E402

BASE = Path(__file__).resolve().parent / "baseline_noobstacle.npz"


def main() -> int:
    ref = np.load(BASE)
    fitness, traj = run_noobstacle()

    deltas = {
        "thorax_xyz": np.abs(traj["thorax_xyz"] - ref["thorax_xyz"]).max(),
        "joint_targets": np.abs(traj["joint_targets"] - ref["joint_targets"]).max(),
        "yaw": np.abs(traj["yaw"] - ref["yaw"]).max(),
        "thorax_z": np.abs(traj["thorax_z"] - ref["thorax_z"]).max(),
        "fitness": abs(fitness - float(ref["fitness"][0])),
    }
    print("Gate 4 — no-obstacle A/B (post-edit vs pre-edit baseline):")
    for k, v in deltas.items():
        print(f"  max|Δ {k}| = {v:.3e}")
    ok = all(v == 0.0 for v in deltas.values())
    print(f"  BYTE-EXACT: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
