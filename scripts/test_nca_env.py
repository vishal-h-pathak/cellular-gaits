"""Step 4: wire NCA into the env, run one rollout with random params, print fitness.

This is the integration test before CMA-ES enters the picture.

Run from repo root:

    uv run python scripts/test_nca_env.py [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.env import DEFAULT_N_STEPS, FlyEnv
from cellular_gaits.nca import NCA


def build_policy(nca: NCA, seed: int):
    state = NCA.init_state(seed=seed)

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)
        return nca.motor_targets(state)

    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    nca = NCA()
    rng = np.random.default_rng(args.seed)
    vec = rng.normal(0.0, 0.3, size=nca.n_params).astype(np.float64)
    nca.set_params(vec)
    print(f"NCA: {nca.n_params} params, seed={args.seed}")

    env = FlyEnv()
    policy = build_policy(nca, seed=args.seed)
    fitness, traj = env.rollout(policy, n_steps=DEFAULT_N_STEPS)

    print(
        f"rollout: fitness={fitness:.4f}  "
        f"forward_dx={traj['forward_dx']:.4f}  "
        f"n_below={traj['n_below']}/{traj['n_steps']}  "
        f"z_threshold={traj['z_threshold']:.4f}"
    )
    print(
        f"thorax start={traj['thorax_xyz'][0]} end={traj['thorax_xyz'][-1]}"
    )


if __name__ == "__main__":
    main()
