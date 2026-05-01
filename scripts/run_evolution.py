"""CLI for the CMA-ES evolution loop.

Examples:
    # Tiny end-to-end smoke check (Step 5 verification)
    uv run python scripts/run_evolution.py --pop 8 --gens 5

    # Full v1 run
    uv run python scripts/run_evolution.py --pop 32 --gens 50

    # Override seed / sigma / checkpoint cadence
    uv run python scripts/run_evolution.py --seed 1 --sigma 0.2 --checkpoint-every 10

Outputs:
    outputs/fitness_log_<run_id>.csv
    checkpoints/<run_id>/gen_NN.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.env import DEFAULT_N_STEPS
from cellular_gaits.evolve import EvolveConfig, run_evolution


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", "--popsize", dest="pop", type=int, default=32)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--rollout-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--run-id", type=str, default=None)
    args = p.parse_args()

    cfg = EvolveConfig(
        seed=args.seed,
        popsize=args.pop,
        n_gens=args.gens,
        sigma_init=args.sigma,
        rollout_steps=args.rollout_steps,
        checkpoint_every=args.checkpoint_every,
    )

    best_fit, best_params, run_dir = run_evolution(cfg, run_id=args.run_id)
    print(
        f"\n[evolve] DONE  best_fit={best_fit:.4f}  "
        f"params_norm={float((best_params**2).sum() ** 0.5):.4f}  "
        f"checkpoints -> {run_dir.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
