"""CLI for the parallel closed-loop CMA-ES evolution (C2-A).

Examples:
    # Speedup benchmark only (sequential vs parallel on one population)
    uv run python scripts/run_evolution_closed_loop.py --benchmark --pop 16

    # Short validation run (the (b) plan: build everything, validate first)
    uv run python scripts/run_evolution_closed_loop.py --pop 16 --gens 12 --benchmark

    # Full run
    uv run python scripts/run_evolution_closed_loop.py --pop 32 --gens 50

Outputs:
    outputs/fitness_log_<run_id>.csv
    outputs/benchmark_<run_id>.json        (when --benchmark)
    checkpoints/<run_id>/gen_NN.npz/.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.env import DEFAULT_N_STEPS  # noqa: E402
from cellular_gaits.evolve import _utc_run_id  # noqa: E402
from cellular_gaits.evolve_closed_loop import (  # noqa: E402
    ClosedLoopConfig,
    benchmark_speedup,
    run_evolution_closed_loop,
)

OUT_DIR = ROOT / "outputs"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", "--popsize", dest="pop", type=int, default=32)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--rollout-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=0, help="0 -> cpu_count()-1")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--pert-magnitude", type=float, default=3.0)
    p.add_argument("--pert-duration-s", type=float, default=0.05)
    p.add_argument("--uneven-ground", action="store_true")
    p.add_argument("--terrain-seed", type=int, default=0)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Time sequential vs parallel eval before evolving; write JSON.",
    )
    p.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only the speedup benchmark, then exit (no evolution).",
    )
    args = p.parse_args()

    cfg = ClosedLoopConfig(
        seed=args.seed,
        popsize=args.pop,
        n_gens=args.gens,
        sigma_init=args.sigma,
        rollout_steps=args.rollout_steps,
        checkpoint_every=args.checkpoint_every,
        eval_seeds=tuple(args.eval_seeds),
        pert_magnitude=args.pert_magnitude,
        pert_duration_s=args.pert_duration_s,
        uneven_ground=args.uneven_ground,
        terrain_seed=args.terrain_seed,
        n_workers=args.workers,
    )

    run_id = args.run_id or _utc_run_id()

    if args.benchmark or args.benchmark_only:
        bench = benchmark_speedup(cfg)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bpath = OUT_DIR / f"benchmark_{run_id}.json"
        with bpath.open("w") as fp:
            json.dump(bench, fp, indent=2)
        print(f"[run] benchmark -> {bpath.relative_to(ROOT)}")
        if args.benchmark_only:
            return

    best_fit, best_params, run_dir = run_evolution_closed_loop(
        cfg, run_id=run_id, resume_from=args.resume_from
    )
    print(f"[run] done. best_fit={best_fit:.4f}  ckpts -> {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
