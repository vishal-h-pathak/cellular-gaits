"""CLI for the parallel chemotaxis CMA-ES evolution (CH-A).

Examples:
    # Calibration: gen-0 (warm-start) diagnostics over a (distance, lambda) grid
    uv run python scripts/run_evolution_chemotaxis.py --calibrate

    # Speedup benchmark only
    uv run python scripts/run_evolution_chemotaxis.py --benchmark-only --pop 16

    # Short validation run (build everything, validate first), with benchmark
    uv run python scripts/run_evolution_chemotaxis.py --pop 16 --gens 8 --benchmark

    # Full run
    uv run python scripts/run_evolution_chemotaxis.py --pop 24 --gens 50

Outputs:
    outputs/fitness_log_<run_id>.csv
    outputs/benchmark_<run_id>.json        (when --benchmark[-only])
    scratch/cha/calibration.json           (when --calibrate)
    checkpoints/<run_id>/gen_NN.npz/.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cellular_gaits.env import DEFAULT_N_STEPS, FlyEnv  # noqa: E402
from cellular_gaits.evolve import _utc_run_id, load_checkpoint  # noqa: E402
from cellular_gaits.evolve_chemotaxis import (  # noqa: E402
    ChemoConfig,
    benchmark_speedup,
    diagnose_params,
    run_evolution_chemotaxis,
    warm_start_x0,
)

OUT_DIR = ROOT / "outputs"
SCRATCH = ROOT / "scratch" / "cha"


def _cfg_from_args(args) -> ChemoConfig:
    kw = dict(
        seed=args.seed,
        popsize=args.pop,
        n_gens=args.gens,
        sigma_init=args.sigma,
        rollout_steps=args.rollout_steps,
        checkpoint_every=args.checkpoint_every,
        source_distance=args.distance,
        odor_lambda=args.lam,
        reach_radius=args.reach_radius,
        reach_bonus=args.reach_bonus,
        time_penalty=args.time_penalty,
        antenna_forward=args.antenna_forward,
        antenna_lateral=args.antenna_lateral,
        n_workers=args.workers,
    )
    if getattr(args, "azimuths", None):
        kw["azimuths_deg"] = tuple(args.azimuths)
    return ChemoConfig(**kw)


def _print_diag(rows: list[dict]) -> None:
    print(
        f"    {'az':>4} {'src':>14} {'d0':>6} {'dend':>6} {'dmin':>6} "
        f"{'appr':>6} {'reach':>5} {'dyaw':>7} {'sig0':>6} {'sigMax':>6}"
    )
    for r in rows:
        src = f"({r['source_xy'][0]:.0f},{r['source_xy'][1]:.0f})"
        print(
            f"    {r['azimuth_deg']:>4.0f} {src:>14} {r['d_start']:>6.1f} "
            f"{r['d_end']:>6.1f} {r['min_dist']:>6.1f} {r['approach']:>6.1f} "
            f"{str(r['reached']):>5} {r['dyaw_final']:>+7.3f} "
            f"{r['signal_start']:>6.3f} {r['signal_max']:>6.3f}"
        )


def calibrate(args) -> None:
    """Gen-0 (warm-start) diagnostics over a (distance, lambda) grid.

    The warm-start controller walks straight +x blind (chemo weights zero), so
    this tells us, BEFORE evolving: how far it travels in the rollout, whether
    the ahead-source is reachable, whether off-axis sources are NOT trivially
    reached (non-trivial), and how strong the bilateral cue |cL-cR| is along the
    path (learnable). We pick distance/lambda/reach where the cue is informative
    and the task is reachable-but-not-free.
    """
    distances = args.cal_distances
    lams = args.cal_lambdas
    x0 = warm_start_x0(ChemoConfig())
    env = FlyEnv(antenna_forward=args.antenna_forward, antenna_lateral=args.antenna_lateral)

    grid = []
    for dist in distances:
        for lam in lams:
            cfg = _cfg_from_args(args)
            cfg = ChemoConfig(
                **{**cfg.__dict__, "source_distance": dist, "odor_lambda": lam}
            )
            rows = diagnose_params(x0, cfg, env=env)
            ahead = next(r for r in rows if r["azimuth_deg"] == 0.0)
            off = [r for r in rows if r["azimuth_deg"] != 0.0]
            summary = {
                "distance": dist,
                "lambda": lam,
                "reach_radius": cfg.reach_radius,
                "ahead_reached": ahead["reached"],
                "ahead_min_dist": ahead["min_dist"],
                "offaxis_reached_frac": float(np.mean([r["reached"] for r in off])),
                "offaxis_mean_approach": float(np.mean([r["approach"] for r in off])),
                "mean_signal_max": float(np.mean([r["signal_max"] for r in rows])),
                "mean_signal_start": float(np.mean([r["signal_start"] for r in rows])),
                "walk_dx": float(ahead["d_start"] - ahead["d_end"]),
                "rows": rows,
            }
            grid.append(summary)
            print(
                f"[cal] dist={dist:>5.1f} lam={lam:>5.1f}  "
                f"ahead_reached={ahead['reached']!s:>5} ahead_dmin={ahead['min_dist']:>5.1f}  "
                f"offaxis_reached={summary['offaxis_reached_frac']:.2f}  "
                f"sigMax={summary['mean_signal_max']:.3f}  walk_dx={summary['walk_dx']:.1f}"
            )
            _print_diag(rows)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / "calibration.json"
    out.write_text(json.dumps({"grid": grid}, indent=2))
    print(f"[cal] wrote {out.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", "--popsize", dest="pop", type=int, default=24)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--rollout-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=0, help="0 -> cpu_count()-1")
    # task geometry
    p.add_argument("--distance", type=float, default=18.0)
    p.add_argument("--lam", "--lambda", dest="lam", type=float, default=12.0)
    p.add_argument("--reach-radius", type=float, default=3.0)
    p.add_argument("--reach-bonus", type=float, default=8.0)
    p.add_argument("--time-penalty", type=float, default=4.0)
    p.add_argument("--antenna-forward", type=float, default=1.0)
    p.add_argument("--antenna-lateral", type=float, default=0.6)
    p.add_argument(
        "--azimuths", type=float, nargs="+", default=None,
        help="Source azimuths in deg (default: 0 90 180 270).",
    )
    # calibration grid
    p.add_argument("--cal-distances", type=float, nargs="+", default=[12.0, 18.0, 25.0])
    p.add_argument("--cal-lambdas", type=float, nargs="+", default=[8.0, 12.0, 20.0])
    # modes
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--calibrate", action="store_true", help="Gen-0 grid diagnostics, then exit.")
    p.add_argument("--benchmark", action="store_true", help="Time seq vs parallel before evolving.")
    p.add_argument("--benchmark-only", action="store_true", help="Only benchmark, then exit.")
    p.add_argument("--diagnose-ckpt", type=Path, default=None, help="Diagnose a checkpoint best, then exit.")
    args = p.parse_args()

    run_id = args.run_id or _utc_run_id()
    cfg = _cfg_from_args(args)

    if args.calibrate:
        calibrate(args)
        return

    if args.diagnose_ckpt is not None:
        data = load_checkpoint(args.diagnose_ckpt)
        best = np.asarray(data["best_params"], dtype=np.float64)
        print(f"[diag] checkpoint best_fit={float(data['best_fit']):.4f}")
        rows = diagnose_params(best, cfg)
        _print_diag(rows)
        return

    if args.benchmark or args.benchmark_only:
        bench = benchmark_speedup(cfg, n_individuals=min(cfg.popsize, 16))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bpath = OUT_DIR / f"benchmark_{run_id}.json"
        bpath.write_text(json.dumps(bench, indent=2))
        print(f"[run] benchmark -> {bpath.relative_to(ROOT)}")
        if args.benchmark_only:
            return

    best_fit, best_params, run_dir = run_evolution_chemotaxis(
        cfg, run_id=run_id, resume_from=args.resume_from
    )
    print(f"[run] done. best_fit={best_fit:.4f}  ckpts -> {run_dir.relative_to(ROOT)}")
    print("[run] final best per-azimuth diagnostics:")
    rows = diagnose_params(best_params, cfg)
    _print_diag(rows)


if __name__ == "__main__":
    main()
