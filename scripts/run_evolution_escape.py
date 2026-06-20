"""CLI for the parallel escape (looming) CMA-ES evolution (X-A).

Examples:
    # Calibration: gen-0 (warm-start) diagnostics over a (speed, lead) grid —
    # find the threat magnitude where an UNTRAINED walker gets hit but the
    # bilateral loom cue is strong and correctly signed (escape is learnable).
    uv run python scripts/run_evolution_escape.py --calibrate

    # Speedup benchmark only
    uv run python scripts/run_evolution_escape.py --benchmark-only --pop 16

    # Short validation run (build everything, validate first), with benchmark
    uv run python scripts/run_evolution_escape.py --pop 16 --gens 8 --benchmark

    # Full run
    uv run python scripts/run_evolution_escape.py --pop 32 --gens 50

Outputs:
    outputs/fitness_log_<run_id>.csv
    outputs/benchmark_<run_id>.json        (when --benchmark[-only])
    scratch/xa/calibration.json            (when --calibrate)
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

from cellular_gaits.evolve import _utc_run_id, load_checkpoint  # noqa: E402
from cellular_gaits.evolve_escape import (  # noqa: E402
    ESCAPE_ROLLOUT_STEPS,
    EscapeConfig,
    benchmark_speedup,
    diagnose_params,
    run_evolution_escape,
    warm_start_x0,
)

OUT_DIR = ROOT / "outputs"
SCRATCH = ROOT / "scratch" / "xa"


def _cfg_from_args(args) -> EscapeConfig:
    kw = dict(
        seed=args.seed,
        popsize=args.pop,
        n_gens=args.gens,
        sigma_init=args.sigma,
        rollout_steps=args.rollout_steps,
        checkpoint_every=args.checkpoint_every,
        threat_speed=args.speed,
        threat_radius=args.radius,
        threat_start_distance=args.start_distance,
        threat_hit_radius=args.hit_radius,
        threat_lead_distance=args.lead,
        threat_seed=args.threat_seed,
        loom_input_gain=args.loom_gain,
        n_workers=args.workers,
    )
    if getattr(args, "azimuths", None):
        kw["azimuths_deg"] = tuple(args.azimuths)
    return EscapeConfig(**kw)


def _print_diag(rows: list[dict]) -> None:
    print(
        f"    {'az':>4} {'hit':>5} {'dmin':>6} {'fwd':>6} {'awayΔ':>7} "
        f"{'early':>7} {'loomPk':>6} {'L-R':>7} {'fit':>7}"
    )
    for r in rows:
        print(
            f"    {r['azimuth_deg']:>4.0f} {str(r['hit']):>5} {r['min_dist']:>6.2f} "
            f"{r['forward_dx']:>6.1f} {r['total_away_turn']:>+7.3f} "
            f"{r['early_away_turn']:>+7.3f} {r['loom_peak']:>6.3f} "
            f"{r['loom_mean_LmR']:>+7.3f} {r['fitness']:>7.2f}"
        )


def _cue_correct(rows: list[dict]) -> bool:
    """Loom cue correctly signed: left threat (az 90) biases L, right (270) R."""
    by_az = {r["azimuth_deg"]: r for r in rows}
    ok = True
    if 90.0 in by_az:
        ok = ok and by_az[90.0]["loom_mean_LmR"] > 0.02
    if 270.0 in by_az:
        ok = ok and by_az[270.0]["loom_mean_LmR"] < -0.02
    return ok


def calibrate(args) -> None:
    """Gen-0 (warm-start) diagnostics over a (speed, lead_distance) grid.

    The warm-start controller walks straight blind (loom weights zero), so this
    tells us, BEFORE evolving: from which azimuths the straight walker gets HIT
    (so there is something to learn), and whether the bilateral loom cue is
    strong and correctly signed along the approach (so the turn is learnable).
    We want a cell with a high untrained hit-rate (ideally all azimuths) AND a
    correctly-signed, sizeable L-vs-R cue.
    """
    x0 = warm_start_x0(EscapeConfig())
    grid = []
    for speed in args.cal_speeds:
        for lead in args.cal_leads:
            cfg = _cfg_from_args(args)
            cfg = EscapeConfig(
                **{**cfg.__dict__, "threat_speed": speed, "threat_lead_distance": lead}
            )
            rows = diagnose_params(x0, cfg)
            hit_frac = float(np.mean([r["hit"] for r in rows]))
            summary = {
                "speed": speed,
                "lead_distance": lead,
                "hit_radius": cfg.threat_hit_radius,
                "loom_input_gain": cfg.loom_input_gain,
                "untrained_hit_frac": hit_frac,
                "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
                "mean_loom_peak": float(np.mean([r["loom_peak"] for r in rows])),
                "cue_correctly_signed": _cue_correct(rows),
                "rows": rows,
            }
            grid.append(summary)
            print(
                f"[cal] speed={speed:>5.1f} lead={lead:>5.1f}  "
                f"untrained_hit={hit_frac:.2f}  mean_dmin={summary['mean_min_dist']:.2f}  "
                f"loomPk={summary['mean_loom_peak']:.3f}  "
                f"cue_ok={summary['cue_correctly_signed']}"
            )
            _print_diag(rows)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / "calibration.json"
    out.write_text(json.dumps({"grid": grid}, indent=2))
    print(f"[cal] wrote {out.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", "--popsize", dest="pop", type=int, default=32)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--rollout-steps", type=int, default=ESCAPE_ROLLOUT_STEPS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=0, help="0 -> cpu_count()-1")
    # threat geometry
    p.add_argument("--speed", type=float, default=45.0)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--start-distance", type=float, default=18.0)
    p.add_argument("--hit-radius", type=float, default=3.0)
    p.add_argument("--lead", type=float, default=15.0)
    p.add_argument("--threat-seed", type=int, default=7)
    p.add_argument("--loom-gain", type=float, default=8.0)
    p.add_argument(
        "--azimuths", type=float, nargs="+", default=None,
        help="Threat azimuths in deg, onset-heading frame (default: 0 90 270).",
    )
    # calibration grid
    p.add_argument("--cal-speeds", type=float, nargs="+", default=[25.0, 35.0, 45.0])
    p.add_argument("--cal-leads", type=float, nargs="+", default=[10.0, 15.0, 20.0])
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

    # Gen-0 (warm-start) diagnostics for the record before evolving.
    print("[run] gen-0 (warm-start) per-azimuth diagnostics:")
    _print_diag(diagnose_params(warm_start_x0(cfg), cfg))

    best_fit, best_params, run_dir = run_evolution_escape(
        cfg, run_id=run_id, resume_from=args.resume_from
    )
    print(f"[run] done. best_fit={best_fit:.4f}  ckpts -> {run_dir.relative_to(ROOT)}")
    print("[run] final best per-azimuth diagnostics:")
    _print_diag(diagnose_params(best_params, cfg))


if __name__ == "__main__":
    main()
