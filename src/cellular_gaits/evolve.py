"""CMA-ES evolution loop for the NCA controller.

CMA minimizes; we maximize env fitness, so the evolver negates fitness
when calling ``tell``. The optimizer's "best" is the lowest negated
fitness, which is the highest true fitness.

Public surface:
    EvolveConfig   — dataclass of hyperparameters
    run_evolution  — executes the loop, writes per-gen CSV + .npz/.pkl
                     checkpoints, returns (best_fit, best_params, ckpt_path)
    latest_checkpoint — locates the newest gen_NN.npz on disk
    load_checkpoint   — opens an npz / pkl checkpoint pair

Each individual is evaluated once (no fitness averaging across seeds —
v1 is sequential and deterministic).

Resuming:
    Pass ``resume_from=<path-to-gen_NN.npz>`` (or .pkl) to continue an
    interrupted run. When the .pkl sibling is present, the full
    ``cma.CMAEvolutionStrategy`` is restored and the trajectory matches
    an uninterrupted run exactly. Otherwise the resume is approximate:
    a fresh ES is warm-started at the saved ``best_params`` mean with
    ``cfg.sigma_init`` (caller's responsibility to pick something
    smaller than the original sigma — typically 0.1).

Outputs:
    outputs/fitness_log_<run_id>.csv
        gen,best_fit,mean_fit,std_fit,time_s,phase
    checkpoints/<run_id>/gen_NN.npz
        best_params           (n_params,)        float64  — best-so-far
        best_fit              ()                 float64
        pop_xs                (popsize,n_params) float64  — current generation
        pop_fits              (popsize,)         float64
        gen                   ()                 int64    — 1-indexed gens completed
        fitness_history       (gen,4)            float64  — [gen, best, mean, std]
        run_id                ()                 str
        config_json           ()                 str       — EvolveConfig as JSON
    checkpoints/<run_id>/gen_NN.pkl
        full pickled cma.CMAEvolutionStrategy (enables exact resume)
"""

from __future__ import annotations

import csv
import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cma
import numpy as np
import torch

from .env import DEFAULT_N_STEPS, FlyEnv
from .nca import NCA

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "outputs"

CSV_HEADER = ["gen", "best_fit", "mean_fit", "std_fit", "time_s", "phase"]


@dataclass
class EvolveConfig:
    seed: int = 0
    popsize: int = 32
    n_gens: int = 50
    sigma_init: float = 0.3
    rollout_steps: int = DEFAULT_N_STEPS
    checkpoint_every: int = 5


def _make_policy(nca: NCA, seed: int):
    state = NCA.init_state(seed=seed)

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)
        return nca.motor_targets(state)

    return policy


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _ensure_phase_column(log_path: Path) -> None:
    """Migrate an existing CSV without a `phase` column, marking rows `original`.

    No-op when the file is missing or already has the column.
    """
    if not log_path.exists():
        return
    with log_path.open("r", newline="") as fp:
        reader = csv.reader(fp)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    if header == CSV_HEADER:
        return
    if header[:5] != CSV_HEADER[:5]:
        raise ValueError(f"Unexpected CSV header in {log_path}: {header}")
    new_rows = [CSV_HEADER]
    for row in rows[1:]:
        if len(row) == 5:
            new_rows.append([*row, "original"])
        elif len(row) == 6:
            new_rows.append(row)
        else:
            raise ValueError(f"Unexpected row width in {log_path}: {row}")
    with log_path.open("w", newline="") as fp:
        csv.writer(fp).writerows(new_rows)


def latest_checkpoint(run_id: str | None = None) -> Path:
    if run_id is not None:
        run_dir = CKPT_DIR / run_id
    else:
        candidates = [p for p in CKPT_DIR.iterdir() if p.is_dir()] if CKPT_DIR.exists() else []
        if not candidates:
            raise FileNotFoundError(f"No checkpoint runs found under {CKPT_DIR}")
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    npzs = sorted(run_dir.glob("gen_*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No gen_*.npz under {run_dir}")
    return npzs[-1]


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint, returning a dict with npz fields + optional `es` (pickled ES)."""
    npz_path = path.with_suffix(".npz")
    pkl_path = path.with_suffix(".pkl")
    data = {}
    with np.load(npz_path, allow_pickle=True) as d:
        for k in d.keys():
            data[k] = d[k]
    if pkl_path.exists():
        with pkl_path.open("rb") as fp:
            data["es"] = pickle.load(fp)
    return data


def _save_checkpoint(
    run_ckpt_dir: Path,
    gens_completed: int,
    best_params: np.ndarray,
    best_fit: float,
    pop_xs: list,
    pop_fits: list,
    fitness_history: list,
    run_id: str,
    cfg_json: str,
    es: cma.CMAEvolutionStrategy,
) -> Path:
    ckpt = run_ckpt_dir / f"gen_{gens_completed:02d}.npz"
    np.savez(
        ckpt,
        best_params=best_params,
        best_fit=np.float64(best_fit),
        pop_xs=np.asarray(pop_xs, dtype=np.float64),
        pop_fits=np.asarray(pop_fits, dtype=np.float64),
        gen=np.int64(gens_completed),
        fitness_history=np.asarray(fitness_history, dtype=np.float64),
        run_id=run_id,
        config_json=cfg_json,
    )
    pkl = ckpt.with_suffix(".pkl")
    with pkl.open("wb") as fp:
        pickle.dump(es, fp)
    return ckpt


def run_evolution(
    cfg: EvolveConfig,
    run_id: str | None = None,
    on_gen: callable | None = None,
    resume_from: Path | None = None,
) -> tuple[float, np.ndarray, Path]:
    nca = NCA()
    env = FlyEnv()
    n_params = nca.n_params

    if resume_from is not None:
        ckpt_data = load_checkpoint(Path(resume_from))
        prior_run_id = str(ckpt_data["run_id"])
        if run_id is None:
            run_id = prior_run_id
        gens_done = int(ckpt_data["gen"])
        best_fit_overall = float(ckpt_data["best_fit"])
        best_params_overall = np.asarray(ckpt_data["best_params"], dtype=np.float64).copy()
        fitness_history = [list(map(float, row)) for row in ckpt_data["fitness_history"]]
        start_gen = gens_done

        if "es" in ckpt_data:
            es = ckpt_data["es"]
            phase = "resumed-exact"
            print(
                f"[evolve] EXACT resume from {resume_from.name} "
                f"(gens_done={gens_done}, best_fit={best_fit_overall:.4f})"
            )
        else:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            es = cma.CMAEvolutionStrategy(
                best_params_overall,
                cfg.sigma_init,
                {
                    "popsize": cfg.popsize,
                    "seed": cfg.seed + 1,
                    "verbose": -9,
                },
            )
            phase = "resumed"
            print(
                f"[evolve] APPROXIMATE resume from {resume_from.name} "
                f"(gens_done={gens_done}, best_fit={best_fit_overall:.4f}, "
                f"warm-start sigma={cfg.sigma_init})"
            )
    else:
        if run_id is None:
            run_id = _utc_run_id()
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        x0 = nca.flatten_params()
        es = cma.CMAEvolutionStrategy(
            x0,
            cfg.sigma_init,
            {
                "popsize": cfg.popsize,
                "seed": cfg.seed + 1,
                "verbose": -9,
            },
        )
        start_gen = 0
        best_fit_overall = -np.inf
        best_params_overall = x0.copy()
        fitness_history = []
        phase = "original"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir = CKPT_DIR / run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"fitness_log_{run_id}.csv"

    if resume_from is not None:
        _ensure_phase_column(log_path)
        if not log_path.exists():
            with log_path.open("w", newline="") as fp:
                csv.writer(fp).writerow(CSV_HEADER)
    else:
        with log_path.open("w", newline="") as fp:
            csv.writer(fp).writerow(CSV_HEADER)

    cfg_json = json.dumps(asdict(cfg))

    print(
        f"[evolve] run_id={run_id}  n_params={n_params}  "
        f"pop={cfg.popsize} gens={cfg.n_gens} sigma={cfg.sigma_init}  "
        f"start_gen={start_gen}  phase={phase}"
    )

    last_xs: list = []
    last_fits: list = []

    for gen in range(start_gen, cfg.n_gens):
        t0 = time.perf_counter()
        xs = es.ask()
        fits: list[float] = []
        for xi in xs:
            nca.set_params(np.asarray(xi, dtype=np.float64))
            fit, _ = env.rollout(_make_policy(nca, seed=cfg.seed), n_steps=cfg.rollout_steps)
            fits.append(float(fit))
        es.tell(xs, [-f for f in fits])

        gen_best = max(fits)
        gen_mean = float(np.mean(fits))
        gen_std = float(np.std(fits))
        gen_elapsed = time.perf_counter() - t0

        gen_best_idx = int(np.argmax(fits))
        if gen_best > best_fit_overall:
            best_fit_overall = gen_best
            best_params_overall = np.asarray(xs[gen_best_idx], dtype=np.float64).copy()

        fitness_history.append([float(gen), gen_best, gen_mean, gen_std])
        last_xs, last_fits = xs, fits

        with log_path.open("a", newline="") as fp:
            csv.writer(fp).writerow(
                [
                    gen,
                    f"{gen_best:.6f}",
                    f"{gen_mean:.6f}",
                    f"{gen_std:.6f}",
                    f"{gen_elapsed:.3f}",
                    phase,
                ]
            )
        print(
            f"[evolve] gen {gen + 1:3d}/{cfg.n_gens}  "
            f"best={gen_best:.4f}  mean={gen_mean:.4f}  std={gen_std:.4f}  "
            f"({gen_elapsed:.1f}s)"
        )
        if on_gen is not None:
            on_gen(gen, gen_best, gen_mean, gen_std)

        is_last = gen == cfg.n_gens - 1
        if (gen + 1) % cfg.checkpoint_every == 0 or is_last:
            ckpt = _save_checkpoint(
                run_ckpt_dir=run_ckpt_dir,
                gens_completed=gen + 1,
                best_params=best_params_overall,
                best_fit=best_fit_overall,
                pop_xs=last_xs,
                pop_fits=last_fits,
                fitness_history=fitness_history,
                run_id=run_id,
                cfg_json=cfg_json,
                es=es,
            )
            print(f"[evolve]   checkpoint -> {ckpt.relative_to(REPO_ROOT)}")

    return best_fit_overall, best_params_overall, run_ckpt_dir
