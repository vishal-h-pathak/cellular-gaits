"""CMA-ES evolution loop for the NCA controller.

CMA minimizes; we maximize env fitness, so the evolver negates fitness
when calling ``tell``. The optimizer's "best" is the lowest negated
fitness, which is the highest true fitness.

Public surface:
    EvolveConfig   — dataclass of hyperparameters
    run_evolution  — executes the loop, writes per-gen CSV + .npz
                     checkpoints, returns (best_fit, best_params, ckpt_path)

Each individual is evaluated once (no fitness averaging across seeds —
v1 is sequential and deterministic).

Outputs:
    outputs/fitness_log_<run_id>.csv
        gen,best_fit,mean_fit,std_fit,time_s
    checkpoints/<run_id>/gen_NN.npz
        best_params           (n_params,)        float64  — best-so-far
        best_fit              ()                 float64
        pop_xs                (popsize,n_params) float64  — current generation
        pop_fits              (popsize,)         float64
        gen                   ()                 int64
        fitness_history       (gen,4)            float64  — [gen, best, mean, std]
        run_id                ()                 str
        config_json           ()                 str       — EvolveConfig as JSON
"""

from __future__ import annotations

import csv
import json
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


def run_evolution(
    cfg: EvolveConfig,
    run_id: str | None = None,
    on_gen: callable | None = None,
) -> tuple[float, np.ndarray, Path]:
    if run_id is None:
        run_id = _utc_run_id()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    nca = NCA()
    env = FlyEnv()
    n_params = nca.n_params

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir = CKPT_DIR / run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"fitness_log_{run_id}.csv"
    with log_path.open("w", newline="") as fp:
        csv.writer(fp).writerow(["gen", "best_fit", "mean_fit", "std_fit", "time_s"])

    best_fit_overall = -np.inf
    best_params_overall = x0.copy()
    fitness_history: list[list[float]] = []
    cfg_json = json.dumps(asdict(cfg))

    print(
        f"[evolve] run_id={run_id}  n_params={n_params}  "
        f"pop={cfg.popsize} gens={cfg.n_gens} sigma={cfg.sigma_init}"
    )

    for gen in range(cfg.n_gens):
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

        with log_path.open("a", newline="") as fp:
            csv.writer(fp).writerow(
                [gen, f"{gen_best:.6f}", f"{gen_mean:.6f}", f"{gen_std:.6f}", f"{gen_elapsed:.3f}"]
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
            ckpt = run_ckpt_dir / f"gen_{gen + 1:02d}.npz"
            np.savez(
                ckpt,
                best_params=best_params_overall,
                best_fit=np.float64(best_fit_overall),
                pop_xs=np.asarray(xs, dtype=np.float64),
                pop_fits=np.asarray(fits, dtype=np.float64),
                gen=np.int64(gen + 1),
                fitness_history=np.asarray(fitness_history, dtype=np.float64),
                run_id=run_id,
                config_json=cfg_json,
            )
            print(f"[evolve]   checkpoint -> {ckpt.relative_to(REPO_ROOT)}")

    return best_fit_overall, best_params_overall, run_ckpt_dir


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
