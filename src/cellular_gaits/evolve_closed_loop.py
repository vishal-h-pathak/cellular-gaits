"""CMA-ES evolution of the closed-loop NCA controller, evaluated in PARALLEL.

This is the C2-A evolver. Two changes from v1 (`evolve.py`):

1. **Closed loop.** The policy feeds live proprioception (joint angles + foot
   contacts) back into the grid each tick via ``NCA.build_sensor_map`` and
   ``rollout(..., pass_sensors=True)``. Warm-started from the v1 walking
   weights (``NCA.warm_start_from_v1``) so the sensor channels start at zero
   and the search begins exactly at the open-loop optimum.

2. **Parallel population evaluation.** v1 evaluated the 32 individuals
   sequentially (~35 min/run). Here a ``ProcessPoolExecutor`` farms each
   individual out to a worker that owns its own ``FlyEnv`` (MuJoCo handles are
   not fork-safe, so each worker builds the sim once in an initializer and
   reuses it). ``benchmark_speedup`` reports the measured wall-clock speedup.

Fitness (averaged over a few perturbation seeds for robustness):

    fitness = forward_dx
              - STABILITY_PENALTY_PER_STEP * n_below          (fall penalty)
              - HEADING_WEIGHT * mean_post_shove_yaw_error     (heading retention)

The base term (distance minus fall penalty) matches v1 so the numbers are
comparable; the heading term is what the closed loop is supposed to win on —
after the lateral shove, can it hold its original heading instead of being
knocked off course.

Checkpoint format and resume semantics match ``evolve.py`` exactly (the helper
functions are reused), so ``render.py`` / checkpoint tooling work unchanged.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cma
import numpy as np
import torch

from .env import (
    DEFAULT_N_STEPS,
    STABILITY_PENALTY_PER_STEP,
    FlyEnv,
    Perturbation,
)
from .evolve import (
    CKPT_DIR,
    CSV_HEADER,
    OUT_DIR,
    REPO_ROOT,
    _ensure_phase_column,
    _save_checkpoint,
    _utc_run_id,
    load_checkpoint,
)
from .nca import NCA, V1_N_PARAMS

V1_CKPT = REPO_ROOT / "checkpoints" / "2026-05-02T00-01-51Z" / "gen_50.npz"

HEADING_WEIGHT = 3.0  # radians of post-shove yaw drift cost this much distance
CA_INIT_SEED = 0  # fixed NCA.init_state seed across all rollouts
DEFAULT_EVAL_SEEDS = (101, 202, 303)


@dataclass
class ClosedLoopConfig:
    seed: int = 0
    popsize: int = 32
    n_gens: int = 50
    sigma_init: float = 0.3
    rollout_steps: int = DEFAULT_N_STEPS
    checkpoint_every: int = 5
    eval_seeds: tuple[int, ...] = DEFAULT_EVAL_SEEDS
    pert_magnitude: float = 3.0
    pert_window: tuple[float, float] = (0.4, 0.6)
    pert_duration_s: float = 0.05
    pert_randomize_sign: bool = True
    uneven_ground: bool = False
    terrain_seed: int = 0
    n_workers: int = 0  # 0 -> os.cpu_count()-1


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def composite_fitness(traj: dict) -> dict:
    """Compute the closed-loop fitness and its components from a rollout traj."""
    forward = float(traj["forward_dx"])
    n_below = int(traj["n_below"])
    yaw = np.asarray(traj["yaw"], dtype=np.float64)
    yaw0 = float(traj["yaw0"])
    onset = int(traj["impulse_onset_step"])
    if onset >= 0 and onset + 1 < yaw.size:
        post = yaw[onset + 1 :]
    else:
        post = yaw
    post_yaw_err = float(np.mean(np.abs(_wrap_pi(post - yaw0)))) if post.size else 0.0
    fall_pen = STABILITY_PENALTY_PER_STEP * n_below
    head_pen = HEADING_WEIGHT * post_yaw_err
    fitness = forward - fall_pen - head_pen
    return {
        "fitness": float(fitness),
        "forward_dx": forward,
        "n_below": n_below,
        "post_yaw_err": post_yaw_err,
        "fall_pen": float(fall_pen),
        "head_pen": float(head_pen),
    }


def _make_closed_policy(nca: NCA, ca_seed: int = CA_INIT_SEED):
    state = NCA.init_state(seed=ca_seed)

    def policy(t: int, sensors: dict) -> np.ndarray:
        nonlocal state
        smap = NCA.build_sensor_map(
            sensors["joint_angles_unit"], sensors["foot_contacts"]
        )
        with torch.no_grad():
            state = nca.step(state, sensors=smap)
        return nca.motor_targets(state)

    return policy


def _pert_kwargs(cfg: ClosedLoopConfig) -> dict:
    return {
        "magnitude": cfg.pert_magnitude,
        "window": cfg.pert_window,
        "duration_s": cfg.pert_duration_s,
        "randomize_sign": cfg.pert_randomize_sign,
    }


def evaluate_on_env(
    env: FlyEnv,
    nca: NCA,
    params: np.ndarray,
    eval_seeds,
    n_steps: int,
    pert_kwargs: dict,
    return_components: bool = False,
):
    """Evaluate one param vector on an existing env over the perturbation seeds."""
    nca.set_params(np.asarray(params, dtype=np.float64))
    comps = []
    for ps in eval_seeds:
        env.set_perturbation(Perturbation(seed=int(ps), **pert_kwargs))
        _, traj = env.rollout(
            _make_closed_policy(nca), n_steps=n_steps, pass_sensors=True
        )
        comps.append(composite_fitness(traj))
    mean_fit = float(np.mean([c["fitness"] for c in comps]))
    if return_components:
        return mean_fit, comps
    return mean_fit


# --------------------------------------------------------------------------- #
# Parallel worker plumbing (module-global env cache per worker process)
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _worker_init(uneven_ground: bool, terrain_seed: int) -> None:
    torch.set_num_threads(1)  # avoid oversubscription across worker processes
    _WORKER["env"] = FlyEnv(uneven_ground=uneven_ground, terrain_seed=terrain_seed)
    _WORKER["nca"] = NCA()


def _worker_eval(task) -> float:
    params, eval_seeds, n_steps, pert_kwargs = task
    return evaluate_on_env(
        _WORKER["env"], _WORKER["nca"], params, eval_seeds, n_steps, pert_kwargs
    )


def _resolve_workers(n_workers: int) -> int:
    if n_workers and n_workers > 0:
        return n_workers
    return max(1, (os.cpu_count() or 2) - 1)


def evaluate_population(
    xs,
    cfg: ClosedLoopConfig,
    executor: ProcessPoolExecutor | None,
) -> list[float]:
    """Evaluate a whole population. Parallel if executor given, else sequential."""
    pk = _pert_kwargs(cfg)
    tasks = [
        (np.asarray(x, dtype=np.float64), cfg.eval_seeds, cfg.rollout_steps, pk)
        for x in xs
    ]
    if executor is None:
        env = FlyEnv(uneven_ground=cfg.uneven_ground, terrain_seed=cfg.terrain_seed)
        nca = NCA()
        return [
            evaluate_on_env(env, nca, t[0], t[1], t[2], t[3]) for t in tasks
        ]
    return list(executor.map(_worker_eval, tasks))


# --------------------------------------------------------------------------- #
# Speedup benchmark
# --------------------------------------------------------------------------- #
def benchmark_speedup(cfg: ClosedLoopConfig, n_individuals: int | None = None) -> dict:
    """Time the same population sequentially vs in parallel; report speedup."""
    n = n_individuals or cfg.popsize
    nca0 = NCA()
    nca0.warm_start_from_v1(_load_v1_params()[:V1_N_PARAMS])
    x0 = nca0.flatten_params()
    rng = np.random.default_rng(cfg.seed)
    xs = [x0 + rng.normal(0, cfg.sigma_init, size=x0.size) for _ in range(n)]

    t0 = time.perf_counter()
    seq = evaluate_population(xs, cfg, executor=None)
    t_seq = time.perf_counter() - t0

    workers = _resolve_workers(cfg.n_workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg.uneven_ground, cfg.terrain_seed),
    ) as ex:
        # Warm the pool: pay the one-time worker/sim spin-up cost OUTSIDE the
        # timed region. Across a real 50-gen run the pool persists, so the
        # honest per-generation speedup is steady-state throughput, not the
        # first-batch cost.
        warm = evaluate_population(xs[: min(len(xs), workers)], cfg, executor=ex)
        t1 = time.perf_counter()
        par = evaluate_population(xs, cfg, executor=ex)
        t_par = time.perf_counter() - t1
    _ = warm

    max_abs_diff = float(np.max(np.abs(np.asarray(seq) - np.asarray(par))))
    speedup = t_seq / t_par if t_par > 0 else float("nan")
    result = {
        "n_individuals": n,
        "workers": workers,
        "t_sequential_s": t_seq,
        "t_parallel_s": t_par,
        "speedup": speedup,
        "max_abs_fitness_diff": max_abs_diff,
        "fitness_sample": [float(v) for v in seq[: min(5, n)]],
    }
    print(
        f"[benchmark] n={n} workers={workers}  seq={t_seq:.1f}s par={t_par:.1f}s  "
        f"speedup={speedup:.2f}x  max|Δfit|={max_abs_diff:.2e}"
    )
    return result


def _load_v1_params() -> np.ndarray:
    data = np.load(V1_CKPT, allow_pickle=False)
    return np.asarray(data["best_params"], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Evolution loop
# --------------------------------------------------------------------------- #
def run_evolution_closed_loop(
    cfg: ClosedLoopConfig,
    run_id: str | None = None,
    resume_from: Path | None = None,
    on_gen=None,
) -> tuple[float, np.ndarray, Path]:
    nca = NCA()
    n_params = nca.n_params
    workers = _resolve_workers(cfg.n_workers)

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
            print(f"[evolve-cl] EXACT resume from {resume_from.name} (gens_done={gens_done})")
        else:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            es = cma.CMAEvolutionStrategy(
                best_params_overall,
                cfg.sigma_init,
                {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9},
            )
            phase = "resumed"
            print(f"[evolve-cl] APPROXIMATE resume from {resume_from.name}")
    else:
        if run_id is None:
            run_id = _utc_run_id()
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        nca.warm_start_from_v1(_load_v1_params()[:V1_N_PARAMS])
        x0 = nca.flatten_params()
        es = cma.CMAEvolutionStrategy(
            x0,
            cfg.sigma_init,
            {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9},
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
        f"[evolve-cl] run_id={run_id}  n_params={n_params}  pop={cfg.popsize} "
        f"gens={cfg.n_gens} sigma={cfg.sigma_init}  workers={workers}  "
        f"eval_seeds={cfg.eval_seeds}  start_gen={start_gen} phase={phase}"
    )

    last_xs: list = []
    last_fits: list = []

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg.uneven_ground, cfg.terrain_seed),
    )
    try:
        for gen in range(start_gen, cfg.n_gens):
            t0 = time.perf_counter()
            xs = es.ask()
            fits = evaluate_population(xs, cfg, executor=executor)
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
                f"[evolve-cl] gen {gen + 1:3d}/{cfg.n_gens}  best={gen_best:.4f}  "
                f"mean={gen_mean:.4f}  std={gen_std:.4f}  ({gen_elapsed:.1f}s)"
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
                print(f"[evolve-cl]   checkpoint -> {ckpt.relative_to(REPO_ROOT)}")
    finally:
        executor.shutdown(wait=True)

    return best_fit_overall, best_params_overall, run_ckpt_dir
