"""CMA-ES evolution of the chemotaxis NCA controller, evaluated in PARALLEL.

This is the CH-A evolver. It builds on the C2-A closed-loop evolver
(``evolve_closed_loop.py``): same parallel ProcessPool machinery and checkpoint
format, two new ideas on top.

1. **Bilateral odor sensing.** The policy is the 8-input chemo NCA
   (``NCA(chemo=True)``): 4 state + 2 proprio + 2 chemo input channels. The two
   chemo channels carry the left/right antenna concentrations ``cL``, ``cR``
   read each control step from an :class:`~cellular_gaits.env.OdorField`. Warm-
   started from the *closed-loop* walking weights
   (``NCA.warm_start_from_closed_loop``) with the chemo channels at zero, so the
   search starts from a controller that already walks and is robust, and a
   chemo-zeroed rollout reproduces the closed-loop dynamics exactly.

2. **Approach fitness + azimuth generalization.** Reward is approach to the
   source, averaged over several source **azimuths** (ahead / left / behind /
   right relative to the fly's fixed +x spawn heading). Because the source can
   be on either side, a fixed turn bias cannot win on all conditions — the
   controller must turn based on ``cL`` vs ``cR``. That is what makes the
   steering *emergent* rather than hard-coded.

        F_condition = (d_start - d_end)                       (approach)
                      + reach_bonus * [d_end < reach_radius]  (got there)
                      - time_penalty * (steps_to_reach / N)   (got there fast)
                      - STABILITY_PENALTY_PER_STEP * n_below  (stayed upright)

        fitness = mean over azimuths of F_condition

The fall penalty is the same term the walking objective uses, so the controller
keeps walking while it learns to steer. ``steps_to_reach`` is the first control
step the thorax enters ``reach_radius`` (or N if it never does), computed
post-hoc from the logged path — no early termination, so rollouts stay
deterministic and the env stays generic.
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
    ANTENNA_FORWARD_OFFSET,
    ANTENNA_LATERAL_OFFSET,
    DEFAULT_N_STEPS,
    DEFAULT_ODOR_LAMBDA,
    STABILITY_PENALTY_PER_STEP,
    FlyEnv,
    OdorField,
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
from .nca import NCA

CL_CONTROLLER_JSON = REPO_ROOT / "outputs" / "web_data_c2" / "closed_loop_controller.json"

CA_INIT_SEED = 0  # fixed NCA.init_state seed across all rollouts

# Source azimuths (degrees, CCW from +x = the fly's spawn heading): ahead,
# left, behind, right. Symmetric left/right so a fixed turn bias cannot win.
DEFAULT_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)


@dataclass
class ChemoConfig:
    seed: int = 0
    popsize: int = 24
    n_gens: int = 50
    sigma_init: float = 0.2
    rollout_steps: int = DEFAULT_N_STEPS
    checkpoint_every: int = 5
    # --- task geometry ---
    source_distance: float = 18.0
    odor_lambda: float = DEFAULT_ODOR_LAMBDA
    reach_radius: float = 3.0
    azimuths_deg: tuple[float, ...] = DEFAULT_AZIMUTHS_DEG
    antenna_forward: float = ANTENNA_FORWARD_OFFSET
    antenna_lateral: float = ANTENNA_LATERAL_OFFSET
    # --- fitness shaping ---
    # "closest": approach = d_start - min_dist (closest approach over the
    #   rollout), reach = min_dist < r. Correct for a walker that overshoots the
    #   source: it scores how close it steered, not where it happened to stop.
    # "d_end": approach = d_start - d_end, reach = d_end < r (the literal CH-A
    #   spec; assumes the controller can arrive and stay).
    fitness_mode: str = "closest"
    reach_bonus: float = 8.0
    time_penalty: float = 4.0
    n_workers: int = 0  # 0 -> os.cpu_count()-1


# --------------------------------------------------------------------------- #
# Task conditions (source placements)
# --------------------------------------------------------------------------- #
def source_for_azimuth(distance: float, azimuth_deg: float) -> tuple[float, float]:
    """Place the source ``distance`` away at ``azimuth_deg`` from the +x heading."""
    a = np.radians(azimuth_deg)
    return (float(distance * np.cos(a)), float(distance * np.sin(a)))


def conditions(cfg: ChemoConfig) -> list[tuple[float, OdorField]]:
    """One (azimuth_deg, OdorField) per evaluation condition."""
    return [
        (
            az,
            OdorField(
                source_xy=source_for_azimuth(cfg.source_distance, az),
                lam=cfg.odor_lambda,
            ),
        )
        for az in cfg.azimuths_deg
    ]


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def chemo_fitness(
    traj: dict,
    reach_radius: float,
    reach_bonus: float,
    time_penalty: float,
    mode: str = "closest",
) -> dict:
    """Approach fitness + components from a chemo rollout traj.

    ``mode="closest"`` scores the closest approach over the rollout (robust to a
    walker that overshoots); ``mode="d_end"`` scores the final distance (the
    literal CH-A spec). The "reach" event and the time-to-reach use the closest
    approach in both modes (the first step the thorax enters ``reach_radius``).
    """
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    src = np.asarray(traj["odor_source_xy"], dtype=np.float64)
    dist = np.linalg.norm(path - src[None, :], axis=1)
    n = dist.size
    d_start = float(dist[0])
    d_end = float(dist[-1])
    min_dist = float(dist.min())
    n_below = int(traj["n_below"])

    reached_idx = np.where(dist < reach_radius)[0]
    reached = bool(reached_idx.size > 0)
    steps_to_reach = int(reached_idx[0]) if reached else (n - 1)
    reach_frac = steps_to_reach / max(1, n - 1)

    if mode == "d_end":
        approach = d_start - d_end
        reach_for_bonus = bool(d_end < reach_radius)
    else:  # "closest"
        approach = d_start - min_dist
        reach_for_bonus = reached
    bonus = reach_bonus if reach_for_bonus else 0.0
    time_pen = time_penalty * reach_frac
    fall_pen = STABILITY_PENALTY_PER_STEP * n_below
    fitness = approach + bonus - time_pen - fall_pen
    return {
        "fitness": float(fitness),
        "approach": float(approach),
        "d_start": d_start,
        "d_end": d_end,
        "min_dist": min_dist,
        "reached": reached,
        "steps_to_reach": steps_to_reach,
        "reach_frac": float(reach_frac),
        "bonus": float(bonus),
        "time_pen": float(time_pen),
        "fall_pen": float(fall_pen),
        "n_below": n_below,
    }


def _make_chemo_policy(nca: NCA, ca_seed: int = CA_INIT_SEED):
    state = NCA.init_state(seed=ca_seed)

    def policy(t: int, sensors: dict) -> np.ndarray:
        nonlocal state
        smap = NCA.build_chemo_sensor_map(
            sensors["joint_angles_unit"],
            sensors["foot_contacts"],
            sensors.get("c_left", 0.0),
            sensors.get("c_right", 0.0),
        )
        with torch.no_grad():
            state = nca.step(state, sensors=smap)
        return nca.motor_targets(state)

    return policy


def evaluate_on_env(
    env: FlyEnv,
    nca: NCA,
    params: np.ndarray,
    cfg: ChemoConfig,
    return_components: bool = False,
):
    """Evaluate one param vector over all source-azimuth conditions on one env."""
    nca.set_params(np.asarray(params, dtype=np.float64))
    comps = []
    for _az, odor in conditions(cfg):
        env.set_odor(odor)
        _, traj = env.rollout(
            _make_chemo_policy(nca), n_steps=cfg.rollout_steps, pass_sensors=True
        )
        comps.append(
            chemo_fitness(
                traj, cfg.reach_radius, cfg.reach_bonus, cfg.time_penalty, cfg.fitness_mode
            )
        )
    mean_fit = float(np.mean([c["fitness"] for c in comps]))
    if return_components:
        return mean_fit, comps
    return mean_fit


def diagnose_params(
    params: np.ndarray, cfg: ChemoConfig, env: FlyEnv | None = None
) -> list[dict]:
    """Per-azimuth diagnostics for one controller: approach, reach, turning,
    and the bilateral cue strength. Used by calibration / the validation gate.
    """
    own_env = env is None
    if own_env:
        env = FlyEnv(
            antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral
        )
    nca = NCA(chemo=True)
    nca.set_params(np.asarray(params, dtype=np.float64))
    rows = []
    for az, odor in conditions(cfg):
        env.set_odor(odor)
        _, traj = env.rollout(
            _make_chemo_policy(nca), n_steps=cfg.rollout_steps, pass_sensors=True
        )
        fc = chemo_fitness(
            traj, cfg.reach_radius, cfg.reach_bonus, cfg.time_penalty, cfg.fitness_mode
        )
        path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
        src = np.asarray(traj["odor_source_xy"], dtype=np.float64)
        dist = np.linalg.norm(path - src[None, :], axis=1)
        chemo = np.asarray(traj["chemo"], dtype=np.float64)
        sig = np.abs(chemo[:, 0] - chemo[:, 1]) if chemo.size else np.zeros(1)
        yaw = np.asarray(traj["yaw"], dtype=np.float64)
        rows.append(
            {
                "azimuth_deg": float(az),
                "source_xy": [float(v) for v in src],
                "fitness": fc["fitness"],
                "d_start": fc["d_start"],
                "d_end": fc["d_end"],
                "min_dist": float(dist.min()),
                "approach": fc["approach"],
                "reached": fc["reached"],
                "steps_to_reach": fc["steps_to_reach"],
                "n_below": fc["n_below"],
                "dyaw_final": float(yaw[-1] - traj["yaw0"]),
                "signal_start": float(sig[0]) if sig.size else 0.0,
                "signal_max": float(sig.max()) if sig.size else 0.0,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Parallel worker plumbing (module-global env cache per worker process)
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _worker_init(antenna_forward: float, antenna_lateral: float) -> None:
    torch.set_num_threads(1)  # avoid oversubscription across worker processes
    _WORKER["env"] = FlyEnv(
        antenna_forward=antenna_forward, antenna_lateral=antenna_lateral
    )
    _WORKER["nca"] = NCA(chemo=True)


def _worker_eval(task) -> float:
    params, cfg = task
    return evaluate_on_env(_WORKER["env"], _WORKER["nca"], params, cfg)


def _resolve_workers(n_workers: int) -> int:
    if n_workers and n_workers > 0:
        return n_workers
    return max(1, (os.cpu_count() or 2) - 1)


def evaluate_population(
    xs,
    cfg: ChemoConfig,
    executor: ProcessPoolExecutor | None,
) -> list[float]:
    """Evaluate a whole population. Parallel if executor given, else sequential."""
    tasks = [(np.asarray(x, dtype=np.float64), cfg) for x in xs]
    if executor is None:
        env = FlyEnv(
            antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral
        )
        nca = NCA(chemo=True)
        return [evaluate_on_env(env, nca, t[0], t[1]) for t in tasks]
    return list(executor.map(_worker_eval, tasks))


# --------------------------------------------------------------------------- #
# Warm start
# --------------------------------------------------------------------------- #
def load_cl_params() -> np.ndarray:
    data = json.loads(CL_CONTROLLER_JSON.read_text())
    return np.asarray(data["flat_params"], dtype=np.float64)


def warm_start_x0(cfg: ChemoConfig) -> np.ndarray:
    """Chemo NCA flat params warm-started from the closed-loop controller."""
    nca = NCA(chemo=True)
    nca.warm_start_from_closed_loop(load_cl_params())
    return nca.flatten_params()


# --------------------------------------------------------------------------- #
# Speedup benchmark
# --------------------------------------------------------------------------- #
def benchmark_speedup(cfg: ChemoConfig, n_individuals: int | None = None) -> dict:
    """Time the same population sequentially vs in parallel; report speedup."""
    n = n_individuals or cfg.popsize
    x0 = warm_start_x0(cfg)
    rng = np.random.default_rng(cfg.seed)
    xs = [x0 + rng.normal(0, cfg.sigma_init, size=x0.size) for _ in range(n)]

    t0 = time.perf_counter()
    seq = evaluate_population(xs, cfg, executor=None)
    t_seq = time.perf_counter() - t0

    workers = _resolve_workers(cfg.n_workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg.antenna_forward, cfg.antenna_lateral),
    ) as ex:
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


# --------------------------------------------------------------------------- #
# Evolution loop
# --------------------------------------------------------------------------- #
def run_evolution_chemotaxis(
    cfg: ChemoConfig,
    run_id: str | None = None,
    resume_from: Path | None = None,
    on_gen=None,
) -> tuple[float, np.ndarray, Path]:
    nca = NCA(chemo=True)
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
            print(f"[evolve-ch] EXACT resume from {Path(resume_from).name} (gens_done={gens_done})")
        else:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            es = cma.CMAEvolutionStrategy(
                best_params_overall,
                cfg.sigma_init,
                {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9},
            )
            phase = "resumed"
            print(f"[evolve-ch] APPROXIMATE resume from {Path(resume_from).name}")
    else:
        if run_id is None:
            run_id = _utc_run_id()
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        x0 = warm_start_x0(cfg)
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
        f"[evolve-ch] run_id={run_id}  n_params={n_params}  pop={cfg.popsize} "
        f"gens={cfg.n_gens} sigma={cfg.sigma_init}  workers={workers}  "
        f"azimuths={cfg.azimuths_deg}  dist={cfg.source_distance} lam={cfg.odor_lambda} "
        f"reach_r={cfg.reach_radius}  start_gen={start_gen} phase={phase}"
    )

    last_xs: list = []
    last_fits: list = []

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg.antenna_forward, cfg.antenna_lateral),
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
                f"[evolve-ch] gen {gen + 1:3d}/{cfg.n_gens}  best={gen_best:.4f}  "
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
                print(f"[evolve-ch]   checkpoint -> {ckpt.relative_to(REPO_ROOT)}")
    finally:
        executor.shutdown(wait=True)

    return best_fit_overall, best_params_overall, run_ckpt_dir
