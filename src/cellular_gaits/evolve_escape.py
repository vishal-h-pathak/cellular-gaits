"""CMA-ES evolution of the escape (looming) NCA controller, in PARALLEL.

This is the X-A evolver. It is the structural twin of the CH-A chemotaxis
evolver (``evolve_chemotaxis.py``): same parallel ProcessPool machinery and
checkpoint format, the two bilateral cue channels now carrying *looming* instead
of *odor*.

1. **Bilateral looming.** The policy is the 8-input loom NCA (``NCA(loom=True)``):
   4 state + 2 proprio + 2 loom input channels. The two loom channels carry the
   left/right looming signal ``loom_L``, ``loom_R`` read each control step from a
   :class:`~cellular_gaits.env.Threat`. Warm-started from the *closed-loop*
   walking weights (``NCA.warm_start_from_closed_loop``) with the loom channels
   at zero, so the search starts from a controller that already walks, and a
   loom-zeroed rollout reproduces the closed-loop dynamics exactly (A/B).

2. **Escape fitness + azimuth generalization.** A looming threat appears at a
   seeded time mid-episode and flies a collision course from one of several
   **azimuths** (front / left / right relative to the fly's onset heading).
   Reward is *escape*:

        F_condition = w_clear   * min(closest_threat_distance, clear_cap)  (flee away)
                    + w_survive * [not hit]                                 (survival)
                    + w_react   * clip(early away-turn / react_ref, 0, 1)   (react fast)
                    + w_head    * clip(total away-turn, -head_clip, head_clip)  (turn away)
                    - STABILITY_PENALTY_PER_STEP * n_below                  (stay upright)

        fitness = mean over azimuths of F_condition

   Because the threat can come from the left OR the right, a fixed turn bias
   cannot win on all conditions — to escape every azimuth the controller must
   turn based on ``loom_L`` vs ``loom_R``. That is what makes the escape
   *directed/emergent* rather than a hard-coded turn. The "away" direction for a
   condition is set by the threat's onset bearing (left threat -> turn right,
   right threat -> turn left); the frontal threat has no preferred side, so its
   directional terms are zero and survival/clearance carry it.

HONESTY: the looming front-end (``Threat`` + ``FlyEnv.read_loom``) is HAND-BUILT.
It stands in for the real LC4/LPLC2 -> DNp01 (Giant Fiber) circuit — the
connectome endgame, not built here. Episodes are short (~1.2 s) and the azimuth
set is a small symmetric panel; the generalization scope is exactly that panel
plus whatever held-out azimuths the metrics sweep adds.
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
    CONTROL_DT_S,
    LOOM_EXP_GAIN,
    LOOM_EXP_REF,
    LOOM_SIZE_GAIN,
    STABILITY_PENALTY_PER_STEP,
    THREAT_HIT_RADIUS,
    THREAT_LEAD_DISTANCE,
    THREAT_RADIUS,
    THREAT_SPEED,
    THREAT_START_DISTANCE,
    FlyEnv,
    Threat,
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

# Short escape episodes (~1.2 s): walk, threat looms mid-episode, respond.
ESCAPE_ROLLOUT_S = 1.2
ESCAPE_ROLLOUT_STEPS = int(round(ESCAPE_ROLLOUT_S / CONTROL_DT_S))  # 300

# Threat azimuths (degrees in the fly's onset-heading frame): front, left, right.
# Symmetric left/right so a fixed turn bias cannot win all conditions. The 180°
# "behind" case is intentionally omitted (see REPORT_x_a.md): from a forward
# walk it needs a full U-turn inside the short episode and is not the headline.
DEFAULT_AZIMUTHS_DEG = (0.0, 90.0, 270.0)


@dataclass
class EscapeConfig:
    seed: int = 0
    popsize: int = 32
    n_gens: int = 50
    sigma_init: float = 0.2
    rollout_steps: int = ESCAPE_ROLLOUT_STEPS
    checkpoint_every: int = 5
    # --- threat stimulus ---
    # Defaults are the calibrated operating point (validation sweep): speed 45 /
    # lead 15 makes the threat a genuine collision course from all three azimuths
    # so an untrained straight walker is hit 3/3 (something to learn) while the
    # bilateral loom cue stays strong and correctly signed (escape is learnable).
    azimuths_deg: tuple[float, ...] = DEFAULT_AZIMUTHS_DEG
    threat_speed: float = 45.0
    threat_radius: float = THREAT_RADIUS
    threat_start_distance: float = THREAT_START_DISTANCE
    threat_hit_radius: float = THREAT_HIT_RADIUS
    threat_lead_distance: float = 15.0
    threat_seed: int = 7  # fixes onset; identical threats across controllers
    threat_window: tuple[float, float] = (0.2, 0.4)
    # --- loom sensor normalization ---
    loom_size_gain: float = LOOM_SIZE_GAIN
    loom_exp_gain: float = LOOM_EXP_GAIN
    loom_exp_ref: float = LOOM_EXP_REF
    # The bilateral loom signal is read out in [0, 1] (interpretable, logged as
    # such) but multiplied by ``loom_input_gain`` before it enters conv1. The
    # warm-start gait is bang-bang (every motor cell pinned at the +/-1 clamp),
    # so an unamplified [0,1] loom cannot move a motor cell until the loom
    # weights grow to ~2 — a flat fitness plateau CMA-ES cannot climb from zero.
    # Amplifying the cue (the escape analog of CH-A's deliberately strong antenna
    # baseline) gives small loom weights immediate leverage, so steering is
    # learnable in the generation budget. A/B is preserved exactly: amplifying a
    # zero loom, or amplifying through zero loom weights, is still zero.
    loom_input_gain: float = 8.0
    # --- fitness shaping ---
    # Calibrated (validation) so DIRECTED escape wins over the degenerate fixed-
    # turn: with target leading, any large maneuver breaks the intercept, so
    # survival alone does NOT force directedness. Weighting the directional terms
    # (turn AWAY, react fast) well above survival makes opposite left/right turns
    # the fitness optimum — a fixed turn forfeits the away-bonus on one side and
    # loses. Opposite turns then emerge in ~13 generations.
    w_clear: float = 0.15  # weight on graded clearance (closest approach distance)
    clear_cap: float = 18.0  # clearance saturates here (~ start_distance)
    w_survive: float = 3.0  # discrete survival bonus (not hit)
    w_react: float = 4.0  # weight on early away-turn (reaction speed)
    react_ref: float = 0.3  # away-turn (rad) within react_window that scores 1
    react_window: int = 40  # steps after onset counted as the "fast" window
    w_head: float = 5.0  # weight on total away-turn (turned away from threat)
    head_clip: float = 1.5708  # cap the away-turn reward at ~pi/2 (no spin reward)
    n_workers: int = 0  # 0 -> os.cpu_count()-1


# --------------------------------------------------------------------------- #
# Task conditions (threat placements)
# --------------------------------------------------------------------------- #
def threat_for_azimuth(cfg: EscapeConfig, azimuth_deg: float) -> Threat:
    return Threat(
        azimuth_deg=float(azimuth_deg),
        speed=cfg.threat_speed,
        radius=cfg.threat_radius,
        start_distance=cfg.threat_start_distance,
        hit_radius=cfg.threat_hit_radius,
        lead_distance=cfg.threat_lead_distance,
        window=cfg.threat_window,
        seed=cfg.threat_seed,
    )


def conditions(cfg: EscapeConfig) -> list[tuple[float, Threat]]:
    """One (azimuth_deg, Threat) per evaluation condition."""
    return [(az, threat_for_azimuth(cfg, az)) for az in cfg.azimuths_deg]


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _away_sign(azimuth_deg: float) -> float:
    """+1 if the fly should turn left to flee, -1 if right, 0 if frontal.

    A threat on the fly's left (sin(az) > 0) is escaped by turning right
    (yaw decreasing), so the "away" yaw change is negative there: away-turn =
    -sign(sin az) * dyaw. We return that -sign(sin az) factor.
    """
    s = np.sin(np.radians(azimuth_deg))
    if abs(s) < 1e-6:
        return 0.0
    return -float(np.sign(s))


def escape_fitness(traj: dict, cfg: EscapeConfig) -> dict:
    """Escape fitness + components from a loom rollout traj."""
    meta = traj["threat"]
    onset = int(meta["onset_step"]) if meta is not None else -1
    yaw = np.asarray(traj["yaw"], dtype=np.float64)
    n_below = int(traj["n_below"])
    min_dist = float(traj["threat_min_dist"])
    hit = bool(traj["threat_hit"])
    az = float(meta["azimuth_deg"]) if meta is not None else 0.0

    away = _away_sign(az)
    # yaw[onset] is the heading at threat onset; later indices are post-onset.
    onset_idx = min(max(onset, 0), yaw.size - 1)
    yaw_onset = yaw[onset_idx]
    react_idx = min(onset_idx + cfg.react_window, yaw.size - 1)
    total_away = away * float(yaw[-1] - yaw_onset)
    early_away = away * float(yaw[react_idx] - yaw_onset)

    clear = cfg.w_clear * min(min_dist, cfg.clear_cap)
    survive = cfg.w_survive * (0.0 if hit else 1.0)
    react = cfg.w_react * float(np.clip(early_away / cfg.react_ref, 0.0, 1.0))
    head = cfg.w_head * float(np.clip(total_away, -cfg.head_clip, cfg.head_clip))
    fall_pen = STABILITY_PENALTY_PER_STEP * n_below
    fitness = clear + survive + react + head - fall_pen
    return {
        "fitness": float(fitness),
        "min_dist": min_dist,
        "hit": hit,
        "clear": float(clear),
        "survive": float(survive),
        "react": float(react),
        "head": float(head),
        "total_away_turn": float(total_away),
        "early_away_turn": float(early_away),
        "fall_pen": float(fall_pen),
        "n_below": n_below,
        "onset_step": onset,
    }


def _make_escape_policy(
    nca: NCA, ca_seed: int = CA_INIT_SEED, loom_input_gain: float = 8.0
):
    state = NCA.init_state(seed=ca_seed)

    def policy(t: int, sensors: dict) -> np.ndarray:
        nonlocal state
        smap = NCA.build_loom_sensor_map(
            sensors["joint_angles_unit"],
            sensors["foot_contacts"],
            loom_input_gain * sensors.get("loom_left", 0.0),
            loom_input_gain * sensors.get("loom_right", 0.0),
        )
        with torch.no_grad():
            state = nca.step(state, sensors=smap)
        return nca.motor_targets(state)

    return policy


def _env_for_cfg(cfg: EscapeConfig) -> FlyEnv:
    return FlyEnv(
        loom_size_gain=cfg.loom_size_gain,
        loom_exp_gain=cfg.loom_exp_gain,
        loom_exp_ref=cfg.loom_exp_ref,
    )


def evaluate_on_env(
    env: FlyEnv,
    nca: NCA,
    params: np.ndarray,
    cfg: EscapeConfig,
    return_components: bool = False,
):
    """Evaluate one param vector over all threat-azimuth conditions on one env."""
    nca.set_params(np.asarray(params, dtype=np.float64))
    comps = []
    for _az, threat in conditions(cfg):
        env.set_threat(threat)
        _, traj = env.rollout(
            _make_escape_policy(nca, loom_input_gain=cfg.loom_input_gain),
            n_steps=cfg.rollout_steps,
            pass_sensors=True,
        )
        comps.append(escape_fitness(traj, cfg))
    mean_fit = float(np.mean([c["fitness"] for c in comps]))
    if return_components:
        return mean_fit, comps
    return mean_fit


def diagnose_params(
    params: np.ndarray, cfg: EscapeConfig, env: FlyEnv | None = None
) -> list[dict]:
    """Per-azimuth diagnostics for one controller: clearance, hit, the away-turn
    it produced, and the bilateral loom cue strength. Used by calibration / the
    validation gate.
    """
    own_env = env is None
    if own_env:
        env = _env_for_cfg(cfg)
    nca = NCA(loom=True)
    nca.set_params(np.asarray(params, dtype=np.float64))
    rows = []
    for az, threat in conditions(cfg):
        env.set_threat(threat)
        _, traj = env.rollout(
            _make_escape_policy(nca, loom_input_gain=cfg.loom_input_gain),
            n_steps=cfg.rollout_steps,
            pass_sensors=True,
        )
        fc = escape_fitness(traj, cfg)
        loom = np.asarray(traj["loom"], dtype=np.float64)
        mag = loom.sum(axis=1) if loom.size else np.zeros(1)
        meaningful = loom[mag > 0.1] if loom.size else np.zeros((0, 2))
        lr = (meaningful[:, 0] - meaningful[:, 1]) if meaningful.size else np.zeros(1)
        rows.append(
            {
                "azimuth_deg": float(az),
                "fitness": fc["fitness"],
                "min_dist": fc["min_dist"],
                "hit": fc["hit"],
                "survive": fc["survive"],
                "clear": fc["clear"],
                "react": fc["react"],
                "head": fc["head"],
                "total_away_turn": fc["total_away_turn"],
                "early_away_turn": fc["early_away_turn"],
                "n_below": fc["n_below"],
                "forward_dx": float(traj["forward_dx"]),
                "onset_step": fc["onset_step"],
                "loom_peak": float(mag.max()) if mag.size else 0.0,
                "loom_mean_LmR": float(lr.mean()) if lr.size else 0.0,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Parallel worker plumbing (module-global env cache per worker process)
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _worker_init(loom_size_gain: float, loom_exp_gain: float, loom_exp_ref: float) -> None:
    torch.set_num_threads(1)  # avoid oversubscription across worker processes
    _WORKER["env"] = FlyEnv(
        loom_size_gain=loom_size_gain,
        loom_exp_gain=loom_exp_gain,
        loom_exp_ref=loom_exp_ref,
    )
    _WORKER["nca"] = NCA(loom=True)


def _worker_eval(task) -> float:
    params, cfg = task
    return evaluate_on_env(_WORKER["env"], _WORKER["nca"], params, cfg)


def _resolve_workers(n_workers: int) -> int:
    if n_workers and n_workers > 0:
        return n_workers
    return max(1, (os.cpu_count() or 2) - 1)


def evaluate_population(
    xs,
    cfg: EscapeConfig,
    executor: ProcessPoolExecutor | None,
) -> list[float]:
    """Evaluate a whole population. Parallel if executor given, else sequential."""
    tasks = [(np.asarray(x, dtype=np.float64), cfg) for x in xs]
    if executor is None:
        env = _env_for_cfg(cfg)
        nca = NCA(loom=True)
        return [evaluate_on_env(env, nca, t[0], t[1]) for t in tasks]
    return list(executor.map(_worker_eval, tasks))


# --------------------------------------------------------------------------- #
# Warm start
# --------------------------------------------------------------------------- #
def load_cl_params() -> np.ndarray:
    data = json.loads(CL_CONTROLLER_JSON.read_text())
    return np.asarray(data["flat_params"], dtype=np.float64)


def warm_start_x0(cfg: EscapeConfig) -> np.ndarray:
    """Loom NCA flat params warm-started from the closed-loop controller."""
    nca = NCA(loom=True)
    nca.warm_start_from_closed_loop(load_cl_params())
    return nca.flatten_params()


# --------------------------------------------------------------------------- #
# Speedup benchmark
# --------------------------------------------------------------------------- #
def benchmark_speedup(cfg: EscapeConfig, n_individuals: int | None = None) -> dict:
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
        initargs=(cfg.loom_size_gain, cfg.loom_exp_gain, cfg.loom_exp_ref),
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
def run_evolution_escape(
    cfg: EscapeConfig,
    run_id: str | None = None,
    resume_from: Path | None = None,
    on_gen=None,
) -> tuple[float, np.ndarray, Path]:
    nca = NCA(loom=True)
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
            print(f"[evolve-x] EXACT resume from {Path(resume_from).name} (gens_done={gens_done})")
        else:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            es = cma.CMAEvolutionStrategy(
                best_params_overall,
                cfg.sigma_init,
                {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9},
            )
            phase = "resumed"
            print(f"[evolve-x] APPROXIMATE resume from {Path(resume_from).name}")
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
        f"[evolve-x] run_id={run_id}  n_params={n_params}  pop={cfg.popsize} "
        f"gens={cfg.n_gens} sigma={cfg.sigma_init}  workers={workers}  "
        f"azimuths={cfg.azimuths_deg}  speed={cfg.threat_speed} R={cfg.threat_radius} "
        f"lead={cfg.threat_lead_distance} hit_r={cfg.threat_hit_radius}  "
        f"start_gen={start_gen} phase={phase}"
    )

    last_xs: list = []
    last_fits: list = []

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg.loom_size_gain, cfg.loom_exp_gain, cfg.loom_exp_ref),
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
                f"[evolve-x] gen {gen + 1:3d}/{cfg.n_gens}  best={gen_best:.4f}  "
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
                print(f"[evolve-x]   checkpoint -> {ckpt.relative_to(REPO_ROOT)}")
    finally:
        executor.shutdown(wait=True)

    return best_fit_overall, best_params_overall, run_ckpt_dir
