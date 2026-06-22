"""CMA-ES evolution of the navigation NCA controller, in PARALLEL.

This is the N-A evolver — the **synthesis** behavior. It is the structural twin
of the CH-A chemotaxis evolver (``evolve_chemotaxis.py``) and the X-A escape
evolver (``evolve_escape.py``): same parallel ProcessPool machinery and
checkpoint format. The novelty is that the policy carries TWO bilateral cues at
once and must arbitrate them.

1. **Both cues at once.** The policy is the 10-input nav NCA (``NCA(nav=True)``):
   4 state + 2 proprio + 2 odor (goal beacon) + 2 feeler (obstacle proximity).
   We get the *seeking* for free by warm-starting from the trained **chemotaxis**
   controller (``NCA.warm_start_from_chemo``) with the feeler channels at zero, so
   the search starts from a controller that already homes on the odor beacon, and
   a feeler-zeroed rollout reproduces the chemotaxis dynamics exactly (A/B). We
   re-evolve so it learns to **detour around obstacles while still homing**.

2. **Seek-vs-avoid fitness + layout generalization.** Per obstacle layout:

        F_condition = approach   (= d_start - min_dist to goal, chemo's shaping)
                    + reach_bonus * [reached goal]
                    - w_collide  * collision_count        (time-in-contact)
                    - time_penalty * (steps_to_reach / N)
                    - STABILITY_PENALTY_PER_STEP * n_below

        fitness = mean over layouts of F_condition

   Conditions vary the goal azimuth AND the obstacle placement so a fixed swerve
   cannot win: the left-blocking layout must be solved by a RIGHT detour and the
   right-blocking layout by a LEFT detour. Nothing hard-codes the detour
   direction; it must fall out of ``feeler_L`` vs ``feeler_R``, arbitrated against
   the turn-toward-goal carried by the odor L-R. Obstacles are placed ON the
   straight homing path (calibrated so the untrained warm-start forager actually
   collides — the navigation analog of escape's target leading).

HONESTY: the feeler front-end (``ObstacleField`` + ``FlyEnv.read_feelers``) is a
HAND-BUILT LIDAR-like rangefinder. Real Drosophila avoid obstacles via vision /
optic flow / visual looming, NOT a distance sensor — navigation has NO clean
real-circuit seam (unlike escape's LC4/LPLC2 -> DNp01). The "goal" is the
chemotaxis odor beacon reused: this is reactive local avoidance + gradient
homing, NOT global path planning or spatial memory, so it can get trapped in
concave/dead-end obstacle configurations (a local minimum). The nav fitness
scalar is task-specific and not comparable across behaviors.
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
    DEFAULT_ODOR_LAMBDA,
    FEELER_RANGE,
    OBSTACLE_RADIUS,
    STABILITY_PENALTY_PER_STEP,
    FlyEnv,
    ObstacleField,
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

# Warm-start source: the trained 8-input chemotaxis controller (1236 flat
# params). Lives in the gitignored outputs/ tree, produced by the CH-A export.
CHEMO_CONTROLLER_JSON = (
    REPO_ROOT / "outputs" / "web_data_ch" / "chemotaxis_controller.json"
)

CA_INIT_SEED = 0  # fixed NCA.init_state seed across all rollouts

# Navigation episodes run longer than the pure chemo rollout (~4 s vs 3 s): the
# fly needs time to home, detour around the obstacle, and re-home to the goal.
NAV_ROLLOUT_STEPS = 1000  # 4.0 s at 250 Hz

# Condition specs: (name, goal_azimuth_deg, obstacle_fraction, lateral_offset).
# ``goal_azimuth_deg`` is CCW from the +x spawn heading (like chemo). The
# obstacle is placed ``fraction`` of the way (by step index) to the warm-start
# fly's closest approach to the goal along its NATURAL path, shifted
# ``lateral_offset`` world-units along the fly's body-LEFT (+) / body-RIGHT (-) at
# that point. A nonzero offset smaller than the obstacle radius keeps the path
# clipping the disk (so the forager collides) while biasing the dodge direction:
#   block_left  (+offset) -> obstacle on the body-LEFT  -> detour RIGHT
#   block_right (-offset) -> obstacle on the body-RIGHT -> detour LEFT
# Left/right blocking are symmetric so a fixed turn bias cannot win both.
#
# IMPORTANT (warm-start forager is an erratic, biased homer): the trained
# chemotaxis controller does NOT home omnidirectionally — it has a strong
# left-turning bias and only reaches a narrow, non-contiguous set of goal
# azimuths (it reaches ~30/40/55 deg at distance 18 but not 35/45/50, and never
# 0/right/rear goals). Of those, only **az=40** approaches the goal along a
# reasonably direct path (heading ~+44 deg straight at the goal early on); the
# az=30/55 paths are big loops that overshoot and double back, so a mid-path
# obstacle lands on the loop excursion with an ambiguous body-frame bearing. So
# the nav task is built on the single clean azimuth (40 deg) with TWO obstacle
# placements along its approach (near/far) x two block sides = 4 conditions. The
# left/right block at each placement is the anti-bias mechanism (the detour
# direction MUST come from the feeler L-R, not a fixed swerve). Goal-azimuth
# variation is genuinely limited by the forager's narrow homing band — an honest
# constraint of warm-starting from this particular (imperfect) chemotaxis
# controller, documented in the report.
# Lateral offset 1.8 (with radius 2.0): under the CG contact solver the warm-start
# forager's left-turning bias makes a body-RIGHT obstacle read near-dead-ahead, so
# a 1.2 offset gives an ambiguous right-block cue. Pushing the offset to 1.8
# (still < radius, so the path still clips the disk and the baseline collides)
# restores a correctly-signed bilateral feeler cue on both block sides. The cue
# stays stronger on the left than the right — an honest artifact of the forager's
# left bias, documented in the report.
DEFAULT_CONDITION_SPECS = (
    ("g40_near_block_left", 40.0, 0.35, 1.8),
    ("g40_near_block_right", 40.0, 0.35, -1.8),
    ("g40_far_block_left", 40.0, 0.45, 1.8),
    ("g40_far_block_right", 40.0, 0.45, -1.8),
)


@dataclass(frozen=True)
class NavCondition:
    """One evaluation condition: a goal beacon and an obstacle layout."""

    name: str
    azimuth_deg: float
    goal_xy: tuple[float, float]
    obstacles: tuple[tuple[float, float], ...]
    block_side: float  # +1 obstacle on body-left (detour right), -1 right, 0 centered


@dataclass
class NavConfig:
    seed: int = 0
    popsize: int = 48
    n_gens: int = 60
    sigma_init: float = 0.2
    rollout_steps: int = NAV_ROLLOUT_STEPS
    checkpoint_every: int = 5
    # --- task geometry ---
    source_distance: float = 18.0
    odor_lambda: float = DEFAULT_ODOR_LAMBDA
    reach_radius: float = 3.0
    obstacle_radius: float = OBSTACLE_RADIUS
    feeler_range: float = FEELER_RANGE
    antenna_forward: float = ANTENNA_FORWARD_OFFSET
    antenna_lateral: float = ANTENNA_LATERAL_OFFSET
    condition_specs: tuple[tuple, ...] = DEFAULT_CONDITION_SPECS
    # --- feeler input amplification ---
    # The warm-start gait is bang-bang (motor cells pinned at the +/-1 clamp), so
    # an unamplified [0,1] feeler cannot move a motor cell until the feeler
    # weights grow large — a flat plateau CMA-ES cannot climb from zero. We
    # amplify the feeler cue before it enters conv1 (the nav analog of escape's
    # loom_input_gain) so small feeler weights get immediate leverage and the
    # detour is learnable in budget. The ODOR channels are NOT amplified — they
    # come pre-trained from chemo and already steer at raw [0,1] scale. A/B is
    # preserved exactly: amplifying a zero feeler, or through zero feeler weights,
    # is still zero.
    feeler_input_gain: float = 8.0
    # --- fitness shaping ---
    # "closest": approach = d_start - min_dist over the rollout (robust to a fly
    #   that overshoots/gets deflected); reach = it entered reach_radius.
    fitness_mode: str = "closest"
    reach_bonus: float = 8.0
    time_penalty: float = 4.0
    # Collision penalty per in-contact step (time-in-contact). Calibrated so
    # DETOURING beats plowing through: it must outweigh the approach the fly would
    # gain by walking straight into the obstacle. Raised above its starting guess
    # if a fixed swerve wins (documented in the report), the way escape raised its
    # directional weights above survival.
    w_collide: float = 0.2
    n_workers: int = 0  # 0 -> os.cpu_count()-1


# --------------------------------------------------------------------------- #
# Task conditions (goal + obstacle layouts)
# --------------------------------------------------------------------------- #
def _goal_xy(distance: float, azimuth_deg: float) -> tuple[float, float]:
    a = np.radians(azimuth_deg)
    return (float(distance * np.cos(a)), float(distance * np.sin(a)))


# Obstacle layouts are placed on the warm-start fly's NATURAL (no-obstacle) path,
# offset perpendicular to its ACTUAL heading there — not the straight start->goal
# line. The natural homing gait curves (a consistent yaw drift), so a straight-
# line offset would give an ambiguous body-frame bearing; placing relative to the
# real path guarantees both a genuine collision (obstacle is on the path) AND a
# cleanly-signed feeler cue (offset is in the fly's body frame). The natural path
# is deterministic (fixed warm start + CA seed), so every process computes the
# identical layout; results are cached per config.
_NATURAL_CACHE: dict = {}
_LAYOUT_CACHE: dict = {}


def _layout_key(cfg: NavConfig) -> tuple:
    return (
        round(cfg.source_distance, 6),
        round(cfg.odor_lambda, 6),
        round(cfg.antenna_forward, 6),
        round(cfg.antenna_lateral, 6),
        round(cfg.feeler_input_gain, 6),
        int(cfg.rollout_steps),
        tuple(tuple(s) for s in cfg.condition_specs),
    )


def _natural_path(cfg: NavConfig, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Warm-start fly's no-obstacle path + yaw homing on the goal at ``azimuth``."""
    key = (_layout_key(cfg), round(float(azimuth_deg), 6))
    if key in _NATURAL_CACHE:
        return _NATURAL_CACHE[key]
    env = FlyEnv(
        antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral
    )
    env.set_odor(
        OdorField(source_xy=_goal_xy(cfg.source_distance, azimuth_deg), lam=cfg.odor_lambda)
    )
    nca = NCA(nav=True)
    nca.warm_start_from_chemo(load_chemo_params())
    _, traj = env.rollout(
        _make_nav_policy(nca, feeler_input_gain=cfg.feeler_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    yaw = np.asarray(traj["yaw"], dtype=np.float64)
    _NATURAL_CACHE[key] = (path, yaw)
    return path, yaw


def conditions(cfg: NavConfig) -> list[NavCondition]:
    """One NavCondition per spec: goal beacon + a single obstacle on the path.

    For each spec ``(name, azimuth, fraction, lateral)`` we take the warm-start
    natural path toward that goal, walk to ``fraction`` of the way to its closest
    approach to the goal, and place the obstacle there shifted ``lateral`` world-
    units along the fly's body-LEFT (+) / body-RIGHT (-) at that point.
    """
    cached = _LAYOUT_CACHE.get(_layout_key(cfg))
    if cached is not None:
        return cached
    out: list[NavCondition] = []
    for name, az, frac, lateral in cfg.condition_specs:
        goal = _goal_xy(cfg.source_distance, az)
        path, yaw = _natural_path(cfg, az)
        dist = np.linalg.norm(path - np.asarray(goal)[None, :], axis=1)
        idx_close = int(np.argmin(dist))
        idx = int(np.clip(round(frac * idx_close), 1, len(path) - 1))
        y = float(yaw[idx])
        left = np.array([-np.sin(y), np.cos(y)])  # body-left unit vector
        center = path[idx] + float(lateral) * left
        out.append(
            NavCondition(
                name=str(name),
                azimuth_deg=float(az),
                goal_xy=(float(goal[0]), float(goal[1])),
                obstacles=((float(center[0]), float(center[1])),),
                block_side=float(np.sign(lateral)),
            )
        )
    _LAYOUT_CACHE[_layout_key(cfg)] = out
    return out


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def nav_fitness(traj: dict, cfg: NavConfig) -> dict:
    """Seek-vs-avoid fitness + components from a nav rollout traj."""
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    src = np.asarray(traj["odor_source_xy"], dtype=np.float64)
    dist = np.linalg.norm(path - src[None, :], axis=1)
    n = dist.size
    d_start = float(dist[0])
    d_end = float(dist[-1])
    min_dist = float(dist.min())
    n_below = int(traj["n_below"])
    collision_count = int(traj["collision_count"])

    reached_idx = np.where(dist < cfg.reach_radius)[0]
    reached = bool(reached_idx.size > 0)
    steps_to_reach = int(reached_idx[0]) if reached else (n - 1)
    reach_frac = steps_to_reach / max(1, n - 1)

    if cfg.fitness_mode == "d_end":
        approach = d_start - d_end
        reach_for_bonus = bool(d_end < cfg.reach_radius)
    else:  # "closest"
        approach = d_start - min_dist
        reach_for_bonus = reached
    bonus = cfg.reach_bonus if reach_for_bonus else 0.0
    collide_pen = cfg.w_collide * collision_count
    time_pen = cfg.time_penalty * reach_frac
    fall_pen = STABILITY_PENALTY_PER_STEP * n_below
    fitness = approach + bonus - collide_pen - time_pen - fall_pen
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
        "collision_count": collision_count,
        "collide_pen": float(collide_pen),
        "time_pen": float(time_pen),
        "fall_pen": float(fall_pen),
        "n_below": n_below,
    }


def _signed_detour(traj: dict, cfg: NavConfig, cond: NavCondition) -> float:
    """Signed detour (world units) away from the NATURAL path while feeler is hot.

    + = the fly deviated to the LEFT of where it would have gone with no obstacle
    (a LEFT detour), - = RIGHT, in the natural-heading body frame. Measured over
    the steps where the feeler is meaningfully active. Comparing against the
    natural path (not the straight goal line) removes the warm-start gait's yaw
    drift, so the sign is the genuine obstacle-induced detour direction.
    """
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    feeler = np.asarray(traj["feeler"], dtype=np.float64)
    if path.size == 0 or feeler.size == 0:
        return 0.0
    nat_path, nat_yaw = _natural_path(cfg, cond.azimuth_deg)
    m = min(len(path), len(nat_path), len(feeler) + 1)
    devs = []
    for i in range(1, m):
        if feeler[i - 1].sum() <= 0.1:
            continue
        y = float(nat_yaw[i])
        left = np.array([-np.sin(y), np.cos(y)])
        devs.append(float((path[i] - nat_path[i]) @ left))
    return float(np.mean(devs)) if devs else 0.0


def _approach_feeler_LmR(traj: dict) -> float:
    """Mean feeler (L - R) over the APPROACH window (first-hot -> peak magnitude).

    The bilateral cue is only clean while the fly is approaching the obstacle.
    Once it physically contacts and gets deflected, its heading wanders and the
    obstacle bearing (hence the L-R sign) becomes unreliable — averaging over all
    hot steps washes the cue out. We therefore measure the cue over the rising
    edge of the feeler signal (first hot step up to the peak, ~closest approach),
    which is exactly the window in which the controller must decide its dodge.
    """
    feeler = np.asarray(traj["feeler"], dtype=np.float64)
    if feeler.size == 0:
        return 0.0
    mag = feeler.sum(axis=1)
    hot = np.where(mag > 0.1)[0]
    if hot.size == 0:
        return 0.0
    first = int(hot[0])
    peak = max(first, int(np.argmax(mag)))
    seg = feeler[first : peak + 1]
    segmag = mag[first : peak + 1]
    sel = seg[segmag > 0.1]
    return float((sel[:, 0] - sel[:, 1]).mean()) if sel.size else 0.0


def _make_nav_policy(
    nca: NCA, ca_seed: int = CA_INIT_SEED, feeler_input_gain: float = 8.0
):
    state = NCA.init_state(seed=ca_seed)

    def policy(t: int, sensors: dict) -> np.ndarray:
        nonlocal state
        smap = NCA.build_nav_sensor_map(
            sensors["joint_angles_unit"],
            sensors["foot_contacts"],
            sensors.get("c_left", 0.0),
            sensors.get("c_right", 0.0),
            feeler_input_gain * sensors.get("feeler_left", 0.0),
            feeler_input_gain * sensors.get("feeler_right", 0.0),
        )
        with torch.no_grad():
            state = nca.step(state, sensors=smap)
        return nca.motor_targets(state)

    return policy


def build_condition_envs(cfg: NavConfig) -> list[FlyEnv]:
    """One FlyEnv per condition (the obstacle geoms are baked in at construction).

    The matching goal odor field is set on each env here too, so an env is fully
    paired with its condition and ``evaluate_on_envs`` just runs rollouts.
    """
    envs: list[FlyEnv] = []
    for cond in conditions(cfg):
        env = FlyEnv(
            obstacles=ObstacleField(
                centers=cond.obstacles, radius=cfg.obstacle_radius
            ),
            feeler_range=cfg.feeler_range,
            antenna_forward=cfg.antenna_forward,
            antenna_lateral=cfg.antenna_lateral,
        )
        env.set_odor(OdorField(source_xy=cond.goal_xy, lam=cfg.odor_lambda))
        envs.append(env)
    return envs


def evaluate_on_envs(
    envs: list[FlyEnv],
    nca: NCA,
    params: np.ndarray,
    cfg: NavConfig,
    return_components: bool = False,
):
    """Evaluate one param vector over all conditions (one pre-built env each)."""
    nca.set_params(np.asarray(params, dtype=np.float64))
    comps = []
    for env in envs:
        _, traj = env.rollout(
            _make_nav_policy(nca, feeler_input_gain=cfg.feeler_input_gain),
            n_steps=cfg.rollout_steps,
            pass_sensors=True,
        )
        comps.append(nav_fitness(traj, cfg))
    mean_fit = float(np.mean([c["fitness"] for c in comps]))
    if return_components:
        return mean_fit, comps
    return mean_fit


def diagnose_params(
    params: np.ndarray, cfg: NavConfig, envs: list[FlyEnv] | None = None
) -> list[dict]:
    """Per-condition diagnostics for one controller: reach, collisions, the
    detour it produced, and the bilateral feeler cue strength. Used by the
    calibration / validation gate.
    """
    if envs is None:
        envs = build_condition_envs(cfg)
    nca = NCA(nav=True)
    nca.set_params(np.asarray(params, dtype=np.float64))
    conds = conditions(cfg)
    rows = []
    for env, cond in zip(envs, conds):
        _, traj = env.rollout(
            _make_nav_policy(nca, feeler_input_gain=cfg.feeler_input_gain),
            n_steps=cfg.rollout_steps,
            pass_sensors=True,
        )
        fc = nav_fitness(traj, cfg)
        feeler = np.asarray(traj["feeler"], dtype=np.float64)
        mag = feeler.sum(axis=1) if feeler.size else np.zeros(1)
        feeler_lmr = _approach_feeler_LmR(traj)  # cue measured over the approach
        detour = _signed_detour(traj, cfg, cond)
        rows.append(
            {
                "name": cond.name,
                "goal_xy": [float(v) for v in cond.goal_xy],
                "obstacle_xy": [float(v) for v in cond.obstacles[0]],
                "block_side": cond.block_side,
                "fitness": fc["fitness"],
                "approach": fc["approach"],
                "d_start": fc["d_start"],
                "min_dist": fc["min_dist"],
                "reached": fc["reached"],
                "collision_count": fc["collision_count"],
                "collided": bool(fc["collision_count"] > 0),
                "n_below": fc["n_below"],
                "forward_dx": float(traj["forward_dx"]),
                "detour_perp": detour,  # + = went left, - = went right
                "feeler_peak": float(mag.max()) if mag.size else 0.0,
                "feeler_mean_LmR": feeler_lmr,  # approach-window L-R cue
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Parallel worker plumbing (module-global env cache per worker process)
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _worker_init(cfg: NavConfig) -> None:
    torch.set_num_threads(1)  # avoid oversubscription across worker processes
    _WORKER["envs"] = build_condition_envs(cfg)
    _WORKER["nca"] = NCA(nav=True)


def _worker_eval(task) -> float:
    params, cfg = task
    return evaluate_on_envs(_WORKER["envs"], _WORKER["nca"], params, cfg)


def _resolve_workers(n_workers: int) -> int:
    if n_workers and n_workers > 0:
        return n_workers
    return max(1, (os.cpu_count() or 2) - 1)


def evaluate_population(
    xs,
    cfg: NavConfig,
    executor: ProcessPoolExecutor | None,
    envs: list[FlyEnv] | None = None,
) -> list[float]:
    """Evaluate a whole population. Parallel if executor given, else sequential."""
    tasks = [(np.asarray(x, dtype=np.float64), cfg) for x in xs]
    if executor is None:
        if envs is None:
            envs = build_condition_envs(cfg)
        nca = NCA(nav=True)
        return [evaluate_on_envs(envs, nca, t[0], t[1]) for t in tasks]
    return list(executor.map(_worker_eval, tasks))


# --------------------------------------------------------------------------- #
# Warm start
# --------------------------------------------------------------------------- #
def load_chemo_params() -> np.ndarray:
    data = json.loads(CHEMO_CONTROLLER_JSON.read_text())
    return np.asarray(data["flat_params"], dtype=np.float64)


def warm_start_x0(cfg: NavConfig) -> np.ndarray:
    """Nav NCA flat params warm-started from the trained chemotaxis controller."""
    nca = NCA(nav=True)
    nca.warm_start_from_chemo(load_chemo_params())
    return nca.flatten_params()


# --------------------------------------------------------------------------- #
# Speedup benchmark
# --------------------------------------------------------------------------- #
def benchmark_speedup(cfg: NavConfig, n_individuals: int | None = None) -> dict:
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
        max_workers=workers, initializer=_worker_init, initargs=(cfg,)
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
def run_evolution_navigation(
    cfg: NavConfig,
    run_id: str | None = None,
    resume_from: Path | None = None,
    on_gen=None,
) -> tuple[float, np.ndarray, Path]:
    nca = NCA(nav=True)
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
            print(f"[evolve-n] EXACT resume from {Path(resume_from).name} (gens_done={gens_done})")
        else:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            es = cma.CMAEvolutionStrategy(
                best_params_overall,
                cfg.sigma_init,
                {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9},
            )
            phase = "resumed"
            print(f"[evolve-n] APPROXIMATE resume from {Path(resume_from).name}")
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
        f"[evolve-n] run_id={run_id}  n_params={n_params}  pop={cfg.popsize} "
        f"gens={cfg.n_gens} sigma={cfg.sigma_init}  workers={workers}  "
        f"conds={[c.name for c in conditions(cfg)]}  dist={cfg.source_distance} "
        f"obs_r={cfg.obstacle_radius} feeler_range={cfg.feeler_range} "
        f"w_collide={cfg.w_collide} feeler_gain={cfg.feeler_input_gain}  "
        f"start_gen={start_gen} phase={phase}"
    )

    last_xs: list = []
    last_fits: list = []

    executor = ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(cfg,)
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
                f"[evolve-n] gen {gen + 1:3d}/{cfg.n_gens}  best={gen_best:.4f}  "
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
                print(f"[evolve-n]   checkpoint -> {ckpt.relative_to(REPO_ROOT)}")
    finally:
        executor.shutdown(wait=True)

    return best_fit_overall, best_params_overall, run_ckpt_dir
