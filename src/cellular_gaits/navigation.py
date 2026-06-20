"""N-A web exports: controller JSON, detour clips, trajectories, metrics.

Takes an evolved navigation controller (a checkpoint best) and writes the
``outputs/web_data_n/`` bundle:

    navigation_controller.json  weights + full sensors spec (odor beacon + feeler
                                rangefinder geometry, channel layout, input gain)
                                + the design choices, honesty caveats, and the
                                operating envelope
    detour_right.mp4            obstacle on the body-LEFT -> the fly detours RIGHT
    detour_left.mp4             obstacle on the body-RIGHT -> the fly detours LEFT
    trajectories.json           fly paths + obstacle disks over several episodes
    navigation_metrics.json     trained vs HELD-OUT detour success + collision
                                counts (the generalization check)

The task/obstacle/fitness config is read back from the checkpoint's
``config_json`` so the exports match the run exactly. Held-out layouts vary the
block offset, add a third obstacle placement, and use a slightly different
obstacle radius — all still at the single goal azimuth the forager homes to.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .env import CONTROL_DT_S, FLY_NAME, FlyEnv, ObstacleField, OdorField
from .evolve_navigation import (
    NavConfig,
    _approach_feeler_LmR,
    _make_nav_policy,
    _signed_detour,
    build_condition_envs,
    conditions,
    nav_fitness,
)
from .nca import (
    CHANNELS,
    CHEMO_CHANNELS,
    CHEMO_COL_SPLIT,
    FEELER_CHANNELS,
    FEELER_COL_SPLIT,
    GRID_H,
    GRID_W,
    HIDDEN_DEFAULT,
    MOTOR_COLS,
    MOTOR_ROWS,
    N_LEGS,
    N_MOTORS,
    NCA,
    SENSOR_CHANNELS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "outputs" / "web_data_n"

# A condition counts as a clean detour-success if the fly reaches the goal AND
# spends no more than this many steps in contact with the obstacle (a brief graze
# is tolerated; the baseline collides for 100+ steps).
COLLISION_OK_STEPS = 8
# The detour direction is "correct" when its signed magnitude away from the
# natural path exceeds this (body-left obstacle -> detour right -> perp < 0).
DETOUR_TOL = 0.2

# Held-out obstacle layouts (all at the single homing azimuth, az=40). They probe
# generalization beyond the 4 trained layouts: different block offsets, a third
# (nearer) placement, and a slightly larger radius. (name, az, fraction, lateral).
HELD_OUT_SPECS_DEFAULT_RADIUS = (
    ("ho_offset_narrow_left", 40.0, 0.40, 0.8),
    ("ho_offset_narrow_right", 40.0, 0.40, -0.8),
    ("ho_offset_wide_left", 40.0, 0.40, 1.6),
    ("ho_offset_wide_right", 40.0, 0.40, -1.6),
    ("ho_near_left", 40.0, 0.28, 1.2),
    ("ho_near_right", 40.0, 0.28, -1.2),
)
HELD_OUT_SPECS_BIG_RADIUS = (
    ("ho_radius_left", 40.0, 0.40, 1.2),
    ("ho_radius_right", 40.0, 0.40, -1.2),
)
HELD_OUT_BIG_RADIUS = 2.3

DESIGN_CAVEATS = {
    "handbuilt_feeler_frontend": (
        "The feeler front-end (ObstacleField + FlyEnv.read_feelers) is a HAND-BUILT "
        "LIDAR-like rangefinder: it returns the clipped proximity of the nearest "
        "obstacle surface, split left/right by bearing. Real Drosophila avoid "
        "obstacles via VISION / optic flow / visual looming, NOT a distance sensor. "
        "Unlike escape (which maps onto LC4/LPLC2 -> DNp01), navigation has NO clean "
        "real-circuit seam — it is a robotics-flavored capability demo, not a "
        "connectome bridge. Say so plainly on the page."
    ),
    "goal_is_reused_odor_beacon": (
        "The 'goal' is the CH-A chemotaxis odor beacon reused as a homing target. "
        "This is REACTIVE local avoidance + gradient homing, NOT global path "
        "planning or spatial memory. It can get trapped in concave / dead-end "
        "obstacle configurations (a local minimum) — a real limitation, not a bug."
    ),
    "feeler_input_gain": (
        "The bilateral feeler is read out in [0,1] (interpretable, logged as such) "
        "but multiplied by feeler_input_gain before entering conv1. The warm-start "
        "gait is bang-bang (motor cells pinned at the +/-1 clamp), so an unamplified "
        "[0,1] feeler cannot move a motor cell until the feeler weights grow large. "
        "Amplifying the cue gives small feeler weights leverage so the detour is "
        "learnable in budget. A/B is preserved exactly: amplifying a zero feeler, or "
        "through zero feeler weights, is still zero. The ODOR channels are NOT "
        "amplified — they come pre-trained from chemo and steer at raw [0,1] scale."
    ),
    "operating_envelope": (
        "The controller is warm-started from the CH-A chemotaxis forager, which is "
        "an erratic, left-biased homer: it reliably homes only in a NARROW goal-"
        "azimuth band around ~40 deg (at distance 18) and does not reach 0/right/"
        "rear goals. Navigation therefore operates in that same band — the obstacle "
        "task is posed at az=40 deg only (with two placements x two block sides). "
        "The live demo must present a goal in that ~40 deg band; this is the honest "
        "operating envelope inherited from the chemo controller, not a tuning choice."
    ),
    "obstacles_on_natural_path": (
        "Obstacles are placed ON the warm-start fly's natural (no-obstacle) homing "
        "path, offset in its body frame, so the untrained forager genuinely "
        "collides (the navigation analog of escape's target leading). Without this "
        "the fly would reach the goal without ever touching an obstacle and there "
        "would be nothing to learn."
    ),
    "fitness_not_comparable": (
        "The navigation fitness scalar is task-specific (approach + reach bonus - "
        "collision penalty - time/fall). Do NOT compare it across behaviors "
        "(walking / chemotaxis / escape / navigation)."
    ),
}


# --------------------------------------------------------------------------- #
# Config / model helpers
# --------------------------------------------------------------------------- #
def cfg_from_checkpoint(ckpt: Path) -> NavConfig:
    data = np.load(Path(ckpt).with_suffix(".npz"), allow_pickle=True)
    cfg_dict = json.loads(str(data["config_json"]))
    cfg_dict["condition_specs"] = tuple(
        tuple(s) for s in cfg_dict.get("condition_specs", NavConfig().condition_specs)
    )
    return NavConfig(**cfg_dict)


def best_from_checkpoint(ckpt: Path) -> tuple[np.ndarray, float, str]:
    data = np.load(Path(ckpt).with_suffix(".npz"), allow_pickle=True)
    return (
        np.asarray(data["best_params"], dtype=np.float64),
        float(data["best_fit"]),
        str(data["run_id"]),
    )


def make_nav_nca(params: np.ndarray) -> NCA:
    nca = NCA(nav=True)
    nca.set_params(np.asarray(params, dtype=np.float64))
    return nca


def _held_out_cfgs(cfg: NavConfig) -> list[NavConfig]:
    """Held-out configs (same task params, new obstacle layouts/radius)."""
    return [
        replace(cfg, condition_specs=HELD_OUT_SPECS_DEFAULT_RADIUS),
        replace(
            cfg,
            condition_specs=HELD_OUT_SPECS_BIG_RADIUS,
            obstacle_radius=HELD_OUT_BIG_RADIUS,
        ),
    ]


# --------------------------------------------------------------------------- #
# Rollout + metrics
# --------------------------------------------------------------------------- #
def _episode(nca: NCA, cfg: NavConfig, cond, env: FlyEnv) -> dict:
    _, traj = env.rollout(
        _make_nav_policy(nca, feeler_input_gain=cfg.feeler_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    fc = nav_fitness(traj, cfg)
    detour = _signed_detour(traj, cfg, cond)
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    reached = bool(fc["reached"])
    collisions = int(fc["collision_count"])
    avoided = collisions <= COLLISION_OK_STEPS
    if cond.block_side > 0:  # obstacle body-left -> correct detour is RIGHT (perp<0)
        detour_correct = detour < -DETOUR_TOL
    elif cond.block_side < 0:
        detour_correct = detour > DETOUR_TOL
    else:
        detour_correct = abs(detour) > DETOUR_TOL
    return {
        "traj": traj,
        "path": path,
        "name": cond.name,
        "azimuth_deg": cond.azimuth_deg,
        "goal_xy": list(cond.goal_xy),
        "obstacle_xy": list(cond.obstacles[0]),
        "block_side": cond.block_side,
        "reached": reached,
        "collision_count": collisions,
        "avoided": bool(avoided),
        "detour_perp": float(detour),
        "detour_correct": bool(detour_correct),
        "detour_success": bool(reached and avoided and detour_correct),
        "min_dist": fc["min_dist"],
        "approach": fc["approach"],
        "fitness": fc["fitness"],
        "feeler_mean_LmR": _approach_feeler_LmR(traj),
    }


def _run_condition_set(nca: NCA, cfg: NavConfig) -> list[dict]:
    envs = build_condition_envs(cfg)
    conds = conditions(cfg)
    return [_episode(nca, cfg, c, e) for c, e in zip(conds, envs)]


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "detour_success_rate": float(np.mean([r["detour_success"] for r in rows])),
        "reach_rate": float(np.mean([r["reached"] for r in rows])),
        "avoid_rate": float(np.mean([r["avoided"] for r in rows])),
        "detour_correct_rate": float(np.mean([r["detour_correct"] for r in rows])),
        "mean_collisions": float(np.mean([r["collision_count"] for r in rows])),
        "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
    }


def _row_public(r: dict) -> dict:
    return {
        k: r[k]
        for k in (
            "name",
            "azimuth_deg",
            "goal_xy",
            "obstacle_xy",
            "block_side",
            "reached",
            "collision_count",
            "avoided",
            "detour_perp",
            "detour_correct",
            "detour_success",
            "min_dist",
            "approach",
            "feeler_mean_LmR",
        )
    }


def compute_metrics(params: np.ndarray, cfg: NavConfig) -> dict:
    """Trained vs held-out detour success + collision counts (generalization)."""
    nca = make_nav_nca(params)
    trained_rows = _run_condition_set(nca, cfg)
    held_rows: list[dict] = []
    for hcfg in _held_out_cfgs(cfg):
        held_rows.extend(_run_condition_set(nca, hcfg))

    return {
        "config": {
            "source_distance": cfg.source_distance,
            "odor_lambda": cfg.odor_lambda,
            "reach_radius": cfg.reach_radius,
            "obstacle_radius": cfg.obstacle_radius,
            "feeler_range": cfg.feeler_range,
            "feeler_input_gain": cfg.feeler_input_gain,
            "w_collide": cfg.w_collide,
            "rollout_steps": cfg.rollout_steps,
            "rollout_seconds": round(cfg.rollout_steps * CONTROL_DT_S, 3),
            "goal_azimuth_band_deg": 40.0,
            "collision_ok_steps": COLLISION_OK_STEPS,
        },
        "operating_envelope": DESIGN_CAVEATS["operating_envelope"],
        "trained": {
            "per_condition": [_row_public(r) for r in trained_rows],
            "aggregate": _aggregate(trained_rows),
        },
        "held_out": {
            "per_condition": [_row_public(r) for r in held_rows],
            "aggregate": _aggregate(held_rows),
        },
    }


def build_trajectories(params: np.ndarray, cfg: NavConfig) -> dict:
    """Fly path + obstacle disk + goal per episode, decimated for a top-down viz.

    Includes the 4 trained conditions, the held-out conditions, and a no-obstacle
    homing episode for reference.
    """
    nca = make_nav_nca(params)
    episodes = []

    def add(rows, group):
        for r in rows:
            path = r["path"]
            step = max(1, path.shape[0] // 200)
            episodes.append(
                {
                    "group": group,
                    "name": r["name"],
                    "azimuth_deg": r["azimuth_deg"],
                    "goal_xy": r["goal_xy"],
                    "obstacle_xy": r["obstacle_xy"],
                    "block_side": r["block_side"],
                    "reached": r["reached"],
                    "collision_count": r["collision_count"],
                    "detour_perp": round(r["detour_perp"], 4),
                    "detour_success": r["detour_success"],
                    "fly_xy": [
                        [round(float(x), 4), round(float(y), 4)] for x, y in path[::step]
                    ],
                }
            )

    add(_run_condition_set(nca, cfg), "trained")
    obstacle_radii = {"trained": cfg.obstacle_radius}
    for hcfg in _held_out_cfgs(cfg):
        rows = _run_condition_set(nca, hcfg)
        grp = "held_out_r%.1f" % hcfg.obstacle_radius
        add(rows, grp)
        obstacle_radii[grp] = hcfg.obstacle_radius

    # No-obstacle homing reference (the pure forager at the goal azimuth).
    env = FlyEnv(antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral)
    goal = conditions(cfg)[0].goal_xy
    env.set_odor(OdorField(source_xy=goal, lam=cfg.odor_lambda))
    _, traj = env.rollout(
        _make_nav_policy(nca, feeler_input_gain=cfg.feeler_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    p = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    st = max(1, p.shape[0] // 200)
    episodes.append(
        {
            "group": "no_obstacle",
            "name": "g40_no_obstacle",
            "azimuth_deg": float(conditions(cfg)[0].azimuth_deg),
            "goal_xy": list(goal),
            "obstacle_xy": None,
            "fly_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in p[::st]],
        }
    )

    return {
        "feeler_model": {
            "proximity": "p = clip(1 - d_surface / feeler_range, 0, 1)  (d_surface = nearest obstacle surface distance)",
            "field": "forward half-angle; obstacles outside the forward field are ignored",
            "feeler_split": "feeler_L = p*0.5*(1+sin phi), feeler_R = p*0.5*(1-sin phi); phi = obstacle bearing (CCW=left)",
            "feeler_range": cfg.feeler_range,
        },
        "obstacle_radii": obstacle_radii,
        "reach_radius": cfg.reach_radius,
        "episodes": episodes,
    }


# --------------------------------------------------------------------------- #
# Clip rendering
# --------------------------------------------------------------------------- #
def render_detour_clip(
    params: np.ndarray,
    cfg: NavConfig,
    cond,
    out_path: Path,
    camera_res: tuple[int, int] = (240, 320),
    playback_speed: float = 0.4,
    output_fps: int = 25,
) -> dict:
    env = FlyEnv(
        renderer_camera=f"{FLY_NAME}/trackingcam",
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
        obstacles=ObstacleField(centers=cond.obstacles, radius=cfg.obstacle_radius),
        feeler_range=cfg.feeler_range,
        antenna_forward=cfg.antenna_forward,
        antenna_lateral=cfg.antenna_lateral,
    )
    env.set_odor(OdorField(source_xy=cond.goal_xy, lam=cfg.odor_lambda))
    nca = make_nav_nca(params)
    _, traj = env.rollout(
        _make_nav_policy(nca, feeler_input_gain=cfg.feeler_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env.sim.renderer.save_video(out_path)
    fc = nav_fitness(traj, cfg)
    return {
        "name": cond.name,
        "block_side": cond.block_side,
        "reached": bool(fc["reached"]),
        "collision_count": int(fc["collision_count"]),
        "detour_perp": float(_signed_detour(traj, cfg, cond)),
    }


# --------------------------------------------------------------------------- #
# Controller JSON
# --------------------------------------------------------------------------- #
def write_controller_json(
    params: np.ndarray,
    run_id: str,
    cfg: NavConfig,
    out_path: Path,
    best_fit: float | None = None,
) -> None:
    nca = make_nav_nca(params)
    weights = {
        name: p.detach().cpu().numpy().round(6).tolist()
        for name, p in nca.named_parameters()
    }
    odor0 = CHANNELS + SENSOR_CHANNELS
    feel0 = CHANNELS + SENSOR_CHANNELS + CHEMO_CHANNELS
    payload = {
        "meta": {
            "run_id": run_id,
            "task": "N-A navigation (seek odor goal + avoid obstacles -> emergent detour)",
            "n_params": int(nca.n_params),
            "channels": CHANNELS,
            "proprio_channels": SENSOR_CHANNELS,
            "odor_channels": CHEMO_CHANNELS,
            "feeler_channels": FEELER_CHANNELS,
            "hidden": HIDDEN_DEFAULT,
            "grid": [GRID_H, GRID_W],
            "gain": nca.gain,
            "best_fitness": best_fit,
            "warm_start": "chemotaxis_controller.json (CH-A); feeler channels zero-init",
            "architecture": (
                "conv1: Conv2d(4 state + 2 proprio + 2 odor + 2 feeler = 10 -> 16, "
                "3x3, zero-pad) -> tanh(gain*.) -> conv2: Conv2d(16 -> 4, 1x1) -> "
                "clamp[-1,1]. State stays 4 channels; proprio+odor+feeler enter only "
                "at conv1's input. Feeler-zeroed == chemotaxis dynamics (A/B exact)."
            ),
            "compositional_story": (
                "Obstacle avoidance is stacked ADDITIVELY on the trained chemotaxis "
                "forager: seeking comes for free from the warm start, the two new "
                "feeler channels add avoidance. The detour direction must be "
                "arbitrated from the odor L-R (turn toward goal) vs the feeler L-R "
                "(turn away from wall) — nothing hard-codes which way to dodge."
            ),
            "connectome_alignment": (
                "NONE. Unlike escape (LC4/LPLC2 -> DNp01), navigation has no clean "
                "real-circuit seam — the feeler is a robotics rangefinder abstraction, "
                "not a Drosophila sensory pathway. This is a capability demo."
            ),
            "operating_envelope": DESIGN_CAVEATS["operating_envelope"],
            "design_choices_and_caveats": DESIGN_CAVEATS,
        },
        "weights": weights,
        "flat_params": np.asarray(params, dtype=np.float64).round(6).tolist(),
        "sensors": {
            "description": (
                "Live proprioception (2 channels), a bilateral odor goal-beacon "
                "(2 channels), and a bilateral feeler obstacle-proximity reading "
                "(2 channels) are laid out topographically into 6 extra conv1 input "
                "channels each control step. With all six zeroed the controller "
                "reproduces the C2-A closed-loop walking dynamics; with the feeler "
                "channels zeroed it reproduces the CH-A chemotaxis dynamics exactly."
            ),
            "channels": [
                {
                    "input_channel_index": CHANNELS + 0,
                    "name": "joint_angles",
                    "count": N_MOTORS,
                    "normalization": "joint_angle_rad / 3.14, clipped to [-1, 1]",
                    "placement": "proprio ch0, 7x6 motor block (rows 0-6, cols 0-5)",
                    "source": "mj_data.actuator_length at the 42 position actuators",
                },
                {
                    "input_channel_index": CHANNELS + 1,
                    "name": "foot_contacts",
                    "count": N_LEGS,
                    "normalization": "boolean {0, 1}",
                    "placement": "proprio ch1, bottom row (row 7, cols 0-5)",
                    "source": "get_ground_contact_info()[0] per-leg booleans",
                },
                {
                    "input_channel_index": odor0 + 0,
                    "name": "odor_left",
                    "count": 1,
                    "normalization": "concentration in (0,1] at the left antenna",
                    "placement": (
                        f"odor ch0, left half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols 0-{CHEMO_COL_SPLIT - 1})"
                    ),
                    "source": "odor goal-beacon concentration at the LEFT antenna",
                },
                {
                    "input_channel_index": odor0 + 1,
                    "name": "odor_right",
                    "count": 1,
                    "normalization": "concentration in (0,1] at the right antenna",
                    "placement": (
                        f"odor ch1, right half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols {CHEMO_COL_SPLIT}-{MOTOR_COLS - 1})"
                    ),
                    "source": "odor goal-beacon concentration at the RIGHT antenna",
                },
                {
                    "input_channel_index": feel0 + 0,
                    "name": "feeler_left",
                    "count": 1,
                    "normalization": (
                        "obstacle proximity in [0,1] split to the left field, then "
                        f"multiplied by feeler_input_gain={cfg.feeler_input_gain} before conv1"
                    ),
                    "placement": (
                        f"feeler ch0, left half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols 0-{FEELER_COL_SPLIT - 1})"
                    ),
                    "source": "nearest-obstacle proximity projected onto the LEFT field",
                },
                {
                    "input_channel_index": feel0 + 1,
                    "name": "feeler_right",
                    "count": 1,
                    "normalization": (
                        "obstacle proximity in [0,1] split to the right field, then "
                        f"multiplied by feeler_input_gain={cfg.feeler_input_gain} before conv1"
                    ),
                    "placement": (
                        f"feeler ch1, right half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols {FEELER_COL_SPLIT}-{MOTOR_COLS - 1})"
                    ),
                    "source": "nearest-obstacle proximity projected onto the RIGHT field",
                },
            ],
            "odor_geometry": {
                "field": "C(p) = exp(-||p - source|| / lambda) on the ground plane",
                "odor_lambda": cfg.odor_lambda,
                "antenna_forward": cfg.antenna_forward,
                "antenna_lateral": cfg.antenna_lateral,
                "note": "reused CH-A chemotaxis odor beacon as the navigation goal",
            },
            "feeler_geometry": {
                "proximity": "p = clip(1 - d_surface / feeler_range, 0, 1)  (d_surface = nearest obstacle surface distance)",
                "feeler_range": cfg.feeler_range,
                "field": "forward half-angle; obstacles outside the forward field ignored",
                "eye_split": "feeler_L = p*0.5*(1+sin phi), feeler_R = p*0.5*(1-sin phi); phi = obstacle bearing (CCW=left)",
                "obstacle_radius": cfg.obstacle_radius,
                "caveat": DESIGN_CAVEATS["handbuilt_feeler_frontend"],
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))


def write_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))
