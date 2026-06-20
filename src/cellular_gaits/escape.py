"""X-A web exports: controller JSON, flee clips, trajectories, metrics.

Takes an evolved escape (looming) controller (a checkpoint best) and writes the
``outputs/web_data_x/`` bundle:

    escape_controller.json   weights + full sensors spec (loom geometry, eye
                             projection, channel layout, normalization, input
                             gain) + the design choices & honesty caveats
    flee_left.mp4            threat from the left, same controller
    flee_right.mp4           threat from the right, same controller
    trajectories.json        fly + threat paths over several episodes (top-down)
    escape_metrics.json      escape success rate, reaction latency, final
                             distance, by azimuth

The threat/loom/fitness config is read back from the checkpoint's ``config_json``
so the exports match the run exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import CONTROL_DT_S, FLY_NAME, FlyEnv
from .evolve_escape import (
    EscapeConfig,
    _away_sign,
    _make_escape_policy,
    conditions,
    escape_fitness,
    threat_for_azimuth,
)
from .nca import (
    CHANNELS,
    GRID_H,
    GRID_W,
    HIDDEN_DEFAULT,
    LOOM_CHANNELS,
    LOOM_COL_SPLIT,
    MOTOR_COLS,
    MOTOR_ROWS,
    N_LEGS,
    N_MOTORS,
    NCA,
    SENSOR_CHANNELS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "outputs" / "web_data_x"

# Reaction = first post-onset step the away-turn exceeds this many radians.
REACTION_TURN_THRESHOLD = 0.05

# Design choices and honesty caveats (recorded in both the controller JSON meta
# and REPORT_x_a.md), per the X-A spec's honesty requirements.
DESIGN_CAVEATS = {
    "handbuilt_looming_frontend": (
        "The looming front-end (Threat geometry + FlyEnv.read_loom) is HAND-BUILT. "
        "It is a stand-in for the real LC4/LPLC2 -> DNp01 (Giant Fiber) escape "
        "circuit — size-tuned LPLC2 ~ the angular-size term, expansion-tuned LC4 ~ "
        "the expansion-rate term, DNp01 ~ the descending escape command. Swapping "
        "this analytic front-end for the connectome circuit is the endgame and is "
        "NOT done here; the seam (two bilateral loom input channels) is left clean."
    ),
    "loom_input_gain": (
        "The bilateral loom is read out in [0,1] (interpretable, logged as such) "
        "but multiplied by loom_input_gain before entering conv1. The warm-start "
        "gait is bang-bang (every motor cell pinned at the +/-1 clamp), so an "
        "unamplified [0,1] loom cannot move a motor cell until the loom weights "
        "grow large — a flat fitness plateau CMA-ES cannot climb from zero. "
        "Amplifying the cue (the escape analog of CH-A's deliberately strong "
        "antenna baseline) makes steering learnable in the generation budget. A/B "
        "integrity is preserved exactly: amplifying a zero loom, or amplifying "
        "through zero loom weights, is still zero."
    ),
    "target_leading": (
        "The threat aims not at the fly's onset position but at where a constant-"
        "velocity fly would be when it arrives (aim = onset_pos + lead_distance * "
        "heading). Without leading, a forward-walking fly dodges side threats for "
        "free and only a head-on threat is a real collision course. Leading makes "
        "the threat a genuine collision course from every azimuth, so a fly that "
        "keeps walking straight is hit and only an actual escape maneuver survives."
    ),
    "azimuth_scope_and_episode": (
        "Threat azimuths are {front 0, left 90, right 270} in the fly's onset-"
        "heading frame; the 180 'behind' case is omitted (from a forward walk it "
        "needs a full U-turn inside the short ~1.2 s episode). Episodes are short "
        "and the panel is small, so the generalization claim is scoped to that "
        "panel plus whatever held-out azimuths the metrics sweep adds. Do NOT "
        "compare the escape fitness scalar across behaviors (walking/chemotaxis/"
        "escape) — the reward shaping is task-specific."
    ),
}


# --------------------------------------------------------------------------- #
# Config / model helpers
# --------------------------------------------------------------------------- #
def cfg_from_checkpoint(ckpt: Path) -> EscapeConfig:
    data = np.load(Path(ckpt).with_suffix(".npz"), allow_pickle=True)
    cfg_dict = json.loads(str(data["config_json"]))
    cfg_dict["azimuths_deg"] = tuple(cfg_dict["azimuths_deg"])
    if "threat_window" in cfg_dict and cfg_dict["threat_window"] is not None:
        cfg_dict["threat_window"] = tuple(cfg_dict["threat_window"])
    return EscapeConfig(**cfg_dict)


def best_from_checkpoint(ckpt: Path) -> tuple[np.ndarray, float, str]:
    data = np.load(Path(ckpt).with_suffix(".npz"), allow_pickle=True)
    return (
        np.asarray(data["best_params"], dtype=np.float64),
        float(data["best_fit"]),
        str(data["run_id"]),
    )


def make_escape_nca(params: np.ndarray) -> NCA:
    nca = NCA(loom=True)
    nca.set_params(np.asarray(params, dtype=np.float64))
    return nca


# --------------------------------------------------------------------------- #
# Rollout + metrics
# --------------------------------------------------------------------------- #
def _reaction_latency_s(traj: dict, cfg: EscapeConfig, azimuth_deg: float) -> float | None:
    """Seconds from threat onset to the first away-turn beyond the threshold."""
    meta = traj["threat"]
    if meta is None:
        return None
    onset = int(meta["onset_step"])
    yaw = np.asarray(traj["yaw"], dtype=np.float64)
    away = _away_sign(azimuth_deg)
    if away == 0.0:  # frontal threat has no preferred side -> use |turn|
        rel = np.abs(yaw[onset:] - yaw[onset])
    else:
        rel = away * (yaw[onset:] - yaw[onset])
    idx = np.where(rel > REACTION_TURN_THRESHOLD)[0]
    if idx.size == 0:
        return None
    return float(idx[0] * CONTROL_DT_S)


def _episode(nca: NCA, cfg: EscapeConfig, azimuth_deg: float, env: FlyEnv) -> dict:
    threat = threat_for_azimuth(cfg, azimuth_deg)
    env.set_threat(threat)
    _, traj = env.rollout(
        _make_escape_policy(nca, loom_input_gain=cfg.loom_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    fc = escape_fitness(traj, cfg)
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    # distance from the threat at the final step (None before onset handled by inf)
    tp = traj["threat_path"]
    final_threat_xy = next((p for p in reversed(tp) if p is not None), None)
    final_dist = (
        float(np.linalg.norm(path[-1] - np.asarray(final_threat_xy)))
        if final_threat_xy is not None
        else float("nan")
    )
    return {
        "traj": traj,
        "path": path,
        "min_dist": fc["min_dist"],
        "hit": fc["hit"],
        "escaped": (not fc["hit"]),
        "fitness": fc["fitness"],
        "total_away_turn": fc["total_away_turn"],
        "early_away_turn": fc["early_away_turn"],
        "final_dist": final_dist,
        "reaction_latency_s": _reaction_latency_s(traj, cfg, azimuth_deg),
        "onset_step": fc["onset_step"],
    }


def compute_metrics(params: np.ndarray, cfg: EscapeConfig, extra_azimuths=()) -> dict:
    """Escape success rate, reaction latency, final/closest distance over the
    trained azimuths plus optional held-out azimuths (generalization)."""
    nca = make_escape_nca(params)
    env = FlyEnv(
        loom_size_gain=cfg.loom_size_gain,
        loom_exp_gain=cfg.loom_exp_gain,
        loom_exp_ref=cfg.loom_exp_ref,
    )
    trained = list(cfg.azimuths_deg)
    held_out = [a for a in extra_azimuths if a not in trained]

    def run_set(azis):
        rows = []
        for az in azis:
            ep = _episode(nca, cfg, az, env)
            rows.append(
                {
                    "azimuth_deg": float(az),
                    "escaped": bool(ep["escaped"]),
                    "hit": bool(ep["hit"]),
                    "min_dist": ep["min_dist"],
                    "final_dist": ep["final_dist"],
                    "reaction_latency_s": ep["reaction_latency_s"],
                    "total_away_turn": ep["total_away_turn"],
                    "early_away_turn": ep["early_away_turn"],
                }
            )
        return rows

    def agg(rows):
        if not rows:
            return {}
        lat = [r["reaction_latency_s"] for r in rows if r["reaction_latency_s"] is not None]
        return {
            "escape_success_rate": float(np.mean([r["escaped"] for r in rows])),
            "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
            "mean_final_dist": float(np.mean([r["final_dist"] for r in rows])),
            "mean_reaction_latency_s": float(np.mean(lat)) if lat else None,
            "reacted_frac": float(len(lat) / len(rows)),
        }

    trained_rows = run_set(trained)
    held_rows = run_set(held_out)
    return {
        "config": {
            "threat_speed": cfg.threat_speed,
            "threat_radius": cfg.threat_radius,
            "threat_start_distance": cfg.threat_start_distance,
            "threat_hit_radius": cfg.threat_hit_radius,
            "threat_lead_distance": cfg.threat_lead_distance,
            "loom_input_gain": cfg.loom_input_gain,
            "azimuths_deg": list(cfg.azimuths_deg),
            "rollout_steps": cfg.rollout_steps,
            "rollout_seconds": round(cfg.rollout_steps * CONTROL_DT_S, 3),
        },
        "trained": {"per_azimuth": trained_rows, "aggregate": agg(trained_rows)},
        "held_out": {"per_azimuth": held_rows, "aggregate": agg(held_rows)},
    }


def build_trajectories(params: np.ndarray, cfg: EscapeConfig, azimuths) -> dict:
    """Fly path + threat path per episode, decimated for a top-down web viz."""
    nca = make_escape_nca(params)
    env = FlyEnv(
        loom_size_gain=cfg.loom_size_gain,
        loom_exp_gain=cfg.loom_exp_gain,
        loom_exp_ref=cfg.loom_exp_ref,
    )
    episodes = []
    for az in azimuths:
        ep = _episode(nca, cfg, az, env)
        path = ep["path"]
        traj = ep["traj"]
        step = max(1, path.shape[0] // 200)  # ~<=200 points per path
        tp = traj["threat_path"]  # length = rollout_steps; None before onset
        threat_xy = [
            ([round(p[0], 4), round(p[1], 4)] if p is not None else None)
            for p in tp[::step]
        ]
        meta = traj["threat"]
        episodes.append(
            {
                "azimuth_deg": float(az),
                "onset_step": int(meta["onset_step"]),
                "aim_xy": meta["aim_xy"],
                "hit_radius": cfg.threat_hit_radius,
                "escaped": bool(ep["escaped"]),
                "min_dist": round(ep["min_dist"], 4),
                "reaction_latency_s": ep["reaction_latency_s"],
                "total_away_turn": round(ep["total_away_turn"], 4),
                "fly_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in path[::step]],
                "threat_xy": threat_xy,
            }
        )
    return {
        "loom_model": {
            "angular_size": "theta = 2*atan2(R, d)  (R = threat radius, d = fly-threat distance)",
            "expansion_rate": "rate = max(0, dtheta/dt)",
            "magnitude": "m = clip(size_gain*theta/pi + exp_gain*min(1, rate/exp_ref), 0, 1)",
            "eye_split": "loom_L = m*0.5*(1+sin phi), loom_R = m*0.5*(1-sin phi); phi = threat bearing (CCW=left)",
            "threat_radius": cfg.threat_radius,
        },
        "episodes": episodes,
    }


# --------------------------------------------------------------------------- #
# Clip rendering
# --------------------------------------------------------------------------- #
def render_flee_clip(
    params: np.ndarray,
    cfg: EscapeConfig,
    azimuth_deg: float,
    out_path: Path,
    camera_res: tuple[int, int] = (240, 320),
    playback_speed: float = 0.3,
    output_fps: int = 25,
) -> dict:
    env = FlyEnv(
        renderer_camera=f"{FLY_NAME}/trackingcam",
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
        loom_size_gain=cfg.loom_size_gain,
        loom_exp_gain=cfg.loom_exp_gain,
        loom_exp_ref=cfg.loom_exp_ref,
    )
    nca = make_escape_nca(params)
    env.set_threat(threat_for_azimuth(cfg, azimuth_deg))
    _, traj = env.rollout(
        _make_escape_policy(nca, loom_input_gain=cfg.loom_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env.sim.renderer.save_video(out_path)
    fc = escape_fitness(traj, cfg)
    return {
        "azimuth_deg": float(azimuth_deg),
        "escaped": (not fc["hit"]),
        "min_dist": fc["min_dist"],
        "total_away_turn": fc["total_away_turn"],
    }


# --------------------------------------------------------------------------- #
# Controller JSON
# --------------------------------------------------------------------------- #
def write_controller_json(
    params: np.ndarray,
    run_id: str,
    cfg: EscapeConfig,
    out_path: Path,
    best_fit: float | None = None,
) -> None:
    nca = make_escape_nca(params)
    weights = {
        name: p.detach().cpu().numpy().round(6).tolist()
        for name, p in nca.named_parameters()
    }
    payload = {
        "meta": {
            "run_id": run_id,
            "task": "X-A escape (looming threat -> emergent directed flee)",
            "n_params": int(nca.n_params),
            "channels": CHANNELS,
            "proprio_channels": SENSOR_CHANNELS,
            "loom_channels": LOOM_CHANNELS,
            "hidden": HIDDEN_DEFAULT,
            "grid": [GRID_H, GRID_W],
            "gain": nca.gain,
            "best_fitness": best_fit,
            "warm_start": "closed_loop_controller.json (C2-A); loom channels zero-init",
            "architecture": (
                "conv1: Conv2d(4 state + 2 proprio + 2 loom = 8 -> 16, 3x3, "
                "zero-pad) -> tanh(gain*.) -> conv2: Conv2d(16 -> 4, 1x1) -> "
                "clamp[-1,1]. State stays 4 channels; proprio+loom enter only at "
                "conv1's input. Loom-zeroed == closed-loop dynamics (A/B exact)."
            ),
            "connectome_alignment": (
                "Hand-built front-end stands in for LC4/LPLC2 -> DNp01 (Giant "
                "Fiber). LPLC2 ~ angular-size term, LC4 ~ expansion-rate term, "
                "DNp01 ~ descending escape command. The real circuit swap is the "
                "endgame; the two bilateral loom channels are the clean seam."
            ),
            "design_choices_and_caveats": DESIGN_CAVEATS,
        },
        "weights": weights,
        "flat_params": np.asarray(params, dtype=np.float64).round(6).tolist(),
        "sensors": {
            "description": (
                "Live proprioception (2 channels) and a bilateral looming signal "
                "(2 channels) are laid out topographically into 4 extra conv1 "
                "input channels each control step. With all four zeroed the "
                "controller reproduces the C2-A closed-loop dynamics exactly."
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
                    "input_channel_index": CHANNELS + SENSOR_CHANNELS + 0,
                    "name": "loom_left",
                    "count": 1,
                    "normalization": (
                        "loom magnitude in [0,1] split to the left eye, then "
                        f"multiplied by loom_input_gain={cfg.loom_input_gain} before conv1"
                    ),
                    "placement": (
                        f"loom ch0, left half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols 0-{LOOM_COL_SPLIT - 1})"
                    ),
                    "source": "looming signal projected onto the LEFT eye",
                },
                {
                    "input_channel_index": CHANNELS + SENSOR_CHANNELS + 1,
                    "name": "loom_right",
                    "count": 1,
                    "normalization": (
                        "loom magnitude in [0,1] split to the right eye, then "
                        f"multiplied by loom_input_gain={cfg.loom_input_gain} before conv1"
                    ),
                    "placement": (
                        f"loom ch1, right half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols {LOOM_COL_SPLIT}-{MOTOR_COLS - 1})"
                    ),
                    "source": "looming signal projected onto the RIGHT eye",
                },
            ],
            "loom_geometry": {
                "angular_size": "theta = 2*atan2(R, d)  (R = threat radius, d = fly<->threat distance); ~ LPLC2",
                "expansion_rate": "rate = max(0, dtheta/dt)  (only approach counts); ~ LC4",
                "magnitude": (
                    "m = clip(loom_size_gain*theta/pi + loom_exp_gain*min(1, rate/loom_exp_ref), 0, 1)"
                ),
                "loom_size_gain": cfg.loom_size_gain,
                "loom_exp_gain": cfg.loom_exp_gain,
                "loom_exp_ref": cfg.loom_exp_ref,
                "threat_radius": cfg.threat_radius,
            },
            "eye_projection": {
                "frame": "body pose: heading = (cos yaw, sin yaw); phi = bearing of threat in body frame (CCW positive = left)",
                "loom_left": "m * 0.5 * (1 + sin phi)",
                "loom_right": "m * 0.5 * (1 - sin phi)",
                "note": (
                    "A frontal threat (phi=0) excites both eyes equally; a threat "
                    "directly to one side excites only that eye. The L-vs-R "
                    "difference is the directional cue that makes the escape directed."
                ),
            },
            "threat_model": {
                "speed": cfg.threat_speed,
                "start_distance": cfg.threat_start_distance,
                "hit_radius": cfg.threat_hit_radius,
                "lead_distance": cfg.threat_lead_distance,
                "azimuths_deg": list(cfg.azimuths_deg),
                "onset": "seeded, mid-episode (window of the rollout)",
                "caveat": DESIGN_CAVEATS["target_leading"],
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))


def write_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))
