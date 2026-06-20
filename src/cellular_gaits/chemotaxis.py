"""CH-A web exports: controller JSON, approach clips, trajectories, metrics.

Takes an evolved chemotaxis controller (a checkpoint best) and writes the
``outputs/web_data_ch/`` bundle:

    chemotaxis_controller.json   weights + full sensors spec (chemo channels,
                                 antenna geometry, field formula, normalization)
                                 + the three calibrated design choices & caveats
    approach_left.mp4            source on the left, same controller
    approach_right.mp4           source on the right, same controller
    trajectories.json            fly path(s) + source(s) over several episodes
    chemotaxis_metrics.json      success rate, mean closest distance, tortuosity

The fitness/odor/antenna config is read back from the checkpoint's
``config_json`` so the exports match the run exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .env import FLY_NAME, FlyEnv, OdorField
from .evolve_chemotaxis import (
    ChemoConfig,
    _make_chemo_policy,
    chemo_fitness,
    source_for_azimuth,
)
from .nca import (
    CHANNELS,
    CHEMO_CHANNELS,
    CHEMO_COL_SPLIT,
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
WEB_DIR = REPO_ROOT / "outputs" / "web_data_ch"

# The three calibrated design choices and their caveats (recorded in both the
# controller JSON meta and REPORT_ch_a.md), per the CH-A green-light.
DESIGN_CAVEATS = {
    "antenna_lateral_baseline": (
        "Antenna left/right baseline is 2.0 world units — a deliberately strong, "
        "larger-than-biological bilateral spread. It stands in for the temporal "
        "'casting' a real fly uses to amplify a weak instantaneous gradient: with "
        "a narrow baseline the per-step |cL-cR| signal (~0.02-0.04) is too weak for "
        "CMA-ES to exploit in 50 generations. At 2.0 the cue is ~0.07-0.26 and "
        "steering emerges within ~5 generations."
    ),
    "behind_case_deferred": (
        "Source azimuths are {ahead 0°, left 90°, right 270°}. The 180° 'behind' "
        "case from the spec is deferred: from a forward-walking warm start it "
        "requires a ~180° U-turn within the 3 s rollout and stayed unreached in "
        "calibration. The headline claim — emergent turning toward a source on "
        "either side — is carried by the symmetric left/right pair."
    ),
    "closest_approach_fitness": (
        "Fitness scores closest approach (d_start - min_dist) + reach bonus, NOT "
        "the literal spec's final distance (d_start - d_end). The warm-started "
        "walker travels ~70 units in 3 s and overshoots the source, so d_end is "
        "dominated by overshoot rather than approach. Closest-approach correctly "
        "rewards how close the fly steered; learning to arrive-and-stop is future "
        "work. This is a documented deviation from the literal CH-A reward."
    ),
}


# --------------------------------------------------------------------------- #
# Config / model helpers
# --------------------------------------------------------------------------- #
def cfg_from_checkpoint(ckpt: Path) -> ChemoConfig:
    data = np.load(Path(ckpt).with_suffix(".npz"), allow_pickle=True)
    cfg_dict = json.loads(str(data["config_json"]))
    cfg_dict["azimuths_deg"] = tuple(cfg_dict["azimuths_deg"])
    return ChemoConfig(**cfg_dict)


def best_from_checkpoint(ckpt: Path) -> tuple[np.ndarray, float, str]:
    data = np.load(Path(ckpt).with_suffix(".npz"), allow_pickle=True)
    return (
        np.asarray(data["best_params"], dtype=np.float64),
        float(data["best_fit"]),
        str(data["run_id"]),
    )


def make_chemo_nca(params: np.ndarray) -> NCA:
    nca = NCA(chemo=True)
    nca.set_params(np.asarray(params, dtype=np.float64))
    return nca


# --------------------------------------------------------------------------- #
# Rollout + metrics
# --------------------------------------------------------------------------- #
def _episode(nca: NCA, cfg: ChemoConfig, source_xy, env: FlyEnv) -> dict:
    env.set_odor(OdorField(source_xy=source_xy, lam=cfg.odor_lambda))
    _, traj = env.rollout(
        _make_chemo_policy(nca), n_steps=cfg.rollout_steps, pass_sensors=True
    )
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    src = np.asarray(source_xy, dtype=np.float64)
    dist = np.linalg.norm(path - src[None, :], axis=1)
    closest_idx = int(np.argmin(dist))
    seg = path[: closest_idx + 1]
    path_len = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))) if seg.shape[0] > 1 else 0.0
    straight = float(np.linalg.norm(seg[closest_idx] - seg[0])) if seg.shape[0] > 1 else 0.0
    tortuosity = float(path_len / straight) if straight > 1e-6 else float("inf")
    fc = chemo_fitness(traj, cfg.reach_radius, cfg.reach_bonus, cfg.time_penalty, cfg.fitness_mode)
    return {
        "traj": traj,
        "path": path,
        "dist": dist,
        "min_dist": fc["min_dist"],
        "reached": fc["reached"],
        "fitness": fc["fitness"],
        "approach": fc["approach"],
        "tortuosity": tortuosity,
        "path_len": path_len,
        "straight": straight,
        "dyaw_final": float(np.asarray(traj["yaw"])[-1] - traj["yaw0"]),
    }


def compute_metrics(params: np.ndarray, cfg: ChemoConfig, extra_azimuths=()) -> dict:
    """Success rate, mean closest distance, tortuosity over the trained azimuths
    plus optional held-out azimuths (generalization)."""
    nca = make_chemo_nca(params)
    env = FlyEnv(antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral)
    trained = list(cfg.azimuths_deg)
    held_out = [a for a in extra_azimuths if a not in trained]

    def run_set(azis):
        rows = []
        for az in azis:
            src = source_for_azimuth(cfg.source_distance, az)
            ep = _episode(nca, cfg, src, env)
            rows.append(
                {
                    "azimuth_deg": float(az),
                    "source_xy": [float(v) for v in src],
                    "reached": bool(ep["reached"]),
                    "min_dist": ep["min_dist"],
                    "approach": ep["approach"],
                    "tortuosity": ep["tortuosity"],
                    "dyaw_final": ep["dyaw_final"],
                }
            )
        return rows

    trained_rows = run_set(trained)
    held_rows = run_set(held_out)

    def agg(rows):
        if not rows:
            return {}
        finite_tort = [r["tortuosity"] for r in rows if np.isfinite(r["tortuosity"])]
        return {
            "success_rate": float(np.mean([r["reached"] for r in rows])),
            "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
            "mean_approach": float(np.mean([r["approach"] for r in rows])),
            "mean_tortuosity": float(np.mean(finite_tort)) if finite_tort else float("inf"),
        }

    return {
        "config": {
            "source_distance": cfg.source_distance,
            "odor_lambda": cfg.odor_lambda,
            "reach_radius": cfg.reach_radius,
            "antenna_forward": cfg.antenna_forward,
            "antenna_lateral": cfg.antenna_lateral,
            "azimuths_deg": list(cfg.azimuths_deg),
            "fitness_mode": cfg.fitness_mode,
            "rollout_steps": cfg.rollout_steps,
        },
        "trained": {"per_azimuth": trained_rows, "aggregate": agg(trained_rows)},
        "held_out": {"per_azimuth": held_rows, "aggregate": agg(held_rows)},
    }


def build_trajectories(params: np.ndarray, cfg: ChemoConfig, azimuths) -> dict:
    """Fly path + source per episode, decimated for a top-down web viz."""
    nca = make_chemo_nca(params)
    env = FlyEnv(antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral)
    episodes = []
    for az in azimuths:
        src = source_for_azimuth(cfg.source_distance, az)
        ep = _episode(nca, cfg, src, env)
        path = ep["path"]
        step = max(1, path.shape[0] // 200)  # ~<=200 points per path
        episodes.append(
            {
                "azimuth_deg": float(az),
                "source_xy": [float(v) for v in src],
                "reach_radius": cfg.reach_radius,
                "reached": bool(ep["reached"]),
                "min_dist": round(ep["min_dist"], 4),
                "tortuosity": (
                    round(ep["tortuosity"], 4) if np.isfinite(ep["tortuosity"]) else None
                ),
                "path_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in path[::step]],
            }
        )
    return {
        "odor_field": {
            "formula": "C(p) = exp(-||p - source|| / lambda)",
            "lambda": cfg.odor_lambda,
            "source_distance": cfg.source_distance,
        },
        "episodes": episodes,
    }


# --------------------------------------------------------------------------- #
# Clip rendering
# --------------------------------------------------------------------------- #
def render_approach_clip(
    params: np.ndarray,
    cfg: ChemoConfig,
    source_xy,
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
        antenna_forward=cfg.antenna_forward,
        antenna_lateral=cfg.antenna_lateral,
    )
    nca = make_chemo_nca(params)
    env.set_odor(OdorField(source_xy=source_xy, lam=cfg.odor_lambda))
    _, traj = env.rollout(
        _make_chemo_policy(nca), n_steps=cfg.rollout_steps, pass_sensors=True
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env.sim.renderer.save_video(out_path)
    path = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    dist = np.linalg.norm(path - np.asarray(source_xy)[None, :], axis=1)
    return {
        "source_xy": [float(v) for v in source_xy],
        "min_dist": float(dist.min()),
        "dyaw_final": float(np.asarray(traj["yaw"])[-1] - traj["yaw0"]),
    }


# --------------------------------------------------------------------------- #
# Controller JSON
# --------------------------------------------------------------------------- #
def write_controller_json(
    params: np.ndarray,
    run_id: str,
    cfg: ChemoConfig,
    out_path: Path,
    best_fit: float | None = None,
) -> None:
    nca = make_chemo_nca(params)
    weights = {
        name: p.detach().cpu().numpy().round(6).tolist()
        for name, p in nca.named_parameters()
    }
    payload = {
        "meta": {
            "run_id": run_id,
            "task": "CH-A chemotaxis (bilateral gradient -> emergent steering)",
            "n_params": int(nca.n_params),
            "channels": CHANNELS,
            "proprio_channels": SENSOR_CHANNELS,
            "chemo_channels": CHEMO_CHANNELS,
            "hidden": HIDDEN_DEFAULT,
            "grid": [GRID_H, GRID_W],
            "gain": nca.gain,
            "best_fitness": best_fit,
            "warm_start": "closed_loop_controller.json (C2-A); chemo channels zero-init",
            "architecture": (
                "conv1: Conv2d(4 state + 2 proprio + 2 chemo = 8 -> 16, 3x3, "
                "zero-pad) -> tanh(gain*.) -> conv2: Conv2d(16 -> 4, 1x1) -> "
                "clamp[-1,1]. State stays 4 channels; proprio+chemo enter only at "
                "conv1's input. Chemo-zeroed == closed-loop dynamics (A/B exact)."
            ),
            "design_choices_and_caveats": DESIGN_CAVEATS,
        },
        "weights": weights,
        "flat_params": np.asarray(params, dtype=np.float64).round(6).tolist(),
        "sensors": {
            "description": (
                "Live proprioception (2 channels) and a bilateral odor reading "
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
                    "name": "odor_left",
                    "count": 1,
                    "normalization": "C in (0,1] from the field below; already normalized",
                    "placement": (
                        f"chemo ch0, left half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols 0-{CHEMO_COL_SPLIT - 1})"
                    ),
                    "source": "concentration at the LEFT antenna",
                },
                {
                    "input_channel_index": CHANNELS + SENSOR_CHANNELS + 1,
                    "name": "odor_right",
                    "count": 1,
                    "normalization": "C in (0,1] from the field below; already normalized",
                    "placement": (
                        f"chemo ch1, right half of motor block "
                        f"(rows 0-{MOTOR_ROWS - 1}, cols {CHEMO_COL_SPLIT}-{MOTOR_COLS - 1})"
                    ),
                    "source": "concentration at the RIGHT antenna",
                },
            ],
            "odor_field": {
                "formula": "C(p) = exp(-||p - source|| / lambda)  on the flat ground (x,y)",
                "lambda": cfg.odor_lambda,
                "note": "C = 1 at the source, decays with length lambda; values in (0,1].",
            },
            "antenna_geometry": {
                "frame": "body pose: forward = (cos yaw, sin yaw), left = (-sin yaw, cos yaw)",
                "forward_offset": cfg.antenna_forward,
                "lateral_offset": cfg.antenna_lateral,
                "left_antenna": "thorax_xy + forward*fwd + lateral*left",
                "right_antenna": "thorax_xy + forward*fwd - lateral*left",
                "caveat": DESIGN_CAVEATS["antenna_lateral_baseline"],
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))


def write_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))
