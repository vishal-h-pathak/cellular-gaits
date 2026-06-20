"""Open- vs closed-loop perturbation comparison + web exports (C2-A).

Runs the v1 open-loop best and the new closed-loop best on *identical*
perturbation seeds and records, per controller and seed:

    upright_pct        fraction of post-shove control steps with thorax above
                       the fall threshold (z >= z_threshold)
    stayed_upright     final thorax z >= z_threshold (didn't end fallen)
    heading_error_rad  mean |yaw - yaw0| over the post-shove window
    post_shove_dx      forward x progress from the shove onset to the end
    recovery_time_s    time from onset until |yaw - yaw0| first returns below
                       RECOVER_TOL_RAD (inf if it never recovers)

Both controllers are run through the *same* 6-channel NCA class: the open-loop
one is warm-started from the v1 weights and run sensor-blind (zeroed sensor
channels), which reproduces v1 dynamics exactly (verified in tests). The
closed-loop one is run with live sensors. The perturbation is seeded, so both
see the identical shove.

Exports (to outputs/web_data_c2/):
    closed_loop_controller.json   weights + sensors spec
    robustness_metrics.json       per-controller aggregates across the seed set
    perturbation_openloop.mp4     same shove, open loop
    perturbation_closedloop.mp4   same shove, closed loop
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .env import CONTROL_DT_S, FlyEnv, Perturbation
from .evolve_closed_loop import _load_v1_params, _make_closed_policy, _pert_kwargs
from .nca import (
    CHANNELS,
    GRID_H,
    GRID_W,
    HIDDEN_DEFAULT,
    N_LEGS,
    N_MOTORS,
    NCA,
    SENSOR_CHANNELS,
    V1_N_PARAMS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "outputs" / "web_data_c2"

RECOVER_TOL_RAD = 0.15  # heading within ~8.6 deg of original counts as recovered


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def make_open_loop_policy(v1_params: np.ndarray, ca_seed: int = 0):
    """v1 open loop: 6-channel NCA warm-started from v1, run sensor-blind."""
    nca = NCA()
    nca.warm_start_from_v1(np.asarray(v1_params, dtype=np.float64)[:V1_N_PARAMS])
    state = NCA.init_state(seed=ca_seed)

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)  # sensors zeroed
        return nca.motor_targets(state)

    return policy, nca


def make_closed_loop_nca(cl_params: np.ndarray) -> NCA:
    nca = NCA()
    nca.set_params(np.asarray(cl_params, dtype=np.float64))
    return nca


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def metrics_from_traj(traj: dict) -> dict:
    yaw = np.asarray(traj["yaw"], dtype=np.float64)
    yaw0 = float(traj["yaw0"])
    z = np.asarray(traj["thorax_z"], dtype=np.float64)
    x = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, 0]
    z_thresh = float(traj["z_threshold"])
    onset = int(traj["impulse_onset_step"])
    n = yaw.size

    post = slice(onset + 1, n) if (0 <= onset < n - 1) else slice(0, n)
    yaw_err_post = np.abs(_wrap_pi(yaw[post] - yaw0))
    z_post = z[post]

    upright_pct = float(np.mean(z_post >= z_thresh)) if z_post.size else 1.0
    stayed_upright = bool(z[-1] >= z_thresh)
    heading_error_rad = float(np.mean(yaw_err_post)) if yaw_err_post.size else 0.0

    onset_x = float(x[onset]) if 0 <= onset < n else float(x[0])
    post_shove_dx = float(x[-1] - onset_x)

    # recovery time: first post-onset step where heading returns within tol.
    recovery_time_s = float("inf")
    if 0 <= onset < n - 1:
        rel = np.abs(_wrap_pi(yaw[onset + 1 :] - yaw0))
        below = np.where(rel < RECOVER_TOL_RAD)[0]
        if below.size:
            recovery_time_s = float((below[0] + 1) * CONTROL_DT_S)

    return {
        "upright_pct": upright_pct,
        "stayed_upright": stayed_upright,
        "heading_error_rad": heading_error_rad,
        "heading_error_deg": float(np.degrees(heading_error_rad)),
        "post_shove_dx": post_shove_dx,
        "recovery_time_s": recovery_time_s,
        "recovered": bool(np.isfinite(recovery_time_s)),
        "forward_dx": float(traj["forward_dx"]),
        "impulse_onset_step": onset,
    }


def _aggregate(per_seed: list[dict]) -> dict:
    def m(key):
        return float(np.mean([d[key] for d in per_seed]))

    finite_rec = [d["recovery_time_s"] for d in per_seed if d["recovered"]]
    return {
        "upright_pct": m("upright_pct"),
        "stayed_upright_rate": float(np.mean([d["stayed_upright"] for d in per_seed])),
        "heading_error_rad": m("heading_error_rad"),
        "heading_error_deg": m("heading_error_deg"),
        "post_shove_dx": m("post_shove_dx"),
        "forward_dx": m("forward_dx"),
        "recovered_rate": float(np.mean([d["recovered"] for d in per_seed])),
        "recovery_time_s_mean": (
            float(np.mean(finite_rec)) if finite_rec else float("inf")
        ),
        "n_seeds": len(per_seed),
    }


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass
class CompareConfig:
    eval_seeds: tuple[int, ...] = (101, 202, 303, 404, 505)
    n_steps: int = 750
    pert_magnitude: float = 3.0
    pert_window: tuple[float, float] = (0.4, 0.6)
    pert_duration_s: float = 0.05
    pert_randomize_sign: bool = True
    uneven_ground: bool = False
    terrain_seed: int = 0

    def pert_kwargs(self) -> dict:
        return {
            "magnitude": self.pert_magnitude,
            "window": self.pert_window,
            "duration_s": self.pert_duration_s,
            "randomize_sign": self.pert_randomize_sign,
        }


def run_comparison(
    v1_params: np.ndarray,
    cl_params: np.ndarray,
    cfg: CompareConfig,
) -> dict:
    pk = cfg.pert_kwargs()
    env = FlyEnv(uneven_ground=cfg.uneven_ground, terrain_seed=cfg.terrain_seed)
    cl_nca = make_closed_loop_nca(cl_params)

    open_seed_metrics: list[dict] = []
    closed_seed_metrics: list[dict] = []
    per_seed_rows = []
    for ps in cfg.eval_seeds:
        env.set_perturbation(Perturbation(seed=int(ps), **pk))
        open_pol, _ = make_open_loop_policy(v1_params)
        _, traj_o = env.rollout(open_pol, n_steps=cfg.n_steps)
        m_o = metrics_from_traj(traj_o)

        env.set_perturbation(Perturbation(seed=int(ps), **pk))
        _, traj_c = env.rollout(
            _make_closed_policy(cl_nca), n_steps=cfg.n_steps, pass_sensors=True
        )
        m_c = metrics_from_traj(traj_c)

        open_seed_metrics.append(m_o)
        closed_seed_metrics.append(m_c)
        per_seed_rows.append({"seed": int(ps), "open": m_o, "closed": m_c})

    return {
        "config": {
            "eval_seeds": list(cfg.eval_seeds),
            "n_steps": cfg.n_steps,
            "control_dt_s": CONTROL_DT_S,
            "pert_magnitude": cfg.pert_magnitude,
            "pert_window": list(cfg.pert_window),
            "pert_duration_s": cfg.pert_duration_s,
            "pert_randomize_sign": cfg.pert_randomize_sign,
            "uneven_ground": cfg.uneven_ground,
            "recover_tol_rad": RECOVER_TOL_RAD,
        },
        "open_loop": _aggregate(open_seed_metrics),
        "closed_loop": _aggregate(closed_seed_metrics),
        "per_seed": per_seed_rows,
    }


# --------------------------------------------------------------------------- #
# Rendering a single shove (one seed) for each controller
# --------------------------------------------------------------------------- #
def render_perturbation_clip(
    params: np.ndarray,
    closed_loop: bool,
    seed: int,
    cfg: CompareConfig,
    out_path: Path,
    camera_res: tuple[int, int] = (240, 320),
    playback_speed: float = 0.2,
    output_fps: int = 25,
) -> dict:
    from .env import FLY_NAME

    env = FlyEnv(
        renderer_camera=f"{FLY_NAME}/trackingcam",
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
        uneven_ground=cfg.uneven_ground,
        terrain_seed=cfg.terrain_seed,
    )
    env.set_perturbation(Perturbation(seed=int(seed), **cfg.pert_kwargs()))
    if closed_loop:
        nca = make_closed_loop_nca(params)
        _, traj = env.rollout(
            _make_closed_policy(nca), n_steps=cfg.n_steps, pass_sensors=True
        )
    else:
        pol, _ = make_open_loop_policy(params)
        _, traj = env.rollout(pol, n_steps=cfg.n_steps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env.sim.renderer.save_video(out_path)
    return metrics_from_traj(traj)


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def write_controller_json(
    cl_params: np.ndarray,
    run_id: str,
    out_path: Path,
    best_fit: float | None = None,
) -> None:
    nca = make_closed_loop_nca(cl_params)
    weights = {
        name: p.detach().cpu().numpy().round(6).tolist()
        for name, p in nca.named_parameters()
    }
    payload = {
        "meta": {
            "run_id": run_id,
            "n_params": int(nca.n_params),
            "channels": CHANNELS,
            "sensor_channels": SENSOR_CHANNELS,
            "hidden": HIDDEN_DEFAULT,
            "grid": [GRID_H, GRID_W],
            "gain": nca.gain,
            "control_dt_s": CONTROL_DT_S,
            "best_fitness": best_fit,
            "architecture": (
                "conv1: Conv2d(4+2 -> 16, 3x3, zero-pad) -> tanh(gain*.) -> "
                "conv2: Conv2d(16 -> 4, 1x1) -> clamp[-1,1]. State stays 4 "
                "channels; 2 sensor channels enter only at conv1's input."
            ),
        },
        "weights": weights,
        "flat_params": np.asarray(cl_params, dtype=np.float64).round(6).tolist(),
        "sensors": {
            "description": (
                "Live proprioception is read each control step and laid out "
                "topographically into 2 extra conv1 input channels. With these "
                "channels zeroed the controller reproduces the v1 open-loop "
                "dynamics exactly (A/B integrity)."
            ),
            "channels": [
                {
                    "input_channel_index": CHANNELS + 0,
                    "name": "joint_angles",
                    "count": N_MOTORS,
                    "normalization": "joint_angle_rad / 3.14, clipped to [-1, 1]",
                    "placement": "sensor-channel 0, 7x6 motor block (rows 0-6, cols 0-5)",
                    "source": "mj_data.actuator_length at the 42 position actuators",
                },
                {
                    "input_channel_index": CHANNELS + 1,
                    "name": "foot_contacts",
                    "count": N_LEGS,
                    "normalization": "boolean {0, 1}",
                    "placement": "sensor-channel 1, bottom row (row 7, cols 0-5)",
                    "source": "get_ground_contact_info()[0] per-leg booleans",
                },
            ],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))


def write_metrics_json(comparison: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2))


# --------------------------------------------------------------------------- #
# Perturbation-magnitude sweep on the v1 open-loop best
# --------------------------------------------------------------------------- #
DEGRADE_HEADING_DEG = 45.0  # "lost heading" if post-shove mean error exceeds this


def _seed_degraded(m: dict) -> bool:
    """Open-loop 'clearly degrades' on a seed: it falls, fails to recover its
    heading, or ends with a large mean post-shove heading error."""
    return (
        (not m["stayed_upright"])
        or (not m["recovered"])
        or (m["heading_error_deg"] > DEGRADE_HEADING_DEG)
    )


def sweep_open_loop(
    v1_params: np.ndarray,
    magnitudes,
    seeds,
    n_steps: int = 750,
    pert_window: tuple[float, float] = (0.4, 0.6),
    pert_duration_s: float = 0.05,
    pert_randomize_sign: bool = True,
) -> list[dict]:
    """For each magnitude, run the v1 open-loop best on the seed set and report
    how often it clearly degrades. Returns one row per magnitude."""
    env = FlyEnv()
    rows = []
    for mag in magnitudes:
        pk = {
            "magnitude": float(mag),
            "window": pert_window,
            "duration_s": pert_duration_s,
            "randomize_sign": pert_randomize_sign,
        }
        per_seed = []
        for ps in seeds:
            env.set_perturbation(Perturbation(seed=int(ps), **pk))
            pol, _ = make_open_loop_policy(v1_params)
            _, traj = env.rollout(pol, n_steps=n_steps)
            per_seed.append(metrics_from_traj(traj))
        degraded = [_seed_degraded(m) for m in per_seed]
        rows.append(
            {
                "magnitude": float(mag),
                "n_seeds": len(seeds),
                "degraded_fraction": float(np.mean(degraded)),
                "fell_fraction": float(np.mean([not m["stayed_upright"] for m in per_seed])),
                "no_recover_fraction": float(np.mean([not m["recovered"] for m in per_seed])),
                "heading_error_deg_mean": float(np.mean([m["heading_error_deg"] for m in per_seed])),
                "post_shove_dx_mean": float(np.mean([m["post_shove_dx"] for m in per_seed])),
            }
        )
    return rows


def pick_operating_magnitude(
    rows: list[dict], lo: float = 0.5, hi: float = 0.67
) -> float | None:
    """Smallest magnitude whose degraded_fraction is in [lo, hi]; else the
    smallest with degraded_fraction >= lo; else None."""
    in_band = [r for r in rows if lo <= r["degraded_fraction"] <= hi]
    if in_band:
        return min(in_band, key=lambda r: r["magnitude"])["magnitude"]
    above = [r for r in rows if r["degraded_fraction"] >= lo]
    if above:
        return min(above, key=lambda r: r["magnitude"])["magnitude"]
    return None
