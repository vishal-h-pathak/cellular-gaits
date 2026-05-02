"""Render a checkpoint: deterministic rollout -> mp4 + CA state JSON.

The mp4 is written via MuJoCo's offscreen renderer through Flygym's
tracking camera. The JSON log is consumed by the React widget; its
schema lives in viz/state_log.py and must remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .env import DEFAULT_N_STEPS, FLY_NAME, FlyEnv
from .evolve import latest_checkpoint
from .nca import NCA
from .viz.state_log import state_to_frame, write_state_log

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"
VIDEO_DIR = OUT_DIR / "videos"


@dataclass
class RenderResult:
    fitness: float
    forward_dx: float
    n_below: int
    n_steps: int
    video_path: Path
    state_log_path: Path


def render_checkpoint(
    ckpt_path: Path,
    out_dir: Path = OUT_DIR,
    seed_for_state: int = 0,
    n_steps: int = DEFAULT_N_STEPS,
    name: str = "best",
    camera_res: tuple[int, int] = (240, 320),
    playback_speed: float = 0.2,
    output_fps: int = 25,
) -> RenderResult:
    data = np.load(ckpt_path, allow_pickle=False)
    best_params = np.asarray(data["best_params"], dtype=np.float64)
    run_id = str(data["run_id"])

    nca = NCA()
    nca.set_params(best_params)

    env = FlyEnv(
        renderer_camera=f"{FLY_NAME}/trackingcam",
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
    )

    state = NCA.init_state(seed=seed_for_state)
    frames_log: list[list[list[float]]] = []

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)
        frames_log.append(state_to_frame(state))
        return nca.motor_targets(state)

    fitness, traj = env.rollout(policy, n_steps=n_steps)

    video_path = out_dir / "videos" / f"{name}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    env.sim.renderer.save_video(video_path)

    state_log_path = out_dir / f"ca_states_{name}.json"
    write_state_log(state_log_path, frames_log, run_id=run_id)

    return RenderResult(
        fitness=float(fitness),
        forward_dx=float(traj["forward_dx"]),
        n_below=int(traj["n_below"]),
        n_steps=int(traj["n_steps"]),
        video_path=video_path,
        state_log_path=state_log_path,
    )


def render_latest(name: str = "best", **kwargs) -> RenderResult:
    return render_checkpoint(latest_checkpoint(), name=name, **kwargs)
