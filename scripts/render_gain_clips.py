"""Render three web clips of the v1 best controller at low/native/high gain.

Shows the fly's behavior degrade as the controller is pushed off its
operating point: at native gain it walks; below it the gait is sluggish,
above it the CA crosses into chaos and the fly stops making progress.

Outputs (re-encoded to browser-friendly H.264 / yuv420p, < ~3 MB each):

    outputs/web_data/clip_gain_lo.mp4      (gain 0.5)
    outputs/web_data/clip_gain_native.mp4  (gain 1.0)
    outputs/web_data/clip_gain_hi.mp4      (gain 2.2)

Run from repo root:

    uv run python scripts/render_gain_clips.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import imageio_ffmpeg

from cellular_gaits.env import DEFAULT_N_STEPS, FLY_NAME, FlyEnv
from cellular_gaits.nca import NCA

CKPT = ROOT / "checkpoints" / "2026-05-02T00-01-51Z" / "gen_50.npz"
OUT_DIR = ROOT / "outputs" / "web_data"
CA_SEED = 0
CLIPS = [("lo", 0.5), ("native", 1.0), ("hi", 2.2)]

CAMERA_RES = (240, 320)
PLAYBACK_SPEED = 0.3  # ~0.3x slow-mo -> 10 s clip from a 3 s rollout
OUTPUT_FPS = 25


def _encode_web(src: Path, dst: Path) -> None:
    """Re-encode to H.264 + yuv420p (browser-safe, small) via imageio-ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
            "-preset", "veryfast", "-movflags", "+faststart",
            # pad to even dimensions (libx264 requirement)
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def render_clip(label: str, gain: float, best_params: np.ndarray) -> None:
    nca = NCA(gain=gain)
    nca.set_params(best_params)
    env = FlyEnv(
        renderer_camera=f"{FLY_NAME}/trackingcam",
        camera_res=CAMERA_RES,
        playback_speed=PLAYBACK_SPEED,
        output_fps=OUTPUT_FPS,
    )
    state = NCA.init_state(seed=CA_SEED)

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)
        return nca.motor_targets(state)

    fitness, traj = env.rollout(policy, n_steps=DEFAULT_N_STEPS)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        env.sim.renderer.save_video(raw)
        dst = OUT_DIR / f"clip_gain_{label}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        _encode_web(raw, dst)
    size_mb = dst.stat().st_size / 1e6
    print(
        f"[clip] {label:<6} gain={gain}  F={fitness:7.3f}  dist={traj['forward_dx']:7.3f}mm"
        f"  -> {dst.name} ({size_mb:.2f} MB)"
    )


def main() -> None:
    data = np.load(CKPT, allow_pickle=False)
    best_params = np.asarray(data["best_params"], dtype=np.float64)
    print(f"[clip] checkpoint={CKPT.name}")
    for label, gain in CLIPS:
        render_clip(label, gain, best_params)


if __name__ == "__main__":
    main()
