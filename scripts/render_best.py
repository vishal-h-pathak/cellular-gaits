"""Render the latest checkpoint to outputs/videos/<name>.mp4 + outputs/ca_states_<name>.json.

Examples:
    # Render best from the most recent run
    uv run python scripts/render_best.py

    # Render a specific checkpoint
    uv run python scripts/render_best.py --ckpt checkpoints/2026-05-01T23-58-23Z/gen_05.npz

    # Override output name
    uv run python scripts/render_best.py --name gen10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.evolve import latest_checkpoint
from cellular_gaits.render import render_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None, help="Path to gen_NN.npz; default = latest")
    p.add_argument("--name", type=str, default="best", help="Output basename (mp4 + JSON)")
    p.add_argument("--seed", type=int, default=0, help="Seed for NCA initial state")
    p.add_argument("--steps", type=int, default=None, help="Override n_steps (default = 750)")
    args = p.parse_args()

    ckpt = args.ckpt if args.ckpt is not None else latest_checkpoint()
    print(f"[render] checkpoint = {ckpt}")
    kwargs = {"name": args.name, "seed_for_state": args.seed}
    if args.steps is not None:
        kwargs["n_steps"] = args.steps

    result = render_checkpoint(ckpt, **kwargs)
    print(
        f"[render] fitness={result.fitness:.4f}  "
        f"forward_dx={result.forward_dx:.4f}  "
        f"n_below={result.n_below}/{result.n_steps}\n"
        f"[render] mp4  -> {result.video_path}\n"
        f"[render] json -> {result.state_log_path}"
    )


if __name__ == "__main__":
    main()
