"""Build the CH-A web-export bundle from an evolved checkpoint.

    uv run python scripts/export_chemotaxis.py --ckpt checkpoints/cha_full/gen_50

Writes to outputs/web_data_ch/:
    chemotaxis_controller.json, approach_left.mp4, approach_right.mp4,
    trajectories.json, chemotaxis_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.chemotaxis import (  # noqa: E402
    WEB_DIR,
    best_from_checkpoint,
    build_trajectories,
    cfg_from_checkpoint,
    compute_metrics,
    render_approach_clip,
    write_controller_json,
    write_json,
)
from cellular_gaits.evolve_chemotaxis import source_for_azimuth  # noqa: E402

# Held-out azimuths for the generalization check / extra trajectory episodes.
HELD_OUT_AZIMUTHS = (45.0, 135.0, 315.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="checkpoint stem, e.g. checkpoints/cha_full/gen_50")
    p.add_argument("--left-azimuth", type=float, default=90.0)
    p.add_argument("--right-azimuth", type=float, default=270.0)
    p.add_argument("--no-clips", action="store_true", help="skip mp4 rendering (faster)")
    args = p.parse_args()

    params, best_fit, run_id = best_from_checkpoint(args.ckpt)
    cfg = cfg_from_checkpoint(args.ckpt)
    print(f"[export] run_id={run_id} best_fit={best_fit:.4f} cfg: dist={cfg.source_distance} "
          f"lam={cfg.odor_lambda} lateral={cfg.antenna_lateral} azimuths={cfg.azimuths_deg}")

    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # Controller JSON
    cjson = WEB_DIR / "chemotaxis_controller.json"
    write_controller_json(params, run_id, cfg, cjson, best_fit=best_fit)
    print(f"[export] controller -> {cjson.relative_to(ROOT)} ({cjson.stat().st_size/1024:.1f} KB)")

    # Metrics (trained azimuths + held-out generalization)
    metrics = compute_metrics(params, cfg, extra_azimuths=HELD_OUT_AZIMUTHS)
    write_json(metrics, WEB_DIR / "chemotaxis_metrics.json")
    tr = metrics["trained"]["aggregate"]
    print(f"[export] metrics: trained success={tr['success_rate']:.2f} "
          f"mean_min_dist={tr['mean_min_dist']:.2f} mean_tort={tr['mean_tortuosity']:.2f}")
    for r in metrics["trained"]["per_azimuth"]:
        print(f"           az={r['azimuth_deg']:>5.0f} reached={r['reached']!s:>5} "
              f"min_dist={r['min_dist']:>5.2f} dyaw={r['dyaw_final']:>+6.2f}")

    # Trajectories (trained + held-out episodes)
    traj_azis = list(cfg.azimuths_deg) + list(HELD_OUT_AZIMUTHS)
    trajs = build_trajectories(params, cfg, traj_azis)
    write_json(trajs, WEB_DIR / "trajectories.json")
    print(f"[export] trajectories -> {len(trajs['episodes'])} episodes")

    # Clips: source on the left and on the right, same controller.
    if not args.no_clips:
        for side, az, name in (
            ("left", args.left_azimuth, "approach_left.mp4"),
            ("right", args.right_azimuth, "approach_right.mp4"),
        ):
            src = source_for_azimuth(cfg.source_distance, az)
            info = render_approach_clip(params, cfg, src, WEB_DIR / name)
            mb = (WEB_DIR / name).stat().st_size / 1e6
            print(f"[export] {name}: src={info['source_xy']} min_dist={info['min_dist']:.2f} "
                  f"dyaw={info['dyaw_final']:+.2f}  ({mb:.2f} MB)")

    print("[export] done.")


if __name__ == "__main__":
    main()
