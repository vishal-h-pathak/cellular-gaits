"""Build the X-A web-export bundle from an evolved checkpoint.

    uv run python scripts/export_escape.py --ckpt checkpoints/xa_full/gen_50

Writes to outputs/web_data_x/:
    escape_controller.json, flee_left.mp4, flee_right.mp4,
    trajectories.json, escape_metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.escape import (  # noqa: E402
    WEB_DIR,
    best_from_checkpoint,
    build_trajectories,
    cfg_from_checkpoint,
    compute_metrics,
    render_flee_clip,
    write_controller_json,
    write_json,
)

# Held-out azimuths for the generalization check / extra trajectory episodes.
HELD_OUT_AZIMUTHS = (45.0, 315.0, 135.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="checkpoint stem, e.g. checkpoints/xa_full/gen_50")
    p.add_argument("--left-azimuth", type=float, default=90.0)
    p.add_argument("--right-azimuth", type=float, default=270.0)
    p.add_argument("--no-clips", action="store_true", help="skip mp4 rendering (faster)")
    args = p.parse_args()

    params, best_fit, run_id = best_from_checkpoint(args.ckpt)
    cfg = cfg_from_checkpoint(args.ckpt)
    print(f"[export] run_id={run_id} best_fit={best_fit:.4f} cfg: speed={cfg.threat_speed} "
          f"lead={cfg.threat_lead_distance} hit_r={cfg.threat_hit_radius} "
          f"gain={cfg.loom_input_gain} azimuths={cfg.azimuths_deg}")

    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # Controller JSON
    cjson = WEB_DIR / "escape_controller.json"
    write_controller_json(params, run_id, cfg, cjson, best_fit=best_fit)
    print(f"[export] controller -> {cjson.relative_to(ROOT)} ({cjson.stat().st_size/1024:.1f} KB)")

    # Metrics (trained azimuths + held-out generalization)
    metrics = compute_metrics(params, cfg, extra_azimuths=HELD_OUT_AZIMUTHS)
    write_json(metrics, WEB_DIR / "escape_metrics.json")
    tr = metrics["trained"]["aggregate"]
    print(f"[export] metrics: trained escape_success={tr['escape_success_rate']:.2f} "
          f"mean_min_dist={tr['mean_min_dist']:.2f} "
          f"mean_latency={tr['mean_reaction_latency_s']}")
    for r in metrics["trained"]["per_azimuth"]:
        print(f"           az={r['azimuth_deg']:>5.0f} escaped={r['escaped']!s:>5} "
              f"min_dist={r['min_dist']:>5.2f} awayΔ={r['total_away_turn']:>+6.2f} "
              f"latency={r['reaction_latency_s']}")

    # Trajectories (trained + held-out episodes)
    traj_azis = list(cfg.azimuths_deg) + list(HELD_OUT_AZIMUTHS)
    trajs = build_trajectories(params, cfg, traj_azis)
    write_json(trajs, WEB_DIR / "trajectories.json")
    print(f"[export] trajectories -> {len(trajs['episodes'])} episodes")

    # Clips: threat on the left and on the right, same controller -> opposite turns.
    if not args.no_clips:
        for side, az, name in (
            ("left", args.left_azimuth, "flee_left.mp4"),
            ("right", args.right_azimuth, "flee_right.mp4"),
        ):
            info = render_flee_clip(params, cfg, az, WEB_DIR / name)
            mb = (WEB_DIR / name).stat().st_size / 1e6
            print(f"[export] {name}: az={info['azimuth_deg']:.0f} escaped={info['escaped']} "
                  f"min_dist={info['min_dist']:.2f} awayΔ={info['total_away_turn']:+.2f}  ({mb:.2f} MB)")

    print("[export] done.")


if __name__ == "__main__":
    main()
