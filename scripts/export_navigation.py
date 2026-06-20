"""Build the N-A web-export bundle from an evolved checkpoint.

    uv run python scripts/export_navigation.py --ckpt checkpoints/na-full/gen_70

Writes to outputs/web_data_n/:
    navigation_controller.json, detour_right.mp4, detour_left.mp4,
    trajectories.json, navigation_metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.evolve_navigation import conditions  # noqa: E402
from cellular_gaits.navigation import (  # noqa: E402
    WEB_DIR,
    best_from_checkpoint,
    build_trajectories,
    cfg_from_checkpoint,
    compute_metrics,
    render_detour_clip,
    write_controller_json,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="checkpoint stem, e.g. checkpoints/na-full/gen_70")
    p.add_argument("--no-clips", action="store_true", help="skip mp4 rendering (faster)")
    args = p.parse_args()

    params, best_fit, run_id = best_from_checkpoint(args.ckpt)
    cfg = cfg_from_checkpoint(args.ckpt)
    print(
        f"[export] run_id={run_id} best_fit={best_fit:.4f} cfg: dist={cfg.source_distance} "
        f"obs_r={cfg.obstacle_radius} feeler_range={cfg.feeler_range} "
        f"w_collide={cfg.w_collide} gain={cfg.feeler_input_gain} "
        f"conds={[c.name for c in conditions(cfg)]}"
    )

    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # Controller JSON
    cjson = WEB_DIR / "navigation_controller.json"
    write_controller_json(params, run_id, cfg, cjson, best_fit=best_fit)
    print(f"[export] controller -> {cjson.relative_to(ROOT)} ({cjson.stat().st_size/1024:.1f} KB)")

    # Metrics (trained conditions + held-out generalization)
    metrics = compute_metrics(params, cfg)
    write_json(metrics, WEB_DIR / "navigation_metrics.json")
    tr, ho = metrics["trained"]["aggregate"], metrics["held_out"]["aggregate"]
    print(
        f"[export] metrics TRAINED: detour_success={tr['detour_success_rate']:.2f} "
        f"reach={tr['reach_rate']:.2f} avoid={tr['avoid_rate']:.2f} "
        f"mean_collisions={tr['mean_collisions']:.1f}"
    )
    print(
        f"[export] metrics HELD-OUT: detour_success={ho['detour_success_rate']:.2f} "
        f"reach={ho['reach_rate']:.2f} avoid={ho['avoid_rate']:.2f} "
        f"mean_collisions={ho['mean_collisions']:.1f}"
    )
    for grp, key in (("TRAINED", "trained"), ("HELD-OUT", "held_out")):
        print(f"    --- {grp} ---")
        for r in metrics[key]["per_condition"]:
            print(
                f"    {r['name']:>22} reach={r['reached']!s:>5} coll={r['collision_count']:>4} "
                f"detour={r['detour_perp']:>+6.2f} correct={r['detour_correct']!s:>5} "
                f"success={r['detour_success']!s:>5}"
            )

    # Trajectories (trained + held-out + no-obstacle reference)
    trajs = build_trajectories(params, cfg)
    write_json(trajs, WEB_DIR / "trajectories.json")
    print(f"[export] trajectories -> {len(trajs['episodes'])} episodes")

    # Clips: obstacle on the body-LEFT (detour right) and body-RIGHT (detour left),
    # same controller -> opposite detours. Use the 'far' trained placements.
    if not args.no_clips:
        conds = conditions(cfg)
        left_block = next(c for c in conds if c.name == "g40_far_block_left")
        right_block = next(c for c in conds if c.name == "g40_far_block_right")
        for cond, name in ((left_block, "detour_right.mp4"), (right_block, "detour_left.mp4")):
            info = render_detour_clip(params, cfg, cond, WEB_DIR / name)
            mb = (WEB_DIR / name).stat().st_size / 1e6
            print(
                f"[export] {name}: {info['name']} reached={info['reached']} "
                f"collisions={info['collision_count']} detour_perp={info['detour_perp']:+.2f}  ({mb:.2f} MB)"
            )

    print("[export] done.")


if __name__ == "__main__":
    main()
