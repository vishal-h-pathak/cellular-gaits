"""Sweep the lateral-impulse magnitude on the v1 open-loop best.

Finds the fair operating point: the smallest magnitude at which the open-loop
controller clearly degrades (falls / fails to recover heading / large heading
error) on roughly half to two-thirds of the perturbation seeds.

    uv run python scripts/perturbation_sweep.py --mags 3 6 10 --seeds 101 202 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.compare import (  # noqa: E402
    WEB_DIR,
    DEGRADE_HEADING_DEG,
    pick_operating_magnitude,
    sweep_open_loop,
)
from cellular_gaits.evolve_closed_loop import _load_v1_params  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mags", type=float, nargs="+", default=[3.0, 6.0, 10.0])
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505, 606, 707, 808, 909],
    )
    p.add_argument("--n-steps", type=int, default=750)
    p.add_argument("--out", type=Path, default=WEB_DIR / "perturbation_sweep.json")
    args = p.parse_args()

    v1 = _load_v1_params()
    rows = sweep_open_loop(v1, args.mags, args.seeds, n_steps=args.n_steps)
    chosen = pick_operating_magnitude(rows)

    print(f"\nOpen-loop degradation sweep ({len(args.seeds)} seeds, "
          f"'degraded' = fell OR no-recover OR heading>{DEGRADE_HEADING_DEG:.0f}deg):\n")
    print(f"{'mag':>6} {'degraded':>9} {'fell':>6} {'noRecov':>8} "
          f"{'head_deg':>9} {'postDx':>8}")
    for r in rows:
        print(
            f"{r['magnitude']:>6.1f} {r['degraded_fraction']*100:>8.0f}% "
            f"{r['fell_fraction']*100:>5.0f}% {r['no_recover_fraction']*100:>7.0f}% "
            f"{r['heading_error_deg_mean']:>9.1f} {r['post_shove_dx_mean']:>8.2f}"
        )
    print(f"\nChosen operating magnitude: {chosen}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"rows": rows, "chosen_magnitude": chosen, "seeds": args.seeds,
             "degrade_heading_deg": DEGRADE_HEADING_DEG},
            indent=2,
        )
    )
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"sweep -> {shown}")


if __name__ == "__main__":
    main()
