"""Transform the v1 CMA-ES fitness log into the web evolution curve.

Pure CSV -> JSON; no re-run. Reads
``outputs/fitness_log_2026-05-02T00-01-51Z.csv`` and emits
``outputs/web_data/evolution.json``:

    { "meta": {...},
      "curve": [{step, gen, best, mean, std, phase}, ...] }

`gen` is the generation number as logged. The run has two phases: the
`original` run (gens 0-37) plateaued, then a warm-started `resumed` run
re-numbered from gen 35 and broke through to best_fit ~86.6 (the
premature-convergence-then-resume story). Because both phases reuse gen
numbers, a monotonic `step` index is added so the curve plots without
folding back on itself.

Run from repo root:

    uv run python scripts/build_evolution_json.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "outputs" / "fitness_log_2026-05-02T00-01-51Z.csv"
OUT_PATH = ROOT / "outputs" / "web_data" / "evolution.json"


def main() -> None:
    with CSV_PATH.open(newline="") as fp:
        rows = list(csv.DictReader(fp))

    curve = []
    for step, r in enumerate(rows):
        curve.append(
            {
                "step": step,
                "gen": int(r["gen"]),
                "best": round(float(r["best_fit"]), 4),
                "mean": round(float(r["mean_fit"]), 4),
                "std": round(float(r["std_fit"]), 4),
                "phase": r["phase"],
            }
        )

    phases = {}
    for c in curve:
        p = phases.setdefault(c["phase"], {"steps": [], "best": []})
        p["steps"].append(c["step"])
        p["best"].append(c["best"])

    meta = {
        "run_id": "2026-05-02T00-01-51Z",
        "source_csv": CSV_PATH.name,
        "n_points": len(curve),
        "best_fit_overall": max(c["best"] for c in curve),
        "fitness_units": "mm (forward distance) - 0.05 * n_below",
        "phases": {
            name: {
                "step_start": p["steps"][0],
                "step_end": p["steps"][-1],
                "best_at_phase_end": p["best"][-1],
                "best_in_phase": max(p["best"]),
            }
            for name, p in phases.items()
        },
        "note": (
            "Phase 'resumed' warm-started from the gen-35 checkpoint of "
            "'original' and re-uses gen numbers; use 'step' for the x-axis."
        ),
    }

    payload = {"meta": meta, "curve": curve}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[evolution] wrote {OUT_PATH}  ({len(curve)} points)")
    for name, p in meta["phases"].items():
        print(
            f"[evolution] {name:<9} steps {p['step_start']}-{p['step_end']}  "
            f"best_in_phase={p['best_in_phase']:.3f}"
        )


if __name__ == "__main__":
    main()
