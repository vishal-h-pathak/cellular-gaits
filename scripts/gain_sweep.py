"""Gain -> gait sweep on the v1 best controller (the Stage-1 headline).

For each gain, run a full MuJoCo rollout of the v1 best NCA and record both
sides of the criticality story:

  - gait quality:   fitness F, forward distance (mm), n_below (stability)
  - CA criticality: poor-man's Lyapunov lambda + state-change rate

Writes a single web-friendly JSON the page scrubs:

    outputs/web_data/gain_sweep.json
      { "meta": {...}, "sweep": [{gain, fitness, distance_mm, n_below,
                                  lambda, change_rate}, ...] }

Run from repo root:

    uv run python scripts/gain_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cellular_gaits.criticality import (
    DEFAULT_EPS,
    DEFAULT_WARMUP,
    measure_criticality,
)
from cellular_gaits.env import DEFAULT_N_STEPS, FlyEnv
from cellular_gaits.nca import NCA

GAINS = [0.25, 0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 3.0, 4.0]
CKPT = ROOT / "checkpoints" / "2026-05-02T00-01-51Z" / "gen_50.npz"
OUT_PATH = ROOT / "outputs" / "web_data" / "gain_sweep.json"
CA_SEED = 0  # matches render.py seed_for_state -> same trajectory drives legs


def _rollout_fitness(env: FlyEnv, nca: NCA) -> dict:
    """Full MuJoCo rollout of the (gain-configured) NCA; gait metrics only."""
    state = NCA.init_state(seed=CA_SEED)

    def policy(t: int) -> np.ndarray:
        nonlocal state
        with torch.no_grad():
            state = nca.step(state)
        return nca.motor_targets(state)

    fitness, traj = env.rollout(policy, n_steps=DEFAULT_N_STEPS)
    return {
        "fitness": float(fitness),
        "distance_mm": float(traj["forward_dx"]),
        "n_below": int(traj["n_below"]),
    }


def main() -> None:
    data = np.load(CKPT, allow_pickle=False)
    best_params = np.asarray(data["best_params"], dtype=np.float64)
    run_id = str(data["run_id"])
    best_fit = float(data["best_fit"])
    print(f"[sweep] checkpoint={CKPT.name}  run_id={run_id}  best_fit={best_fit:.3f}")

    env = FlyEnv()  # no renderer: rollouts only, reused across gains
    sweep = []
    for g in GAINS:
        nca = NCA(gain=g)
        nca.set_params(best_params)
        gait = _rollout_fitness(env, nca)
        crit = measure_criticality(nca, n_steps=DEFAULT_N_STEPS, seed=CA_SEED)
        row = {
            "gain": g,
            "fitness": round(gait["fitness"], 4),
            "distance_mm": round(gait["distance_mm"], 4),
            "n_below": gait["n_below"],
            "lambda": round(crit.lyapunov, 5),
            "change_rate": round(crit.change_rate, 5),
        }
        sweep.append(row)
        print(
            f"[sweep] gain={g:<5} F={row['fitness']:>9.3f}  "
            f"dist={row['distance_mm']:>8.3f}mm  n_below={row['n_below']:>4}  "
            f"lambda={row['lambda']:+.4f}  change_rate={row['change_rate']:.4f}"
        )

    finite = all(
        np.isfinite([r["fitness"], r["distance_mm"], r["lambda"], r["change_rate"]]).all()
        for r in sweep
    )
    assert finite, "non-finite value in sweep"

    payload = {
        "meta": {
            "run_id": run_id,
            "checkpoint": CKPT.name,
            "best_fit_native": best_fit,
            "n_steps": DEFAULT_N_STEPS,
            "ca_seed": CA_SEED,
            "lyapunov": {
                "method": "twin-trajectory, renormalized, warmup discarded",
                "warmup": DEFAULT_WARMUP,
                "eps": DEFAULT_EPS,
            },
            "fitness_formula": "forward_dx - 0.05 * n_below (3 s rollout)",
            "distance_units": "mm (Flygym world units)",
            "native_gain": 1.0,
        },
        "sweep": sweep,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[sweep] wrote {OUT_PATH}  ({len(sweep)} gains)")


if __name__ == "__main__":
    main()
