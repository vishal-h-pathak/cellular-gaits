"""EB-0A validation: reproduce the known sugar-GRN -> feeding-circuit result.

Activating sugar gustatory receptor neurons (GRNs) should drive the feeding
pathway, read out at the feeding/proboscis motor neuron MN9. We show the causal
effect with a negative control (no activation -> MN9 silent) vs. activation
(MN9 fires), averaged over a few trials. This is the proof the connectome brain
is wired correctly.

Run:  uv run python scratch/eb/validate_feeding.py
"""

import json
import time
from pathlib import Path

import numpy as np

from cellular_gaits.brain import BrainModel

# Sugar GRNs from the upstream example.ipynb (these are 630-materialization IDs;
# 20 of 21 still resolve in v783 -- FlyWire re-IDs neurons between releases).
NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265  # feeding/proboscis motor neuron (upstream readout)

T_RUN = 1.0      # seconds per trial
N_TRIAL = 3      # trials for the activation condition (Poisson is stochastic)
HZ = 150         # activation rate


def mn9_rate(rates):
    return rates.get(MN9, 0.0)


def main():
    brain = BrainModel.load("783")
    sugar = [s for s in NEU_SUGAR if brain.has(s)]
    print(f"# Brain: {brain.n_neurons} neurons (v783)")
    print(f"# Sugar GRNs resolving in v783: {len(sugar)}/{len(NEU_SUGAR)} "
          f"(dropped: {[s for s in NEU_SUGAR if not brain.has(s)]})")
    print(f"# MN9 ({MN9}) resolves: {brain.has(MN9)}\n")

    # --- negative control: no activation -> MN9 should be silent ---
    t0 = time.time()
    base = brain.run(T_RUN, query=[MN9], rebuild=True)
    print(f"[control] no activation, {T_RUN}s: "
          f"MN9 = {mn9_rate(base):.1f} Hz, "
          f"{len(base)-1} other neurons fired  ({time.time()-t0:.1f}s)")

    # --- activation: sugar GRNs -> feeding circuit ---
    brain.clear()
    brain.activate(sugar, hz=HZ)
    mn9_trials = []
    last = None
    for k in range(N_TRIAL):
        t0 = time.time()
        rates = brain.run(T_RUN, query=[MN9], rebuild=True)
        mn9_trials.append(mn9_rate(rates))
        last = rates
        print(f"[activate] trial {k+1}/{N_TRIAL}: MN9 = {mn9_rate(rates):.1f} Hz, "
              f"{len(rates)} neurons active  ({time.time()-t0:.1f}s)")

    mn9_mean, mn9_std = float(np.mean(mn9_trials)), float(np.std(mn9_trials))
    print(f"\n>>> MN9 firing rate: control {mn9_rate(base):.1f} Hz  ->  "
          f"activated {mn9_mean:.1f} +/- {mn9_std:.1f} Hz "
          f"(n={N_TRIAL} trials, {HZ} Hz drive)")

    # downstream feeding-circuit neurons: top firers that are NOT the driven GRNs
    sugar_set = set(sugar)
    downstream = sorted(((r, fid) for fid, r in last.items()
                         if fid not in sugar_set and fid != MN9),
                        reverse=True)[:10]
    print("\nTop downstream (non-GRN) neurons in the last activation trial:")
    for r, fid in downstream:
        print(f"  {fid}  {r:6.1f} Hz")

    out = {
        "release": "783",
        "n_neurons": brain.n_neurons,
        "n_sugar_resolved": len(sugar),
        "n_sugar_total": len(NEU_SUGAR),
        "hz_drive": HZ,
        "t_run_s": T_RUN,
        "n_trials": N_TRIAL,
        "mn9_control_hz": mn9_rate(base),
        "mn9_activated_mean_hz": mn9_mean,
        "mn9_activated_std_hz": mn9_std,
        "mn9_trials_hz": mn9_trials,
        "n_neurons_active_control": len(base) - 1,
        "n_neurons_active_activated": len(last),
        "top_downstream": [{"flywire_id": fid, "rate_hz": r} for r, fid in downstream],
    }
    p = Path(__file__).resolve().parent / "validation_result.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
