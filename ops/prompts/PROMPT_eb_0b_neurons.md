# Claude Code — EB-0B · Resolve LC4/LPLC2→DNp01 in the brain + a brain-only looming→giant-fiber check

> ## SAFETY (read first; bypassPermissions)
> Additive — adds to the `brain/` module; don't break EB-0A's API or any prior behavior. No `rm -rf` of
> anything you didn't create; scratch in `scratch/eb/`. `ops/` off-limits. **Commit on your branch
> before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **Runs on the MAC** (brain-only, laptop-runnable — **no sentry**). On branch `feat/eb-neurons`,
> **AFTER EB-0A is merged** (you use its `BrainModel` API). Use a todo list. **Read first:**
> `../portfolio/docs/cellular-gaits/EMBODIED_BRAIN_PLAN.md`, EB-0A's `brain_explainer.md` + API, and the
> CX-1 connectome artifact if present (`src/cellular_gaits/connectome/lc_dnp01_subcircuit.json` —
> from the `feat/cx-connectome` branch; if it's not on your base, re-resolve the IDs from the FlyWire
> annotations as CX-1 did). **Understanding-first:** include an explainer note (report material).

## Context
This bridges the **escape circuit** to the running brain. CX-1 extracted the real **LC4/LPLC2 → DNp01**
(giant fiber) sub-circuit. Now confirm those neurons live in the brain's v783 neuron list and that the
**real connectome reproduces the convergence** — the brain half of the escape loop, validated in
isolation, before any body.

## Your slice — own `src/cellular_gaits/brain/neurons.py`
1. **Cell-type → FlyWire ID mapping** in the brain's v783 neuron list: **LC4**, **LPLC2**, **DNp01**
   (a.k.a. Giant Fiber). Confirm the CX-1 IDs resolve in v783 (reconcile any release/version diffs).
   Build a small helper returning the ID sets per type and hemisphere (L/R), with provenance.
2. **Brain-only circuit check (NO body):** using EB-0A's `BrainModel`, **activate LC4 + LPLC2** (the
   looming input pathway) and confirm **DNp01 firing rate rises**. Sweep the input activation Hz →
   record the **DNp01 response curve** (input Hz → giant-fiber rate). Optionally confirm specificity
   (silencing LC4/LPLC2 removes the DNp01 response).
3. **Report** → `scratch/eb/neurons_report.md`: the ID mapping (counts per type/hemisphere), the DNp01
   response curve, how it squares with the biology (the LC4/LPLC2→GF convergence, Ache 2019), and the
   honest caveats (whole-brain context vs isolated; activation-Hz is a modeling choice). Explainer-grade.

## Contract (so EB-1 can call you)
Expose: the neuron-ID sets (`lc4_ids`, `lplc2_ids`, `dnp01_ids` by hemisphere), and ideally a helper
`looming_to_giant_fiber(brain, lc4_hz, lplc2_hz) -> dnp01_rate` that EB-1 uses as the brain step.

## Definition of done
- `src/cellular_gaits/brain/neurons.py` (ID mapping + the looming→GF helper).
- The brain-only looming→DNp01 demo runs; the response curve is reported.
- `scratch/eb/neurons_report.md` written.
- **Committed on `feat/eb-neurons`** (not merged). STOP and report.
