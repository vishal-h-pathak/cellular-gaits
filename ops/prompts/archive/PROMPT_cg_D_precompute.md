# Claude Code prompt — D · Precompute real rollouts & sweeps

> **Run from the `cellular-gaits` repo root** (uv project, Python 3.12, FlyGym 2.0.1).
> Wave 1 of the page redesign; independent, runs in parallel with A and C. Long-running
> (real MuJoCo rollouts). Produces the "recorded-real" data the web tabs scrub when live
> in-browser physics isn't enough. Use a todo list. Branch `feat/web-precompute`; don't
> merge. **Keep v1 intact — this is analysis on the existing best controller.**

## Context

The redesigned page runs short rollouts live in WASM, but the heavy results are
precomputed here in the real engine and shipped as JSON/mp4. The existing best controller
is `checkpoints/2026-05-02T00-01-51Z/gen_50.npz` (`best_fit` ≈ 86.6 mm). The exported
weights already live at `portfolio/public/cellular-gaits/controller_best.json`.

## What to produce (scope to what's buildable now)

1. **The gain knob (do this first).** Add a scalar `gain` to `src/cellular_gaits/nca.py`
   that multiplies the pre-activation input to `tanh` (default `1.0` = current behavior):
   `h = tanh(gain * (conv1(s) + b1))`. This is the dial the whole criticality story turns
   on. Add a unit check that `gain=1.0` reproduces the current rollout exactly.

2. **The gain → gait sweep (the real Stage 1 — the headline).** Take the v1 best
   controller and, for `gain ∈ {0.25, 0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 3.0, 4.0}`, run a
   full MuJoCo rollout and record per gain:
   - **gait quality:** forward distance, `n_below`, the existing fitness `F`.
   - **CA-side criticality:** state-change rate and the poor-man's Lyapunov λ
     (twin-trajectory, renormalized, warmup discarded — same method as the playground).
   Export a single `outputs/web_data/gain_sweep.json`:
   `[{gain, fitness, distance_mm, n_below, lambda, change_rate}, ...]` plus a `meta` block.
   This is the data that finally answers "do good gaits sit just inside the edge of
   chaos?" — report what you find in 3–5 sentences.

3. **Three rendered clips** at low / native / high gain (e.g. 0.5, 1.0, 2.2) showing the
   fly's behavior degrade as the controller is pushed off its operating point —
   `outputs/web_data/clip_gain_{lo,native,hi}.mp4`, web-friendly (H.264, small).

4. **The evolution curve.** From the existing `outputs/fitness_log_2026-05-02T00-01-51Z.csv`
   produce `outputs/web_data/evolution.json`:
   `[{gen, best, mean, std, phase}, ...]` — including the `original` vs `resumed` phase
   split (the premature-convergence story). No re-run needed; just transform the CSV.

## Explicitly deferred (do NOT attempt here — they need work that doesn't exist yet)

- Open- vs closed-loop **perturbation-recovery** data → needs the closed-loop architecture
  (Stage 2). Note it as a follow-up.
- A **CPG baseline** rollout → needs a CPG controller (not implemented). Note it.

Leave clear TODO markers in the report for these so the wave-3 prompts (E3 Sensing, E6
Optimizer) know what's available now vs pending.

## Verify

- `gain=1.0` rollout is bit-for-bit (or within float tol) the same as before the change.
- `gain_sweep.json` has all 9 gains, finite values, and λ that crosses sign between ~1.0
  and ~1.5 (sanity vs the CA-side analysis already done).
- Clips play and are < ~3 MB each.

## Output & handoff

- Everything under `outputs/web_data/`. These belong at
  `portfolio/public/cellular-gaits/data/` — you don't have portfolio access from here, so
  leave them and end the report with the exact copy command (or note for `/add-dir`).
- Update row **D** status in `portfolio/docs/cellular-gaits/build-plan.md` only if granted
  access; else note for the user.

## Deliverable

Branch `feat/web-precompute`: the gain knob + `outputs/web_data/` (gain_sweep.json,
3 clips, evolution.json) + a SHORT report including the criticality finding and the two
deferred TODOs. Then stop.
