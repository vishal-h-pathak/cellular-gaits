# Claude Code — CH-A · Chemotaxis: bilateral gradient sensor + reach-the-source

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create (`.worktrees/`, `AGENT_SAFETY.md`,
> `setup-*.sh`, `PROMPT_*.md`, `scratch/`, checkpoints/outputs you didn't make are off-limits).
> No `rm -rf` of anything you didn't create; no broad/glob deletes. Do scratch work in one dir
> you create (`scratch/cha/`) and remove only that. Keep v1 + the closed-loop code intact.
> When unsure, leave it and note it. (Full: `AGENT_SAFETY.md`)

> **Run from the `cellular-gaits` repo root.** Chemotaxis campaign, wave 1 — the compute job.
> Runs in parallel with CH-B (portfolio scaffold). Branch `feat/ch-chemotaxis`; don't merge.
> Use a todo list.

## Goal

Teach the fly to **walk to a food source** by sensing an odor/taste gradient. The headline
must be **emergent steering**: turning toward the source falls out of a left-vs-right sensor
asymmetry, not a hard-coded turn.

## 1. Bilateral chemosensor + odor field (`env.py`, `nca.py`)

- Add an **odor source** at a configurable `(x, y)` on the flat ground, with a smooth
  concentration field `C(p)` (e.g. `exp(-||p-src|| / λ)` or `1/(1+(d/λ)^2)` — pick one,
  document λ). 
- Sample it at **two antenna positions** (offset left/right of the head, from the body pose)
  → `cL`, `cR`. Feed normalized `cL`, `cR` into **two new chemo input channels** (bilateral —
  this asymmetry is the whole point). Place them topographically like the proprio channels.
- **Build on the closed-loop controller** (it already walks + has proprioception). Final
  conv1 input = 4 state + 2 proprio + 2 chemo = **8→16**. New chemo weights zero-init,
  **warm-start from `closed_loop_controller.json`** so it already walks and is robust; A/B
  integrity: chemo-zeroed == closed-loop behavior. Document the final channel layout in the
  exported `sensors` spec (incl. antenna geometry + the field formula).

## 2. Fitness — force generalization (so steering is emergent)

Reward approach: `F = (d_start − d_end) + reach_bonus·[d_end < r] − small time/again penalty`.
**Critical:** average each candidate over several **source azimuths (left / right / ahead /
behind) and start headings**, so a fixed turn bias can't win — the controller must turn based
on `cL` vs `cR`. This multiplies rollouts; **parallelize the CMA-ES population** (ProcessPool,
per-worker env, like C2-A). Warm-start, ~50 gens, checkpoint/resumable.

## 3. Validation-first (short run before the full one — stop and report)

Build everything, then a short run to confirm: warm-started gen 0 still walks; pipeline +
parallel eval work; and **calibrate** the field λ + source distances so the task is learnable
but non-trivial (source not so close it's reached by luck, not so far the gradient's flat).
Look for an early signal that it turns toward the source on both sides. **Report, then I
green-light the full run** (same gate we used for perturbation).

## 4. Exports → `outputs/web_data_ch/`

- `chemotaxis_controller.json` — weights + full `sensors` spec (channels, antenna offsets,
  field formula, normalization).
- approach clips showing **emergent turning both ways**: `approach_left.mp4`,
  `approach_right.mp4` (source on opposite sides, same controller).
- `trajectories.json` — fly path(s) + source position(s) over a few episodes (for a top-down
  viz on the page).
- `chemotaxis_metrics.json` — success rate, mean final distance, path efficiency (tortuosity).
- `REPORT_ch_a.md` — field/λ choice, the generalization setup, parallel speedup, success
  metrics + interpretation, ending with the cross-repo copy command (assets belong at
  `portfolio/public/cellular-gaits/data-ch/`; you don't have portfolio access from here).

## Verify

Chemo-zeroed == closed-loop dynamics (A/B); parallel == sequential fitness within noise;
clips play (<~3 MB); metrics finite; the two approach clips visibly turn opposite ways. Then
stop. **Don't merge.**
