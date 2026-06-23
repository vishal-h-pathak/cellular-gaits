# Claude Code — EB-0C · A body-side escape/takeoff motor primitive on NeuroMechFly

> ## SAFETY (read first; bypassPermissions)
> Additive — new module; keep v1 + ALL prior behaviors **bit-exact** (do not change `nca.py`/`env.py`/
> `evolve*.py`/the escape exports). No `rm -rf` of anything you didn't create; scratch in `scratch/eb/`.
> `ops/` off-limits. **Commit on your branch before finishing** (don't merge). Unsure → leave it + note
> it. (`AGENT_SAFETY.md`)

> **Runs on the MAC** (MuJoCo + the existing escape controller are Mac-feasible — **no sentry**). On
> branch `feat/eb-body`. Runs **in parallel** with EB-0A (different files — `embodied/` vs `brain/`).
> Use a todo list. **Read first:** `../portfolio/docs/cellular-gaits/EMBODIED_BRAIN_PLAN.md`,
> `ops/reports/REPORT_x_a.md` (the trained escape behavior) and how the escape controller / `data-x`
> export drives the body. **Understanding-first:** include a plain-English explainer note (report material).

## Context
The embodied loop needs a **body-side escape primitive**: a callable that makes the NeuroMechFly fly
**bolt / flee (and/or take off)** on command. Later (EB-1) this is triggered by the real connectome's
**DNp01 (giant fiber)** readout. This task builds *only* the body half — a clean, on-demand escape
motion — so it can run in parallel with the brain work.

## Your slice — own `src/cellular_gaits/embodied/body.py`
1. **Build a callable escape primitive**, e.g. `trigger_escape(sim, direction=...) ` (or a small class)
   that drives a fast, recognizable bolt/turn-and-flee on the MuJoCo fly, parameterizable by direction
   and magnitude.
2. **REUSE, don't retrain.** The project already has a **trained escape body behavior** (the X-A escape
   controller makes the fly bolt opposite ways for opposite threats; exported in `data-x`). Wrap that as
   the primitive (drive the body with the existing controller / its motor output), **or** if NeuroMechFly
   ships a cleaner takeoff, use that. **Do NOT train a new RL controller here** — if you conclude one is
   genuinely required, **STOP and report it as a follow-up** (that would be the one part needing sentry).
3. **Validation:** invoke the primitive in a short MuJoCo rollout → the body **visibly escapes**
   (bolt/flee), reproducibly, for a couple of directions. Report the motion used + how it's invoked +
   a metric (e.g. displacement / heading change over the window).
4. **Explainer note** → `scratch/eb/body_explainer.md` (report material): how the body executes escape —
   the motor primitive, where it came from, what "escape" looks like at the joint/leg level.

## Contract (so EB-1 can call you)
Expose a stable entry point EB-1 will drive from the brain's DNp01 rate: a function/method that takes a
**scalar escape drive** (and optional direction) and applies the escape motion for one control window.
Document its signature in the explainer.

## Definition of done
- `src/cellular_gaits/embodied/body.py` with the escape primitive + a documented call signature.
- Validation rollout shows a reproducible escape; reported with a metric.
- `scratch/eb/body_explainer.md` written.
- Prior behaviors bit-exact; **committed on `feat/eb-body`** (not merged). STOP and report.
