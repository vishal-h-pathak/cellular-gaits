# Claude Code — N-RL-2c · Bearing curriculum + w_collide anneal + widened held-out (re-calibrate, STOP before full run)

> ## SAFETY (read first; bypassPermissions)
> Edit-in-place only on the files this task names (`rl/nav_env.py`, `scripts/run_rl_navigation.py`,
> config). Keep v1 + ALL prior behaviors **bit-exact** (Gate-3 A/B must stay max|Δ|=0; new curriculum
> defaults must leave the calibrated 2b behavior reproducible when the curriculum is disabled). No
> `rm -rf` of anything you didn't create; scratch in `scratch/nrl/`. `ops/` off-limits. Unsure → leave
> it + note it. (`AGENT_SAFETY.md`)

> **Run on `feat/n-rl-navigation`, on sentry (WIN)** — builds on the **committed 2b** code
> (`w_collide=0.25`, `w_approach=2.0`, `RecordEpisodeStatistics`, collision-gated metrics τ=2).
> Run from the repo root. Use a todo list.
> **Read first:** `ops/reports/REPORT_n_rl_calibration.md` — especially the **"Rebalance pass (2b)"**
> section and the **Gate-2 finding**, plus `rl/nav_env.py` and `scripts/run_rl_navigation.py`.

## Why (what 2b concluded — the real finding)
2b fixed the reward balance (no more stall-collapse; finite return). But the calibration's important
finding is **the held-out bearing band `[20–60°]` is too easy** — it sits inside the chemo forager's
~40° homing cone, where N-A already clean-reaches **62%**, so beating N-A there proves nothing. The
whole point of the RL pivot is to **dissolve that narrow-cone overfit → learn OMNIDIRECTIONAL homing**.
2c wires the levers so the full run tests exactly that, then re-calibrates on a **harder (wider)** band
to confirm a learnable signal. **Decision locked with Vishal: progressive curriculum → omnidirectional.**
**Do NOT launch the full run.**

## The changes (config-driven; used by `--full`, and a compressed form by `--calibrate`)
1. **Progressive bearing curriculum** — widen the *training* DR bearing band from the forager's
   ~`[20–60°]` toward **omnidirectional `[0, 360°)`** over training. Default to a **step-fraction
   schedule** (e.g. widen linearly over the first ~60% of steps); optionally gate widening on held-out
   clean-reach crossing a threshold (state which you implemented). The frozen held-out must stay
   rejection-sampled **disjoint** from training at every stage.
2. **`w_collide` anneal** — ramp **0.25 → ~0.75** over training (after homing stabilizes / over a step
   fraction), so homing is learned **before** avoidance tightens (prevents the 2a stall basin).
3. **Far→near obstacle curriculum** — obstacle starts off-path / distant, slid onto the path as reach
   stabilizes.
4. **Widened + matching held-out set** — extend the frozen held-out bearings to cover the widened range
   (omnidirectional for the full run), frozen + disjoint. **Report held-out broken down by bearing** so
   the homing envelope is visible.
Keep all four behind config with defaults that **reproduce 2b when disabled** (so Gate 3 / prior
behavior stays bit-exact).

## Re-calibrate — the cheap gate, on the HARDER band
- Short PPO run, budget **~100–150k steps** (state what you used — larger than 48k because the wider
  task is harder), curriculum **enabled** (compressed to the budget) **or** a fixed moderately-wide
  band — your call, state it.
- **The success signal is a LEADING INDICATOR, not omnidirectional convergence** (impossible in a cheap
  budget): does the policy extend **clean_reach to held-out bearings JUST BEYOND the forager's ~40°
  cone** (≈60–90°)? **Report held-out clean_reach split into "inside ~40°" vs "outside ~40°", pre→post.**
  An upward trend on the **outside** bearings = green (the curriculum will keep extending under the full
  budget). Dead-flat 0 outside = real no-go → diagnose (warm-start insufficient at wide bearings? need a
  homing-only obstacle-free warm-up phase first?) and report what you tried.
- **STOP.** Do not launch `--full`.

## Finishing — DO THIS IN ORDER (commit before the STOP)
1. `git add -A` then **commit** on `feat/n-rl-navigation` (all 2c changes + the report update).
2. `git push` on `feat/n-rl-navigation`.
3. Only after the push succeeds, end with: *"Committed + pushed. Confirm the leading-edge signal and
   we'll launch the full run."* — "STOP" means do not run `--full`; it does **not** mean skip the commit.

## Report → append "Curriculum pass (2c)" to `ops/reports/REPORT_n_rl_calibration.md`
The schedules used (bearing curriculum, `w_collide` anneal, obstacle curriculum), the widened held-out
definition, the calibration budget, the held-out **clean_reach split inside vs outside ~40°** (pre→post),
the finite return curve, the verdict (does the leading edge move?), and the proposed `--full` command
with the curriculum/anneal/wide config baked in.

## Definition of done
- Changes in `rl/nav_env.py` / `scripts/run_rl_navigation.py` (+ config) only; `nca.py`/`env.py`/
  `evolve*.py` untouched; prior behaviors bit-exact (Gate 3 max|Δ|=0 with curriculum disabled).
- Re-calibration run; `REPORT_n_rl_calibration.md` has the "Curriculum pass (2c)" section.
- **Committed + pushed on `feat/n-rl-navigation`** (verified before stopping). Full run **not launched.**
