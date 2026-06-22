# Claude Code — N-RL-2b · Targeted reward rebalance + re-run the calibration (STOP, no full run)

> ## SAFETY (read first; bypassPermissions)
> Additive/edit-in-place on the files this task names only. Keep v1 + ALL prior behaviors
> **bit-exact** (the Gate-3 A/B check must stay max|Δ|=0). No `rm -rf` of anything you didn't
> create; scratch in `scratch/nrl/`. `ops/` (other prompts/waves) off-limits. **Commit + push on
> `feat/n-rl-navigation` before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **Run on `feat/n-rl-navigation`, on sentry (WIN)** — needs flygym + `outputs/web_data_n/` (the N-A
> baseline) + the cores. Continues the already-wired N-RL-2 code (`scripts/run_rl_navigation.py`,
> `rl/nav_env.py`). Run from the repo root. Use a todo list.
> **Read first:** `ops/reports/REPORT_n_rl_calibration.md` (the calibration you're fixing) and
> `ops/prompts/PROMPT_n_rl_2_integrate.md` (the design/gates).

## Why (what the first calibration showed)
The 4-gate calibration ran and **correctly failed Gate 4**: a short PPO run drove collisions to ~0
**but reach/detour collapsed to 0** — `detour_correct_rate` rose (0.125→0.5) and min-dist improved,
so the policy *is* learning to orient and avoid, but it **won't commit to the final approach** because
`w_collide=0.75` over-dominates (reaching a goal near an obstacle risks contact → the penalty makes
it hang back). Two more issues surfaced: training **return logged `nan`** (vec envs not wrapped with
`RecordEpisodeStatistics`), and **Gate 2's `detour_success` isn't collision-gated** (N-A scored 0.38
held-out by *grazing through* — 119.8 contacts/episode — instead of the expected ≈0).

This pass fixes exactly those, changes **one experimental variable (the reward balance)**, and re-runs
the **same cheap calibration**. **Do NOT launch the full run.** **Do NOT** touch domain
randomization / bearing range / curriculum this pass (that's the next lever if rebalance isn't enough)
— we change one thing so the signal is attributable.

## The four changes (precise)
1. **Lower the collision penalty:** `w_collide` 0.75 → **0.25** (the `NavRLConfig`/calibration
   default; keep the `--w-collide` CLI flag for sweeping, and note 0.2–0.4 is the sweep band).
2. **Strengthen the dense approach reward** so the final approach isn't suppressed: weight the
   Δapproach term up relative to the per-contact penalty so that a clean reach clearly dominates the
   worst plausible per-episode contact cost. Don't just flip a constant — **report the per-episode
   reward decomposition** (mean Δapproach vs collision vs step-cost vs reach-bonus, pre- and post-PPO)
   so the balance is legible.
3. **Fix the `nan` return:** wrap the vectorized envs with `gymnasium.wrappers.RecordEpisodeStatistics`
   (or the vector equivalent) so `charts/episodic_return` is finite; confirm a real return curve
   prints in the calibration log.
4. **Collision-gate the success metric:** make `detour_success` (and `reach`) require collision-free
   (contacts below a small threshold τ — state τ). Keep the raw/ungated values too, for transparency.
   Re-measure the **N-A baseline** under the gated metric — it should now read **≈0** held-out detour
   (honest), reconciling Gate 2.

## Re-run + what we're looking for
Re-run `uv run python scripts/run_rl_navigation.py --calibrate` (same ~6-update / 48k-step budget).
**The signal that says "rebalance worked":** pre→post-PPO held-out **clean** (collision-gated)
reach/detour now trends **UP** (not collapsing to 0), with collisions staying low-ish (they need not
hit exactly 0). If reach still collapses, diagnose honestly (approach still too weak? gate τ too
strict? budget too small to show it?) and report what you tried — do **not** paper over it.

## Report → update `ops/reports/REPORT_n_rl_calibration.md` (append a "Rebalance pass (2b)" section)
Include: the new `w_collide` + approach weighting, the reward decomposition, the 4 gate results under
the new weights/metric (esp. Gate 2 N-A now ≈0 gated, and Gate 4's pre→post clean reach/detour trend),
the return curve sanity (no more `nan`), and a recommendation.

## Finishing — DO THIS IN ORDER (the previous session stranded its work by stopping before committing)
1. **`git add -A` then commit** on `feat/n-rl-navigation` — include the prior session's wired files
   that are still uncommitted (`scripts/run_rl_navigation.py`, `scripts/test_rl_navigation.py`,
   `src/cellular_gaits/rl/nav_env.py`) **plus** your rebalance changes + the updated report.
2. **`git push`** on `feat/n-rl-navigation` (the cockpit pulls from here — do not skip the push).
3. **Only after the commit + push succeed**, end with: *"Committed + pushed. Confirm the trend and
   we'll decide the full run."*
"STOP" / "do not launch the full run" means **do not run `--full`** — it does **NOT** mean skip the
commit. Commit and push first, then stop.

## Definition of done
- The four changes made in `rl/nav_env.py` / `scripts/run_rl_navigation.py` (and config) only.
- Calibration re-run; `REPORT_n_rl_calibration.md` updated with the "Rebalance pass (2b)" section.
- Prior behaviors bit-exact (Gate 3 still max|Δ|=0); existing tests pass.
- **Committed + pushed on `feat/n-rl-navigation`** (verified before stopping). Full run **not launched.**
