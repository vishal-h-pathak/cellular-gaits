# Claude Code — N-RL-2 · Integrate + run the validation-first calibration (STOP before the full run)

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors **bit-exact**.
> Additive only. No `rm -rf` of anything you didn't create; scratch in ONE dir you make
> (`scratch/nrl/`). `.worktrees/`, `ops/` (other prompts/waves) off-limits. **Commit on your
> branch before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **Run on `feat/n-rl-navigation`** — the three wave-1 chunks (`feat/n-rl-env` + `feat/n-rl-policy`
> + `feat/n-rl-ppo`) are **already merged** here (env+policy+ppo + the physical-obstacles work).
> This session runs **on sentry (WIN)** via a delegated Claude session, because the calibration
> needs flygym + `outputs/web_data_n/` (the N-A baseline) + `outputs/web_data_ch/` + the cores,
> which all live on the box. Run from the repo root. Use a todo list.
> **Read first:** `ops/prompts/PROMPT_n_rl_navigation.md` (full design + the four gates + honesty
> caveats), the three wave-1 reports in `scratch/nrl/`, `ops/reports/REPORT_n_a_calibration.md` and
> the N-A result (the overfit baseline you must reproduce: held-out detour **0.00**, reach 4/8), and
> `../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md`.
> **Already verified (don't redo):** `rl/ppo.py`'s done-mask reset is rank-generic for the
> `(B,4,8,8)` CA state (`done_view = (n_envs,) + (1,)*len(state_shape)` broadcasts for any rank) — the
> last open code gate before a real PPO run is cleared.

## Carried findings from the physical-obstacles work (apply these — they postdate the original prompt)
The obstacles are now **physically real** (explicit fly-geom↔obstacle MuJoCo contact pairs),
validated to block head-on with no tunneling and no-obstacle physics byte-exact. Two consequences:
- **Raise `w_collide` 0.2 → ~0.75** (sweep 0.5–1.0 if needed). Real contacts run **11–28/episode**
  (vs N-A's geometric 100+), so the old 0.2 no longer bites. Check the penalty against the Δapproach /
  reach-bonus scale in Gate 4 and **report the balance** — don't just hard-set it and move on.
- **Drop the Newton/CG solver cap.** The "pinned fly explodes the solver" premise did **not**
  reproduce with real contacts (`ncon` ≤23; uncapped was faster). Remove it for the calibration, but
  **assert `ncon` stays bounded (≤~25)** across a rollout so any real instability is still caught.

## Your slice — wire it together, then prove it (don't launch the full run)

1. **`scripts/run_rl_navigation.py`** with `--calibrate` / `--full` gates (mirror the `run_evolution_*`
   CLIs): builds `NavRLEnv` (from 1a) + `NCAPolicy` warm-started from chemo (from 1b) + the PPO loop
   (from 1c), wires nav's **held-out** detour/reach/collision metrics into the PPO `eval_fn`, W&B on.
   Set `w_collide`≈0.75 and drop the Newton cap (see carried findings above).
2. **`scripts/test_rl_navigation.py`** — smoke test (env+policy+one PPO update step run clean).
3. Resolve any interface drift between the three modules (the env contract / the `get_action_and_value`
   / the `eval_fn` signature). This glue is expected — that's why integration is its own session.

## The four gates → `ops/reports/REPORT_n_rl_calibration.md`, then STOP

1. **Env sanity** — confirm 1a's Gate 1 still holds after the merge (shapes, vec throughput) **on the
   now-physical task**; with the Newton cap dropped, **assert `ncon` stays bounded (≤~25)** across a
   rollout instead of confirming the cap.
2. **Baseline = the N-A overfit, measured in THIS env** — load the N-A CMA-ES controller
   (`checkpoints/2026-06-21T05-22-48Z/gen_70` or `outputs/web_data_n/navigation_controller.json`) as the
   policy and eval on the held-out set; it must reproduce ≈0 held-out detour success. This proves the
   held-out yardstick is honest before we claim RL beats it.
3. **A/B integrity** — confirm 1b's gate after merge: feelers-off policy still walks + homes, prior
   behaviors bit-exact, all existing tests pass.
4. **Short PPO run shows generalization signal** — a small step budget lifts **held-out** detour_success
   and reach **above the N-A baseline** with collision rate falling. If flat, diagnose (reward scale, DR
   too hard, curriculum width) and report what you tried — do **not** paper over it.

End the report with: files added/changed, the four gate results, the proposed **full-run command + step
budget + the bearing-width decision** (narrow=safe / wide=bets PPO relearns omnidirectional homing — give
a recommendation), the **what-transfers-to-the-connectome** note, and the honesty caveats from the master
prompt. Then literally: *"Confirm and I'll launch the full run."* **Do not launch it.** (You are
already on sentry, so the green-lit full run is just `--full` here — but still STOP for the go/no-go.)

## Definition of done

- `scripts/run_rl_navigation.py` + `scripts/test_rl_navigation.py`; env+policy+ppo wired.
- All four gates reported in `ops/reports/REPORT_n_rl_calibration.md`; full run **proposed, not launched.**
- All existing tests pass + prior behaviors bit-exact.
- **Committed on `feat/n-rl-navigation`** (not merged). Push so the cockpit can pull the report.
