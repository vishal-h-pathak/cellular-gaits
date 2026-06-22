# Claude Code — N-RL-2 · Integrate + run the validation-first calibration (STOP before the full run)

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors **bit-exact**.
> Additive only. No `rm -rf` of anything you didn't create; scratch in ONE dir you make
> (`scratch/nrl/`). `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh` off-limits. **Commit on your
> branch before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **Run on `feat/n-rl-navigation` AFTER the three wave-1 branches are merged into it**
> (`feat/n-rl-env` + `feat/n-rl-policy` + `feat/n-rl-ppo`). Run from the repo root. Use a todo list.
> **Read first:** `PROMPT_n_rl_navigation.md` (full design + the four gates + honesty caveats), the
> three wave-1 reports in `scratch/nrl/`, `REPORT_n_a_calibration.md` and the N-A result (the overfit
> baseline you must reproduce: held-out detour **0.00**, reach 4/8), and `PARTNER_BRIEF.md`.

## Your slice — wire it together, then prove it (don't launch the full run)

1. **`scripts/run_rl_navigation.py`** with `--calibrate` / `--full` gates (mirror the `run_evolution_*`
   CLIs): builds `NavRLEnv` (from 1a) + `NCAPolicy` warm-started from chemo (from 1b) + the PPO loop
   (from 1c), wires nav's **held-out** detour/reach/collision metrics into the PPO `eval_fn`, W&B on.
2. **`scripts/test_rl_navigation.py`** — smoke test (env+policy+one PPO update step run clean).
3. Resolve any interface drift between the three modules (the env contract / the `get_action_and_value`
   / the `eval_fn` signature). This glue is expected — that's why integration is its own session.

## The four gates → `REPORT_n_rl_calibration.md` (repo root), then STOP

1. **Env sanity** — confirm 1a's Gate 1 still holds after the merge (shapes, Newton cap, vec throughput).
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
prompt. Then literally: *"Confirm and I'll launch the full run on sentry."* **Do not launch it.**

## Definition of done

- `scripts/run_rl_navigation.py` + `scripts/test_rl_navigation.py`; the three modules wired and merged.
- All four gates reported in `REPORT_n_rl_calibration.md`; full run **proposed, not launched.**
- All existing tests pass + prior behaviors bit-exact.
- **Committed on `feat/n-rl-navigation`** (not merged).
