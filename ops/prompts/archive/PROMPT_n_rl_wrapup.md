# Claude Code — N-RL-WRAPUP · Consolidate the navigation→RL work for the night (git only)

> ## SAFETY (read first; bypassPermissions)
> This is a **git-consolidation** session. You WILL run merges and commits (sanctioned). Hard rules:
> **never push, never merge to `main`.** Use **explicit `git add <path>` only — NEVER `git add -A`
> or `git add .`** (both repos have unrelated untracked/in-flight files that must stay untouched).
> Do not edit source logic, do not run RL, do not run flygym sims. If a merge conflict is anything
> other than the expected ones below, STOP and report rather than guessing. (Full: `AGENT_SAFETY.md`)

> **Start in the `cellular-gaits` repo root, on `feat/n-rl-navigation`.** Use a todo list. Goal:
> end the night with ONE consolidated branch (`feat/n-rl-navigation`) holding the N-A foundation +
> the merged RL harness (env+policy+ppo) + the physical-obstacle change, worktrees pruned, both
> repos committed, and the living docs (already updated by Cowork) committed. Nothing left phantom.

## Part A — cellular-gaits

1. **Capture the stray prior work for-record (clears the dirty tree before merging).** The working
   tree has two modified tracked files from a *prior, now-superseded* session: `src/cellular_gaits/
   evolve_navigation.py` (obstacle offset 1.2→1.8) and `REPORT_n_a_calibration.md` (recalibration
   tables, Gate-4 ❌). These are superseded by the RL pivot + physical obstacles, but don't discard
   them — commit them on `feat/n-rl-navigation` as a labeled record:
   `git add src/cellular_gaits/evolve_navigation.py REPORT_n_a_calibration.md`
   `git commit -m "N-A: prior condition recalibration (offset 1.2→1.8), committed for-record — superseded by the RL pivot + physical obstacles"`
   Also commit the physics prompt if untracked: `git add PROMPT_n_rl_phys_obstacles.md` and include it
   in this commit (or a follow-up). **Do not** `git add` the other untracked historical `PROMPT_*.md`
   or old `scratch/*` dirs — leave them.

2. **Merge the three wave-1 RL branches** into `feat/n-rl-navigation`:
   `git merge --no-ff feat/n-rl-env feat/n-rl-policy feat/n-rl-ppo`
   Expected conflicts (resolve exactly these; anything else → STOP and report):
   - `src/cellular_gaits/rl/__init__.py` — each branch created a minimal one. **Union the exports**
     (`EmbodiedRLEnv`, `NavRLEnv`/`NavRLConfig`, `NCAPolicy`, `PPOConfig`/`train`) into one file.
   - `pyproject.toml` — env added `gymnasium`, ppo added `gymnasium[classic-control]`. **Keep the
     superset `gymnasium[classic-control]`** (one entry).
   - `uv.lock` — don't hand-merge; take either side, then regenerate (step 3).
   (`env.py` / `evolve_navigation.py` won't conflict — the RL branches are additive and didn't touch them.)

3. **Reconcile the lockfile:** `uv lock` (then `uv sync` to verify it resolves), and commit the
   result: `git add pyproject.toml uv.lock && git commit -m "N-RL: reconcile deps after merging env+policy+ppo (gymnasium[classic-control])"`. If the merge auto-committed, amend/extend as needed so the lock is committed.

4. **Smoke-check the merged package (no flygym, no RL):** `uv run python -c "import cellular_gaits.rl; from cellular_gaits.rl import EmbodiedRLEnv, NavRLEnv, NCAPolicy, PPOConfig, train; print('rl imports OK')"` and run the artifact-free tests (`uv run python scripts/test_nca.py` etc. — skip anything needing gitignored `outputs/` artifacts or flygym sims). Report pass/fail; do NOT try to fix sim/artifact-dependent failures (they're environmental).

5. **Prune the worktrees** (git already marks them prunable):
   `git worktree remove ~/dev/jarvis/cg-rl-env ~/dev/jarvis/cg-rl-policy ~/dev/jarvis/cg-rl-ppo`
   (use `--force` only if a worktree is clean but flagged; if any has uncommitted changes, STOP and report — don't force-drop work).

6. Confirm `git status` is clean (aside from the deliberately-left historical untracked files) and
   `feat/n-rl-navigation` now contains rl/{env_base,nav_env,policies,ppo,__init__}.py + the physical-
   obstacle env.py. Do **not** merge to main.

## Part B — portfolio docs (commit what Cowork already updated; surgical adds only)

`cd ~/dev/jarvis/portfolio`. The cellular-gaits living docs were just updated by Cowork. Commit
**only these files** (the repo has unrelated in-flight changes — do NOT touch them, NO `git add -A`):
```
git add docs/cellular-gaits/NEXT_SESSION.md docs/cellular-gaits/SYNC.md \
        docs/cellular-gaits/research-roadmap.md docs/cellular-gaits/PARTNER_BRIEF.md \
        setup-rl-navigation.sh
git commit -m "cellular-gaits docs: RL harness + physical obstacles consolidated; NEXT_SESSION/SYNC/roadmap updated for tomorrow's wave-2 calibration"
```
If any of those paths shows no changes, drop it from the add (don't invent changes). Do not commit
or stage anything else in portfolio. Do not push.

## Definition of done (report this)

- `feat/n-rl-navigation` is the single consolidated branch (foundation + merged harness + physical
  obstacles); the three `feat/n-rl-*` chunk branches merged; worktrees removed; `git status` clean.
- `uv.lock` reconciled; `cellular_gaits.rl` imports; artifact-free tests pass.
- Portfolio docs committed (surgical). Nothing pushed, nothing merged to main, unrelated work untouched.
- Report: the commit SHAs, how each expected conflict was resolved, the smoke-check result, and
  anything you had to STOP on.
