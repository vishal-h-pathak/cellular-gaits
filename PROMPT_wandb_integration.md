# Claude Code — Weights & Biases integration (Layer 3 of the cross-machine system)

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + all prior behaviors intact. No
> `rm -rf` of anything you didn't create; scratch in ONE dir you make. Append to shared files;
> don't rewrite. **Commit your work on the branch before finishing** (don't merge to main). When
> unsure, leave it and note it. (Full: `AGENT_SAFETY.md`)

> **Run from the `cellular-gaits` repo root.** Read `CLAUDE.md` and
> `../portfolio/docs/cellular-gaits/SYNC.md` first; claim the branch in SYNC.md. Branch:
> `git checkout -b feat/wandb-logging` off the current tip. Use a todo list.

## Goal

Add **optional** Weights & Biases logging to the evolution loop so CMA-ES runs from **either
machine** land in one dashboard (fitness curves, config, final artifacts) — the Layer-3 run
tracker of the cross-machine workflow. It must be strictly additive and **off by default**: a run
without `--wandb` (or without `wandb` installed / logged in) behaves exactly as today.

## Build

1. **Dependency (optional):** `uv add wandb`. Import it lazily inside the `--wandb` path so the
   package is not required for normal runs (`try: import wandb` guarded).
2. **Shared helper** (e.g. `src/cellular_gaits/wandb_logging.py`): a tiny wrapper —
   `init(project, run_id, cfg, tags)`, `log_generation(gen, best_fit, mean_fit, **extras)`,
   `log_artifact(path, name, type)`, `finish()` — each a no-op when wandb is disabled/unavailable.
   Tag every run with the **hostname** (so MAC vs WIN runs are distinguishable) and the behavior
   (walk/chemo/escape/nav).
3. **Wire into the evolution loop** used by the run scripts (`evolve_*.py` / the shared CMA-ES
   driver). Per generation log `best_fit`, `mean_fit`, `sigma`, wall-time/gen. At the end, log the
   exported controller JSON + any clips as a `wandb.Artifact` so heavy outputs can ride W&B
   instead of git when desired.
4. **CLI flags** on the `run_evolution_*.py` scripts: `--wandb` (enable), `--wandb-project`
   (default `cellular-gaits`), `--wandb-run-id` (default = existing `run_id`). Mirror the flag
   across the run scripts that share the loop (don't fork the loop).
5. **Docs:** append a short "W&B logging" note to `README.md` and tick the Layer-3 line in
   `../portfolio/docs/cellular-gaits/CROSS_MACHINE.md` (append-only). Add the one-time setup
   reminder: `wandb login` per machine.

## Constraints

- **Default path unchanged & bit-exact:** a normal `uv run python scripts/run_evolution_navigation.py …`
  (no `--wandb`) must produce identical results and side effects to today. Confirm the existing
  tests still pass.
- No network calls unless `--wandb` is passed AND wandb is importable AND a login exists; fail
  **soft** (warn, continue the run) if logging can't init — a tracker outage must never kill a
  multi-hour evolution.
- Don't log anything heavy every step; per-generation is enough.

## Definition of done

- `--wandb` works end-to-end on a short smoke run (`--pop 8 --gens 3 --wandb`); a normal run is
  untouched and tests pass. `npx`/lint equivalents clean (`uv run pytest` green).
- **Commit on `feat/wandb-logging`** (don't merge). Update SYNC.md (clear your claim, log the
  result). Report: files added/changed, the smoke-run dashboard URL, and the exact enable command.
