# Claude Code — FLEET SELF-TEST · verify the workstation compute environment (READ-ONLY diagnostic)

> ## SAFETY (read first; bypassPermissions)
> This is a **read-only diagnostic**. Do **not** modify, create, or delete any *tracked* file; do
> **not** commit, push, merge, or launch any training/evolution. The ONLY writes allowed: a report at
> `outputs/fleet/REPORT_fleet_selftest.md` (`outputs/` is gitignored) and scratch under
> `scratch/fleet/` (which you create). Touch nothing else. If a check needs a tool that isn't there,
> record it as a FAIL and move on — never install system packages or change config. (`AGENT_SAFETY.md`)

> **Where this runs:** dispatched to **sentry (the 5900X/3080 Ti workstation)** via the cockpit
> (`cg run` / `run-b64`), headless in tmux. Your stdout is teed to `~/cockpit-logs/<session>.log` and
> mirrored to the Mac by `cg peek`, so **print every result to stdout** with clear markers — that log
> IS how the cockpit reads you. Run from the repo root. Use a todo list.

## Purpose
Prove the full compute path the project depends on is wired and functional **before** we lean on it
for the nav RL calibration and the connectome RL endgame. Check each item, mark **PASS/FAIL**, and
end with a single machine-readable verdict line. Be honest — a FAIL reported now saves a wasted run
later; do not paper over anything.

## Checks (run in order; one PASS/FAIL line each, with the observed value)

1. **Machine** — `hostname`, `uname -a`, logical CPU count (`nproc`). PASS if this is the workstation
   (expect the 5900X box, not the Mac). Report the hostname you see.
2. **uv** — `uv --version`. PASS if present.
3. **Env** — `uv sync` completes without error in this repo (report how long; if it errors, FAIL with
   the error tail).
4. **Core deps** (via `uv run python -c ...`) — import and print versions for: `flygym` (expect
   **2.0.1**), `mujoco`, `gymnasium`, `numpy`, `torch`. Any import error = FAIL for that dep.
5. **GPU** — `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"`.
   PASS if CUDA is available and the device is the **RTX 3080 Ti**. This is the gate for the connectome
   RL endgame — if CUDA is False, say so loudly (training would fall back to CPU).
6. **The reusable harness imports** — `uv run python -c "from cellular_gaits.rl import ppo, env_base, policies, nav_env"`.
   PASS if all import clean (the connectome endgame reuses this exact harness).
7. **Baseline data present** — confirm `outputs/web_data_n/` (the N-A overfit baseline, needed for
   calibration Gate 2) and `outputs/web_data_ch/` (chemotaxis warm-start) exist and are non-empty.
   FAIL (not error) if missing — note which, since the calibration needs them here.
8. **Git** — current branch, `git status -s` (clean working tree?), `git log -1 --oneline`, and that
   `git fetch` succeeds (the remote/transport is reachable — the cross-machine sync depends on it).
9. **Dispatch tooling** — `claude --version`, `tmux -V`. PASS if both present (proves the box can host
   delegated sessions going forward).
10. **W&B (optional, note-only)** — `uv run python -c "import wandb; print(wandb.__version__)"` and
    whether a login/API key is configured. Report as INFO; not a hard FAIL.

## Output (this is the deliverable)
Print, and also write to `outputs/fleet/REPORT_fleet_selftest.md`, a summary block exactly like:

```
===== FLEET SELF-TEST SUMMARY =====
1 machine        : PASS  <hostname> / <uname>
2 uv             : PASS  <version>
3 env (uv sync)  : PASS  <seconds>
4 core deps      : PASS  flygym 2.0.1, mujoco x, gymnasium x, numpy x, torch x
5 gpu/cuda       : PASS  RTX 3080 Ti
6 rl harness     : PASS
7 baseline data  : PASS  web_data_n ok, web_data_ch ok
8 git            : PASS  <branch> clean, fetch ok
9 dispatch tools : PASS  claude x, tmux x
10 wandb         : INFO  <version / not configured>
===================================
FLEET SELFTEST: PASS
```

If anything failed, the last line must be `FLEET SELFTEST: FAIL (<comma-separated failed check names>)`
so the cockpit harness can grep one line for the verdict. Then **STOP** — do not commit, do not launch
anything, do not "fix" the repo.
