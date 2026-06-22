# cellular-gaits — agent instructions

The compute half of the Cellular Gaits project (the web/portfolio half is the sibling
`portfolio` repo). A neural cellular automaton drives a biomechanically real *Drosophila*
(NeuroMechFly/FlyGym in MuJoCo); CMA-ES (and later RL) evolves controllers, the small web-export
bundle is consumed by the portfolio site. See `README.md` for the science.

## Read first (every session)
- **`../portfolio/docs/cellular-gaits/SYNC.md`** — the cross-machine live state board (what's
  running where, branch claims, next actions). Read first, update last, commit with your work.
- **`AGENT_SAFETY.md`** — the 7 safety rules (never delete/modify files you didn't create; scratch
  in one dir you make; **commit your work on your branch before finishing**; don't merge to main
  unless told).
- **`../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md`** — how we work, the taste, the disposition.
- Science status: `../portfolio/docs/cellular-gaits/research-roadmap.md`.

## Repo layout — prompts, waves, reports (keep the root clean)
Operational meta-files live under `ops/`, never the repo root:
- `ops/prompts/` — active `PROMPT_*.md` directives (one self-contained brief per Claude Code
  session); completed ones move to `ops/prompts/archive/`.
- `ops/reports/` — `REPORT_*.md` session outputs (calibration reports, precompute records, etc.).
- Wave launchers (`setup-*.sh`) live in the sibling repo at `../portfolio/ops/waves/` (co-located
  with `cockpit.sh`); used-up ones go to `../portfolio/ops/waves/archive/`.
Going forward every new prompt/report/wave goes straight into the right `ops/` folder — nothing
lands at the root.

## Cross-machine workflow (two machines: MAC cockpit + WIN 5900X/3080Ti compute)
Either machine can do either role. Discipline keeps them from clobbering each other:
- **Pull at session start, commit + push at session stop.** Never leave uncommitted work when
  switching machines (this is AGENT_SAFETY rule #7, extended across machines).
- **One branch is advanced from one machine at a time** — claim it in SYNC.md first.
- Only small things cross git (code, docs, `PROMPT_*.md`, the web-export bundle). **Heavy
  artifacts stay local and gitignored** (`checkpoints/`, `outputs/`, `.venv/`); move them between
  machines via the side channel (Tailscale scp/rsync or a W&B artifact), never git.
- **Heavy compute (full evolutions, RL) runs on WIN.** Tune `--workers` per machine (WIN ~16,
  MAC ~6). Each machine runs its own `uv sync`.
- **`claude` is now installed on sentry (WIN).** Delegated box sessions work: dispatch a full
  Claude Code session to the workstation with `../portfolio/cockpit.sh run cellular-gaits "<directive>"`
  (headless `-p`, tmux+tee, pulls first; review via `cockpit.sh peek`/`attach`). The old
  ssh+tmux-`uv run` workaround is no longer required.
- Full protocol + one-time setup (Tailscale/SSH, W&B): `../portfolio/docs/cellular-gaits/CROSS_MACHINE.md`.

## Environment
- `uv` for deps/run (`uv sync`, `uv run python ...`). Commit `uv.lock` for reproducible envs
  across machines (remove it from `.gitignore` if still ignored).
- Validation-first for new behaviors: build everything, run a short calibration, STOP and report
  before the full evolution (see `ops/prompts/archive/PROMPT_*_a_*.md` + `ops/reports/REPORT_*`
  for the pattern).
