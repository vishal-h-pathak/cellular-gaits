# ops/ — operational meta-files (not source)

Everything an agent session needs to be *run* lives here, so the repo root stays clean.

```
ops/
  prompts/            active PROMPT_*.md — one self-contained brief per Claude Code session
    archive/          completed prompts (kept for the record / re-use of the pattern)
  reports/            REPORT_*.md — session outputs (calibration, precompute records, …)
```

Conventions:
- **One prompt = one session.** Self-contained: inline the AGENT_SAFETY preamble, the contract,
  the validation-first gates, and the definition of done. Naming: `PROMPT_<campaign>_<wave>_<slug>.md`.
- **Move a prompt to `archive/` once its work has landed** (merged or shipped). Active = still to run.
- **Reports** are written by the session into `ops/reports/` (not the root).
- **Wave launchers** (`setup-*.sh`) live in the sibling repo: `../portfolio/ops/waves/`
  (co-located with `cockpit.sh`). They open a session per chunk and paste the directive.

See `../CLAUDE.md` ("Repo layout") and `AGENT_SAFETY.md`.
