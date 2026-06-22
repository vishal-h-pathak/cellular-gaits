# Claude Code — N-RL-1b · The NCA policy (PPO-ready) + warm-start + A/B gate

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors **bit-exact**
> (`nca.py`, `env.py`, `evolve*.py`, `navigation.py`, `checkpoints/`, `outputs/`, `web_data*`).
> Additive only. No `rm -rf` of anything you didn't create; scratch in ONE dir you make
> (`scratch/nrl/`). `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh` off-limits. **Commit on your
> branch before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **You are in a git worktree on branch `feat/n-rl-policy`.** Run from the repo root. Use a todo list.
> **Read first:** `PROMPT_n_rl_navigation.md` (full design — you build the `policies.py` slice),
> `../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md`, and `src/cellular_gaits/nca.py` (the `nav` mode,
> `NCA(nav=True)` ≈1524 params, `warm_start_from_chemo`, the forward pass). You do NOT need the new env
> — you code against the documented obs/action contract below (it already exists in `nca.py` from N-A).

## Your slice

Build **`src/cellular_gaits/rl/policies.py`** only — the NCA wrapped as a PPO-ready policy. No env, no
training loop (parallel sessions own those).

- **`NCAPolicy(nn.Module)`** wrapping `NCA(nav=True)`: a **Gaussian policy** — the NCA forward pass is
  the action **mean**, plus a learned global `log_std`; a small **value head** (MLP on the pooled grid).
- Implement the **cleanrl Agent interface exactly** (the PPO session codes against this):
  - `get_action_and_value(obs, action=None) -> (action, logprob, entropy, value)` — samples when
    `action is None`, else scores the given action.
  - `mean_action(obs) -> action` — deterministic (eval / A/B).
  - `warm_start_from_chemo(path)` — load the trained chemo controller into the nav model, **zero-init
    the two feeler input channels** (so feelers-off == the pure forager), copy conv1[0:7]+bias+conv2;
    source `outputs/web_data_ch/chemotaxis_controller.json`. Validate param count vs `NCA(nav=True)`.
- **Contract you consume:** obs = `(6,8,8)` float32 (batched `(B,6,8,8)`); action = `(N_motor,)`
  float32 in [-1,1]. If you need N_motor, read it from `NCA(nav=True)`'s motor-cell count, not a literal.

## Gate you own — A/B integrity, report in `scratch/nrl/gate_ab.md`

With feelers zeroed, `mean_action` reproduces the chemotaxis forward pass **bit-exact** (max|Δ| = 0 vs
the 8-input chemo path on shared weights) — same A/B check the N-A calibration used (warm-start changes
nothing until the feeler weights move). Warm-started value head + log_std are present and finite.

## Definition of done

- `rl/policies.py` (+ extend `rl/__init__.py` if it exists; if the env session hasn't created it,
  create a minimal one and note the merge).
- A/B gate reported in `scratch/nrl/`. `nca.py` untouched; all existing tests pass + bit-exact.
- **Committed on `feat/n-rl-policy`** (not merged). Report: the `get_action_and_value` signature as
  implemented, the warm-start param-count check, the A/B max|Δ|.
