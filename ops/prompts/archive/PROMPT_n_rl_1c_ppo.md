# Claude Code — N-RL-1c · The PPO loop (policy-/env-agnostic — the reusable connectome harness)

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors **bit-exact**.
> Additive only. No `rm -rf` of anything you didn't create; scratch in ONE dir you make
> (`scratch/nrl/`). `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh` off-limits. **Commit on your
> branch before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **You are in a git worktree on branch `feat/n-rl-ppo`.** Run from the repo root. Use a todo list.
> **Read first:** `PROMPT_n_rl_navigation.md` (full design — you build the `ppo.py` slice + this is
> explicitly the **reusable harness for the connectome endgame**, so keep it task-agnostic),
> `../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md`, and `PROMPT_wandb_integration.md` (W&B logging
> convention). You do NOT depend on the concrete env or policy — you code against the two protocols below.

## Your slice

Build **`src/cellular_gaits/rl/ppo.py`** only — a clean, single-file **PPO** loop (cleanrl
`ppo_continuous_action` is the reference). It must be **agnostic** to the task: it takes an env factory
and an agent, so a future `ConnectomePolicy` + a different env drop in unchanged. That reuse is the point.

- **Protocols you depend on (do not import concrete classes):**
  - *Env:* a `gymnasium`-compatible env via a `make_env()` factory, vectorized with
    `gymnasium.AsyncVectorEnv` (N parallel CPU envs — sim is CPU-bound; target ~16 on the workstation).
  - *Agent:* a `torch.nn.Module` exposing `get_action_and_value(obs, action=None) -> (action, logprob,
    entropy, value)` and `mean_action(obs)` (the cleanrl interface; the policy session implements it).
- **Implement:** rollout collection, GAE(λ), clipped surrogate + value loss + entropy bonus, minibatch
  epochs, grad clipping, LR schedule; GPU for the update (CUDA on the workstation), CPU for the vec envs.
- **Checkpointing** (`scratch/nrl/ckpt/`) + resume. **W&B logging** per `PROMPT_wandb_integration.md`:
  return, episode length, value/policy loss, entropy, KL, and a **pluggable eval hook** (`eval_fn(agent)`)
  for held-out metrics — the integrate session wires nav's held-out detour/reach into it.
- A `PPOConfig` dataclass (steps, n_envs, n_minibatch, clip, lr, gamma, lambda, ...).

## Gate you own — loop correctness, report in `scratch/nrl/gate_ppo.md`

Prove the loop **learns** on a cheap standard control env (e.g. `Pendulum-v1` or `HalfCheetah`-free
`gymnasium` classic-control) with a tiny MLP stub agent implementing the same interface: return rises
over a short budget. This validates the PPO math independently of the fly sim, so the integrate session
only has to wire, not debug PPO.

## Definition of done

- `rl/ppo.py` (+ extend `rl/__init__.py` if present; else create minimal + note the merge).
- The standard-env learning check reported in `scratch/nrl/`. No prior files touched; existing tests pass.
- **Committed on `feat/n-rl-ppo`** (not merged). Report: the `PPOConfig` knobs, the eval-hook signature,
  the learning-check curve, and an explicit note on **what here is task-agnostic** (transfers to the
  connectome endgame) vs anything nav-specific that leaked in (there should be none).
