# Gate — PPO loop correctness (`feat/n-rl-ppo`)

**Question the gate answers:** does `rl/ppo.py` actually *learn*, independent of the fly
sim? If yes, the integrate session only has to *wire* PPO to the nav env + NCA policy, not
*debug* PPO.

**Setup.** A tiny MLP stub agent (`scratch/nrl/gate_ppo_run.py::MLPStubAgent`) implementing
the Agent contract **verbatim** — only `get_action_and_value(obs, action=None)` and
`mean_action(obs)`, **no `get_value`** — trained on `Pendulum-v1` via `rl.ppo.train`. The
standard cleanrl wrappers (ClipAction / NormalizeObservation / NormalizeReward /
RecordEpisodeStatistics) live in the gate's `make_env` factory, so `ppo.py` stays
env-agnostic. CPU-only on the Mac (the loop auto-selects CUDA on the workstation).

**Config (short budget):** `total_steps=120k`, `n_envs=4`, `n_steps=1024`, `n_minibatch=32`,
`update_epochs=10`, `gamma=0.9`, `gae_lambda=0.95`, `lr=1e-3` (linear anneal), `clip=0.2`,
`ent_coef=0.0` → 29 updates.

## Result — return rises (PASS)

Training episodic return (rolling, per update), Pendulum (solved ≈ −200; higher = better):

| update | step  | train return |
|-------:|------:|-------------:|
| 1      | 4k    | −1277        |
| 5      | 20k   | −1174        |
| 10     | 41k   | −1042        |
| 15     | 61k   |  −866        |
| 20     | 82k   |  −575        |
| 25     | 102k  |  −402        |
| 29     | 119k  |  −496        |

Monotone improvement, return roughly **−1277 → −475 (≈2.6×)** in 120k steps — the PPO math
(rollout, GAE(λ), clipped surrogate, clipped value loss, minibatch epochs, grad clip, LR
anneal) is correct. Value loss collapses to ~1e-3 and `approx_kl` stays in the healthy
0.002–0.013 band (no policy blow-up).

Held-out **deterministic** (`mean_action`) return via the `eval_fn` hook:
random-init **−1024 → −815** final (noisier — deterministic Pendulum control is sensitive at
this tiny budget; the training-return curve is the cleaner learning signal). The point of
including it is to exercise the pluggable hook end-to-end.

Artifacts: `scratch/nrl/gate_ppo_result.json`, checkpoints in `scratch/nrl/ckpt/`.
Resume verified separately (a run resumed cleanly from `*_latest.pt` at the right update/step).

## `PPOConfig` knobs (the surface the integrate session tunes)

`exp_name, seed, torch_deterministic, device("auto")` · `total_steps, n_envs, n_steps,
n_minibatch, update_epochs` · `gamma, gae_lambda, clip_coef, clip_vloss, ent_coef, vf_coef,
max_grad_norm, norm_adv, target_kl` · `learning_rate, anneal_lr, eps` · `ckpt_dir, ckpt_every,
eval_every, log_every` · `wandb, wandb_project, wandb_run_id, wandb_tags, behavior`. Derived:
`batch_size, minibatch_size, num_updates`.

## Eval-hook signature

```python
eval_fn(agent: nn.Module) -> dict   # {metric_name: float}, logged under eval/*
```
Called every `cfg.eval_every` updates under `torch.no_grad()` with the agent in `eval()` mode.
The integrate session passes nav's held-out detour/reach evaluator here; keys land in W&B as
`eval/holdout_detour`, etc.

## Task-agnostic vs leaked (audit)

**Task-agnostic (transfers verbatim to the connectome endgame):** *everything in `ppo.py`.*
The loop imports no concrete env or policy class — it depends only on (i) a `make_env()` thunk
(vectorized with `gymnasium.AsyncVectorEnv`) and (ii) an agent with the two-method contract.
Obs/action shapes are read from `single_observation_space` / `single_action_space` at runtime;
nothing assumes `(6,8,8)` or Pendulum's `(3,)`. The same `train(cfg, make_env, agent, eval_fn)`
call drives the fly nav task and, later, `ConnectomePolicy` on its env, unchanged.

**Nav-specific / leaked into `ppo.py`:** **none.** All Pendulum/nav specifics live in the gate
script under `scratch/nrl/` (the stub agent, the env factory + wrappers, the eval_fn), never in
the harness.

## Notes for the merge / other sessions

- **`get_value`:** intentionally **NOT** required. The GAE bootstrap value is taken from
  `get_action_and_value(next_obs)` with the sampled action discarded. The policy + integrate
  sessions therefore do **not** need to add a `get_value` method. (If a value-only forward pass
  is ever wanted for speed, add it across policy + integrate together, not silently here.)
- **`rl/__init__.py`:** this session shipped a minimal one re-exporting `PPOConfig, train`. If a
  wave-1 branch (env/policy) also ships an `__init__.py`, keep theirs and re-add the `ppo`
  re-export by hand.
- **`uv.lock`:** `gymnasium[classic-control]` was added as the superset (the env session also
  adds `gymnasium`; this version wins cleanly at merge). **`uv lock` must be re-run post-merge**
  to reconcile the wave-1 branch lockfiles — same standing note as the `__init__.py` merge.
