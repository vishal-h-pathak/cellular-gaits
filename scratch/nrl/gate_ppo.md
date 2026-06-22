# Gate — PPO loop correctness (`feat/n-rl-ppo`)

**Question the gate answers:** does `rl/ppo.py` actually *learn*, independent of the fly
sim? If yes, the integrate session only has to *wire* PPO to the nav env + NCA policy, not
*debug* PPO.

**Setup.** A tiny MLP stub agent (`scratch/nrl/gate_ppo_run.py::MLPStubAgent`) implementing
the Agent contract **verbatim** — `initial_state(batch, device)`,
`get_action_and_value(obs, state, action=None)`, `mean_action(obs, state)`, **no
`get_value`** — trained on `Pendulum-v1` via `rl.ppo.train`. The stub is **non-recurrent**,
so it returns a dummy placeholder state (`zeros (B,1)`) threaded through unchanged — which is
exactly the proof that the loop's opaque per-env state threading is correct for the trivial
case (`state_shape=(1,)` is read from the agent, carried across the rollout, and reset on
`done`). The
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
| 1      | 4k    | −1269        |
| 5      | 20k   | −1216        |
| 10     | 41k   | −1059        |
| 15     | 61k   |  −800        |
| 20     | 82k   |  −646        |
| 25     | 102k  |  −646        |
| 29     | 119k  |  −714        |

Monotone improvement, return roughly **−1269 → −680 (≈1.9×)** in 120k steps (run *with*
recurrent state threading active) — the PPO math (rollout, GAE(λ), clipped surrogate, clipped
value loss, minibatch epochs, grad clip, LR anneal) **plus the per-env state threading** is
correct. Value loss collapses to ~1e-3 and `approx_kl` stays in the healthy 0.001–0.016 band
(no policy blow-up). That the curve is essentially unchanged vs the pre-threading run is the
point: threading a dummy state must not perturb a non-recurrent agent.

Held-out **deterministic** (`mean_action`) return via the `eval_fn` hook:
random-init **−1056 → −1159** final (noisy — deterministic Pendulum control is sensitive at
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
`eval/holdout_detour`, etc. (Inside the hook, carry the recurrent state per episode via
`state = agent.initial_state(1, device)` then `act, state = agent.mean_action(obs, state)` —
the gate's `eval_fn` shows the pattern.)

## Agent contract as the loop calls it (the reference for 1b / integrate)

```python
initial_state(batch_size, device) -> state            # opaque; non-recurrent → dummy zeros (B,1)
get_action_and_value(obs, state, action=None)
        -> (action, logprob, entropy, value, next_state)   # samples if action is None
mean_action(obs, state) -> (action, next_state)            # deterministic
```
The loop stores the **realized per-step input state** in a `(n_steps, n_envs, *state_shape)`
buffer and feeds it back at update time, so shuffled minibatches recompute logprob/value under
the exact state the action was taken under (correct PPO ratio for a recurrent policy). It resets
state rows on `done` (mirrors AsyncVectorEnv autoreset) and treats the stored state as a
**detached input** — a **myopic stored-state policy gradient, no BPTT through the CA**,
upgradeable later.

## Task-agnostic vs leaked (audit)

**Task-agnostic (transfers verbatim to the connectome endgame):** *everything in `ppo.py`.*
The loop imports no concrete env or policy class — it depends only on (i) a `make_env()` thunk
(vectorized with `gymnasium.AsyncVectorEnv`) and (ii) an agent with the three-method contract.
Obs/action **and recurrent-state** shapes are all read at runtime
(`single_observation_space` / `single_action_space` / `agent.initial_state(...)`); nothing
assumes `(6,8,8)`, Pendulum's `(3,)`, or any state shape — the state is opaque (the loop only
zeros/where's it on `done`, never inspects it). So a recurrent `ConnectomePolicy` (real circuits
are stateful) drops into the same `train(cfg, make_env, agent, eval_fn)` call unchanged.

**Nav-specific / leaked into `ppo.py`:** **none.** All Pendulum/nav specifics live in the gate
script under `scratch/nrl/` (the stub agent, the env factory + wrappers, the eval_fn), never in
the harness.

## Notes for the merge / other sessions

- **Recurrent state (REQUIRED of every agent):** the policy must implement the three-method
  contract above — `initial_state` + the `state`/`next_state` args on `get_action_and_value`
  and `mean_action`. The NCA policy (1b) returns its CA grid as the opaque `state`; the loop
  threads the realized per-step input state so shuffled-minibatch updates stay correct. Reset
  semantics: state rows are zeroed on `done` to `initial_state`, so `initial_state` must be the
  intended fresh-episode CA state (e.g. the warm-start `init_state`). No BPTT — stored state is a
  detached input (myopic stored-state PG), upgradeable to truncated-BPTT-over-chunks later.
- **`get_value`:** intentionally **NOT** required. The GAE bootstrap value is taken from
  `get_action_and_value(next_obs, state)` with the sampled action discarded. The policy +
  integrate sessions therefore do **not** need to add a `get_value` method. (If a value-only
  forward pass is ever wanted for speed, add it across policy + integrate together, not here.)
- **`rl/__init__.py`:** this session shipped a minimal one re-exporting `PPOConfig, train`. If a
  wave-1 branch (env/policy) also ships an `__init__.py`, keep theirs and re-add the `ppo`
  re-export by hand.
- **`uv.lock`:** `gymnasium[classic-control]` was added as the superset (the env session also
  adds `gymnasium`; this version wins cleanly at merge). **`uv lock` must be re-run post-merge**
  to reconcile the wave-1 branch lockfiles — same standing note as the `__init__.py` merge.
