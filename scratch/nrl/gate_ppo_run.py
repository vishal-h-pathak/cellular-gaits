"""PPO loop correctness gate — does the loop *learn* on a standard control env?

This validates the PPO math independently of the fly sim: a tiny MLP stub agent
(implementing the *exact* Agent interface the policy session must implement) trained
on Pendulum-v1 should see episodic return rise over a short budget.

The stub agent here is the REFERENCE the policy + integrate sessions check against:
only ``get_action_and_value(obs, action=None)`` and ``mean_action(obs)`` — no
``get_value``. Run:  uv run python scratch/nrl/gate_ppo_run.py
"""

from __future__ import annotations

import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from cellular_gaits.rl.ppo import PPOConfig, train

ENV_ID = "Pendulum-v1"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPStubAgent(nn.Module):
    """A tiny continuous-control Gaussian-policy + value MLP.

    Implements the Agent contract VERBATIM — the loop depends on nothing else:
      - get_action_and_value(obs, action=None) -> (action, logprob, entropy, value)
      - mean_action(obs) -> action
    The value bootstrap in the loop comes from get_action_and_value(obs) (action
    discarded), so there is intentionally no get_value method here.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        return Normal(mean, logstd.exp())

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        dist = self._dist(obs)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(obs)
        return action, logprob, entropy, value

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(obs)


def make_env() -> gym.Env:
    """Env factory for the gate. The standard cleanrl wrappers (action clip,
    obs/reward normalization) live HERE in the factory — the PPO loop stays
    env-agnostic. RecordEpisodeStatistics gives the loop its return curve."""
    env = gym.make(ENV_ID)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda o: np.clip(o, -10, 10), env.observation_space
    )
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    return env


def eval_fn(agent: nn.Module) -> dict:
    """Pluggable held-out eval hook: deterministic (mean_action) return on a fresh
    seed, NOT reward-normalized — the honest yardstick. Demonstrates the hook the
    integrate session wires nav's detour/reach into."""
    device = next(agent.parameters()).device
    env = gym.make(ENV_ID)
    returns = []
    with torch.no_grad():
        for ep in range(5):
            obs, _ = env.reset(seed=10_000 + ep)
            done = False
            total = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act = agent.mean_action(obs_t).squeeze(0).cpu().numpy()
                obs, r, term, trunc, _ = env.step(np.clip(act, -2.0, 2.0))
                total += float(r)
                done = term or trunc
            returns.append(total)
    env.close()
    return {"holdout_return": float(np.mean(returns))}


def main() -> None:
    cfg = PPOConfig(
        exp_name="gate_pendulum",
        seed=1,
        device="cpu",            # gate runs CPU-only on the Mac (CUDA on the workstation).
        total_steps=120_000,     # short budget — enough to show the curve rises.
        n_envs=4,
        n_steps=1024,
        n_minibatch=32,
        update_epochs=10,
        gamma=0.9,               # Pendulum is short-horizon.
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=1e-3,
        ckpt_dir="scratch/nrl/ckpt",
        ckpt_every=10,
        eval_every=5,
        log_every=1,
    )

    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()

    agent = MLPStubAgent(obs_dim, act_dim)

    # capture the learning curve via a wrapping eval at start + the loop's own logs
    print(f"[gate] {ENV_ID} obs_dim={obs_dim} act_dim={act_dim} updates={cfg.num_updates}")
    baseline = eval_fn(agent)["holdout_return"]
    print(f"[gate] random-init holdout return: {baseline:.1f}")

    train(cfg, make_env, agent, eval_fn=eval_fn)

    final = eval_fn(agent)["holdout_return"]
    print(f"[gate] trained holdout return: {final:.1f}")

    os.makedirs("scratch/nrl", exist_ok=True)
    with open("scratch/nrl/gate_ppo_result.json", "w") as f:
        json.dump({"env": ENV_ID, "baseline_holdout_return": baseline,
                   "final_holdout_return": final, "config": cfg.exp_name}, f, indent=2)
    print("[gate] wrote scratch/nrl/gate_ppo_result.json")


if __name__ == "__main__":
    main()
