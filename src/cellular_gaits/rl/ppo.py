"""PPO — a clean, single-file, task-agnostic training loop.

This is the reusable connectome harness's training core. It is **agnostic to both
the env and the policy**: it takes an env factory (``make_env``) and an agent
(``torch.nn.Module``), so a future ``ConnectomePolicy`` + a different env drop in
unchanged. cleanrl's ``ppo_continuous_action`` is the reference implementation; this
generalizes it (no hard-coded obs/action shapes, no concrete env/agent imports).

What lives here (transfers verbatim to the connectome endgame):
  - rollout collection over ``gymnasium.AsyncVectorEnv`` (N CPU envs — sim is CPU-bound),
  - GAE(λ) advantage estimation,
  - the clipped PPO surrogate + clipped value loss + entropy bonus,
  - minibatch epochs, global-norm grad clipping, linear LR anneal, optional KL early-stop,
  - GPU for the policy/update (CUDA on the workstation), CPU for the vec envs,
  - checkpoint + resume, optional W&B logging, and a pluggable ``eval_fn(agent)`` hook.

What is NOT here (stays task-specific, by design):
  - the env (observation builder, action→actuator map, reward, domain randomization)
    — supplied via ``make_env``,
  - the policy (architecture, warm-start) — supplied as ``agent``.

Agent contract (the cleanrl interface — exactly two methods, nothing more):
  - ``get_action_and_value(obs, action=None) -> (action, logprob, entropy, value)``
    samples when ``action is None``, else scores the given ``action``. ``value`` is
    shape ``(B,)`` or ``(B, 1)``.
  - ``mean_action(obs) -> action`` — deterministic action (eval / A/B).

The GAE bootstrap value is taken from ``get_action_and_value(obs)`` (action discarded)
— this loop deliberately does **not** depend on a separate ``get_value`` method, so the
policy session does not need to implement one. (If a value-only forward pass is ever
wanted for speed, it must be added across the policy + integrate sessions together.)
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Type aliases for the two protocols this loop depends on (structural, not nominal).
MakeEnv = Callable[[], gym.Env]
EvalFn = Callable[[nn.Module], dict]


@dataclass
class PPOConfig:
    """Knobs for the PPO loop. Defaults are sane for cheap continuous-control;
    the integrate session overrides ``n_envs`` (~16 on the 5900X), ``total_steps``,
    and the curriculum-driven schedule for the fly sim."""

    # --- run identity / bookkeeping ---
    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "auto"  # "auto" -> cuda if available else cpu; or "cuda"/"cpu"/"mps".

    # --- budget & rollout shape ---
    total_steps: int = 1_000_000  # total env steps across all parallel envs.
    n_envs: int = 8               # parallel AsyncVectorEnv workers (CPU). ~16 on the workstation.
    n_steps: int = 2048           # rollout length per env per update.
    n_minibatch: int = 32         # minibatches per epoch.
    update_epochs: int = 10       # passes over each rollout.

    # --- PPO objective ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True       # cleanrl-style clipped value loss.
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True         # per-minibatch advantage normalization.
    target_kl: Optional[float] = None  # early-stop epochs when approx_kl exceeds this.

    # --- optimizer / schedule ---
    learning_rate: float = 3e-4
    anneal_lr: bool = True        # linear decay to 0 over the run.
    eps: float = 1e-5             # Adam epsilon (cleanrl default).

    # --- checkpoint / eval / logging cadence (in *updates*) ---
    ckpt_dir: str = "scratch/nrl/ckpt"
    ckpt_every: int = 25          # save a checkpoint every N updates (0 = never).
    eval_every: int = 25          # call eval_fn every N updates (0 = never).
    log_every: int = 1            # console/W&B log every N updates.

    # --- Weights & Biases (lazy, optional, off by default, soft-fail) ---
    wandb: bool = False
    wandb_project: str = "cellular-gaits"
    wandb_run_id: Optional[str] = None
    wandb_tags: tuple = field(default_factory=tuple)
    behavior: str = "harness"     # behavior tag (walk/chemo/escape/nav/...); per W&B convention.

    # --- derived (filled in __post_init__) ---
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_updates: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.batch_size = int(self.n_envs * self.n_steps)
        self.minibatch_size = int(self.batch_size // self.n_minibatch)
        self.num_updates = int(self.total_steps // self.batch_size)


# --------------------------------------------------------------------------- #
# Device & vec-env helpers
# --------------------------------------------------------------------------- #
def resolve_device(spec: str) -> torch.device:
    """Auto-select the update device: CUDA on the workstation, else MPS/CPU.
    The vec envs always run on CPU regardless (the sim is CPU-bound)."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_vector_env(make_env: MakeEnv, n_envs: int) -> gym.vector.VectorEnv:
    """Build an AsyncVectorEnv (N parallel CPU envs) from a thunk.

    ``make_env`` is a zero-arg factory returning a fresh ``gym.Env`` — the env
    session supplies the fly env; the gate supplies a classic-control env. The
    loop never imports a concrete env class."""
    return gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])


# --------------------------------------------------------------------------- #
# W&B — lazy, optional, soft-fail (mirrors PROMPT_wandb_integration.md convention)
# --------------------------------------------------------------------------- #
class _WandbRun:
    """Tiny lazy W&B wrapper: a no-op unless enabled AND importable AND init succeeds.
    A tracker outage must never kill a multi-hour run, so every path fails soft."""

    def __init__(self, cfg: PPOConfig):
        self.enabled = False
        self._wandb = None
        if not cfg.wandb:
            return
        try:
            import wandb  # lazy: not required for normal runs.

            tags = [socket.gethostname(), cfg.behavior, "rl", "ppo", *cfg.wandb_tags]
            self._wandb = wandb
            self._wandb.init(
                project=cfg.wandb_project,
                id=cfg.wandb_run_id,
                name=cfg.wandb_run_id or cfg.exp_name,
                config=asdict(cfg),
                tags=tags,
                resume="allow",
            )
            self.enabled = True
        except Exception as exc:  # import error, no login, network down — warn & continue.
            print(f"[ppo] W&B disabled (soft-fail): {exc}")
            self.enabled = False

    def log(self, data: dict, step: int) -> None:
        if not self.enabled:
            return
        try:
            self._wandb.log(data, step=step)
        except Exception as exc:
            print(f"[ppo] W&B log failed (soft-fail): {exc}")
            self.enabled = False

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, agent: nn.Module, optimizer: optim.Optimizer,
                    global_step: int, update: int, cfg: PPOConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "update": update,
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(path: str, agent: nn.Module, optimizer: Optional[optim.Optimizer],
                    device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# --------------------------------------------------------------------------- #
# The PPO loop
# --------------------------------------------------------------------------- #
def train(
    cfg: PPOConfig,
    make_env: MakeEnv,
    agent: nn.Module,
    eval_fn: Optional[EvalFn] = None,
    resume_from: Optional[str] = None,
) -> nn.Module:
    """Train ``agent`` on the vectorized env from ``make_env`` with PPO.

    Args:
        cfg: hyperparameters (see ``PPOConfig``).
        make_env: zero-arg factory -> fresh ``gym.Env`` (vectorized N=cfg.n_envs times).
        agent: an ``nn.Module`` implementing ``get_action_and_value`` + ``mean_action``.
        eval_fn: optional ``eval_fn(agent) -> dict`` of held-out metrics, called every
            ``cfg.eval_every`` updates and logged under ``eval/*`` (the integrate session
            wires nav's held-out detour/reach here).
        resume_from: optional checkpoint path to resume agent+optimizer+step from.

    Returns:
        the trained ``agent`` (also checkpointed under ``cfg.ckpt_dir``).
    """
    # --- reproducibility ---
    random_seed(cfg.seed, cfg.torch_deterministic)
    device = resolve_device(cfg.device)
    print(f"[ppo] device={device} n_envs={cfg.n_envs} batch={cfg.batch_size} "
          f"minibatch={cfg.minibatch_size} updates={cfg.num_updates}")

    # --- envs (CPU) + agent (device) ---
    envs = make_vector_env(make_env, cfg.n_envs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), \
        "PPO loop is for continuous (Box) action spaces."
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=cfg.eps)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape

    # --- rollout storage ---
    obs = torch.zeros((cfg.n_steps, cfg.n_envs, *obs_shape), device=device)
    actions = torch.zeros((cfg.n_steps, cfg.n_envs, *act_shape), device=device)
    logprobs = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    rewards = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    dones = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    values = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)

    global_step = 0
    start_update = 1
    if resume_from is not None and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from, agent, optimizer, device)
        global_step = int(ckpt.get("global_step", 0))
        start_update = int(ckpt.get("update", 0)) + 1
        print(f"[ppo] resumed from {resume_from} @ update={start_update - 1} step={global_step}")

    wandb_run = _WandbRun(cfg)
    start_time = time.time()

    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(np.asarray(next_obs_np), dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.n_envs, device=device)

    # rolling episode-stat trackers (RecordEpisodeStatistics-compatible info parsing)
    recent_returns: list[float] = []
    recent_lengths: list[float] = []

    for update in range(start_update, cfg.num_updates + 1):
        # linear LR anneal
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / cfg.num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ---------------- rollout ----------------
        for step in range(cfg.n_steps):
            global_step += cfg.n_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _entropy, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            action_np = action.cpu().numpy()
            next_obs_np, reward, terminations, truncations, infos = envs.step(action_np)
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(np.asarray(reward), dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(np.asarray(next_obs_np), dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np.astype(np.float32), device=device)

            _collect_episode_stats(infos, recent_returns, recent_lengths)

        # ---------------- GAE(λ) ----------------
        # bootstrap value: get_action_and_value(obs) and discard the sampled action
        # (deliberately no get_value dependency — see module docstring).
        with torch.no_grad():
            _a, _lp, _e, next_value = agent.get_action_and_value(next_obs)
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # ---------------- flatten the batch ----------------
        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *act_shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---------------- PPO update ----------------
        b_inds = np.arange(cfg.batch_size)
        clipfracs: list[float] = []
        approx_kl = torch.tensor(0.0)
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # clipped policy surrogate
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (optionally clipped, cleanrl-style)
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl.item() > cfg.target_kl:
                break

        # explained variance (value-fit diagnostic)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ---------------- eval hook ----------------
        eval_metrics: dict = {}
        if eval_fn is not None and cfg.eval_every and update % cfg.eval_every == 0:
            agent.eval()
            with torch.no_grad():
                eval_metrics = eval_fn(agent) or {}
            agent.train()

        # ---------------- logging ----------------
        if cfg.log_every and update % cfg.log_every == 0:
            sps = int(global_step / (time.time() - start_time))
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            mean_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
            metrics = {
                "charts/episodic_return": mean_return,
                "charts/episodic_length": mean_length,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": sps,
                "losses/value_loss": float(v_loss.item()),
                "losses/policy_loss": float(pg_loss.item()),
                "losses/entropy": float(entropy_loss.item()),
                "losses/approx_kl": float(approx_kl.item()),
                "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "losses/explained_variance": explained_var,
            }
            for k, v in eval_metrics.items():
                metrics[f"eval/{k}"] = v
            wandb_run.log(metrics, step=global_step)
            print(f"[ppo] update={update}/{cfg.num_updates} step={global_step} "
                  f"return={mean_return:.2f} v_loss={v_loss.item():.3f} "
                  f"kl={approx_kl.item():.4f} SPS={sps}"
                  + (f" eval={eval_metrics}" if eval_metrics else ""))
            # reset the rolling window each log so the curve tracks recent performance
            recent_returns.clear()
            recent_lengths.clear()

        # ---------------- checkpoint ----------------
        if cfg.ckpt_every and update % cfg.ckpt_every == 0:
            path = os.path.join(cfg.ckpt_dir, f"{cfg.exp_name}_update{update}.pt")
            save_checkpoint(path, agent, optimizer, global_step, update, cfg)
            latest = os.path.join(cfg.ckpt_dir, f"{cfg.exp_name}_latest.pt")
            save_checkpoint(latest, agent, optimizer, global_step, update, cfg)

    # final checkpoint
    final_path = os.path.join(cfg.ckpt_dir, f"{cfg.exp_name}_final.pt")
    save_checkpoint(final_path, agent, optimizer, global_step, cfg.num_updates, cfg)
    print(f"[ppo] done. final checkpoint -> {final_path}")

    envs.close()
    wandb_run.finish()
    return agent


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def random_seed(seed: int, torch_deterministic: bool) -> None:
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def _collect_episode_stats(infos: dict, returns: list[float], lengths: list[float]) -> None:
    """Parse episode returns from a vectorized ``RecordEpisodeStatistics`` info dict.

    Gymnasium's vector wrapper reports finished episodes under ``infos["episode"]``
    with a boolean ``_episode`` mask. No-op when the wrapper isn't present, so the
    loop still runs (it just won't have return curves) — the gate wraps with it."""
    if not isinstance(infos, dict) or "episode" not in infos:
        return
    ep = infos["episode"]
    mask = infos.get("_episode")
    r = np.asarray(ep["r"]).reshape(-1)
    length = np.asarray(ep["l"]).reshape(-1)
    if mask is None:
        mask = np.ones_like(r, dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1)
    for i in range(r.shape[0]):
        if mask[i]:
            returns.append(float(r[i]))
            lengths.append(float(length[i]))
