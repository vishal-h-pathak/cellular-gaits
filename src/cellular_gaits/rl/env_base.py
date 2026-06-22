"""``EmbodiedRLEnv`` — a policy-agnostic Gymnasium wrapper over ``FlyEnv``.

This is the reusable half of the N-RL harness. It turns the existing
:class:`cellular_gaits.env.FlyEnv` (built for deterministic CMA-ES rollouts)
into an episodic :class:`gymnasium.Env`, with the *task* factored out into four
pluggable callables passed at construction:

    (i)   ``sample_layout(rng, eval_mode, eval_index) -> layout``
          domain-randomization sampler, called on ``reset()``. ``layout`` is an
          arbitrary task-defined object (a dict here) describing one episode's
          world (e.g. obstacle positions + goal). ``eval_mode`` selects a fixed,
          held-out distribution; ``eval_index`` counts eval resets so a frozen
          held-out *set* can be cycled deterministically.
    (ii)  ``build_fly_env(layout) -> FlyEnv``
          constructs a fresh ``FlyEnv`` for that layout. A fresh env per reset is
          required because obstacle geoms are baked into the MuJoCo world at
          construction (see ``FlyEnv._build_default_world``); this is also where
          the Newton<=20 / ls 10 contact cap is applied — automatically, by
          ``FlyEnv.__init__``, ONLY when obstacles are present, so chemo / loom /
          closed-loop / v1 physics stay byte-exact.
    (iii) ``build_obs(fly, layout, ctx) -> np.ndarray``
          observation builder, read from the *current* sim state (so obs_t is the
          body state the policy reacts to before issuing action_t).
    (iv)  ``map_action(action) -> np.ndarray (N_ACTUATORS,)``
          action->actuator map: turns a policy action into the 42-vector
          ``FlyEnv.step`` expects (which itself clips to [-1,1] and rescales to
          the ctrlrange — that IS the actuator map, reused verbatim).
    plus  ``compute_reward(fly, layout, ctx, step_reward, fly_obs, action)
            -> (reward, terminated, info)``
          per-step reward + task-termination + diagnostics, and an optional
          ``init_ctx(fly, layout) -> dict`` that seeds a per-episode mutable
          context shared between ``build_obs`` and ``compute_reward``.

``terminated`` is the task-terminal flag (goal reached / fell). ``truncated`` is
owned here: it fires when ``max_episode_steps`` control steps have elapsed.

Nothing in this module is nav-specific: ``NavRLEnv`` supplies the callables.
A future ``ConnectomePolicy`` task would supply its own and reuse this verbatim.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from ..env import FlyEnv, N_ACTUATORS

# Pluggable-hook type aliases (documentation only — kept loose on purpose).
Layout = Any
SampleLayout = Callable[[np.random.Generator, bool, int], Layout]
BuildFlyEnv = Callable[[Layout], FlyEnv]
BuildObs = Callable[[FlyEnv, Layout, dict], np.ndarray]
MapAction = Callable[[np.ndarray], np.ndarray]
ComputeReward = Callable[..., tuple[float, bool, dict]]
InitCtx = Callable[[FlyEnv, Layout], dict]


class EmbodiedRLEnv(gym.Env):
    """Episodic Gymnasium wrapper over a ``FlyEnv`` body (policy-agnostic)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        sample_layout: SampleLayout,
        build_fly_env: BuildFlyEnv,
        build_obs: BuildObs,
        map_action: MapAction,
        compute_reward: ComputeReward,
        init_ctx: Optional[InitCtx] = None,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._sample_layout = sample_layout
        self._build_fly_env = build_fly_env
        self._build_obs = build_obs
        self._map_action = map_action
        self._compute_reward = compute_reward
        self._init_ctx = init_ctx
        self.max_episode_steps = int(max_episode_steps)

        # Per-episode run-state (populated on reset()).
        self.fly: Optional[FlyEnv] = None
        self.layout: Optional[Layout] = None
        self.ctx: dict = {}
        self._elapsed: int = 0
        self._eval_index: int = 0  # number of eval-mode resets so far

    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        # Seeds self.np_random the first time (or whenever seed is given); the
        # stream then persists across resets so successive episodes draw
        # different layouts while staying deterministic given the initial seed.
        super().reset(seed=seed)
        eval_mode = bool(options.get("eval", False)) if options else False

        layout = self._sample_layout(self.np_random, eval_mode, self._eval_index)
        if eval_mode:
            self._eval_index += 1

        # Build a fresh FlyEnv for this layout (obstacle geoms baked in -> the
        # Newton cap is applied here, obstacles-only) and run the standard
        # reset+warmup so the body settles before the first observation.
        self.fly = self._build_fly_env(layout)
        self.fly.reset()
        self.layout = layout
        self.ctx = self._init_ctx(self.fly, layout) if self._init_ctx else {}
        self._elapsed = 0

        obs = np.asarray(
            self._build_obs(self.fly, layout, self.ctx),
            dtype=self.observation_space.dtype,
        )
        info: dict = {"layout": layout, "eval": eval_mode}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.fly is None:
            raise RuntimeError("step() called before reset()")
        targets = np.asarray(self._map_action(action), dtype=np.float64).reshape(-1)
        if targets.size != N_ACTUATORS:
            raise ValueError(
                f"map_action must return {N_ACTUATORS} targets, got {targets.size}"
            )
        # FlyEnv.step does the actuator set + PHYSICS_PER_CONTROL physics ticks
        # and returns (obs_dict, StepReward, done=False, {}).
        fly_obs, step_reward, _, _ = self.fly.step(targets)
        self._elapsed += 1

        reward, terminated, info = self._compute_reward(
            self.fly, self.layout, self.ctx, step_reward, fly_obs, action
        )
        truncated = self._elapsed >= self.max_episode_steps

        obs = np.asarray(
            self._build_obs(self.fly, self.layout, self.ctx),
            dtype=self.observation_space.dtype,
        )
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def close(self) -> None:  # noqa: D401 - Gymnasium hook
        self.fly = None
