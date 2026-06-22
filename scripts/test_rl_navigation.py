"""N-RL-2 smoke test: env + NCA policy + one real PPO update step run clean.

This is the integration smoke test for the wired N-RL stack — it proves the three
wave-1 modules (``NavRLEnv`` + ``NCAPolicy`` + ``rl.ppo.train``) actually fit
together end to end, not just in isolation. It is deliberately tiny (a few env
steps, one PPO update) so it runs in well under a minute; the heavy four-gate
calibration lives in ``scripts/run_rl_navigation.py --calibrate``.

What it checks:
  1. ``NavRLEnv`` builds with the calibration config (w_collide=0.25, Newton cap
     dropped); obs/action spaces match the contract; a random rollout stays in
     space with all info keys and bounded ``ncon``.
  2. ``NCAPolicy`` warm-starts from the chemo forager and implements the cleanrl
     agent contract against the env's real obs (initial_state / get_action_and_value
     / mean_action), with shapes as advertised.
  3. ``rl.ppo.train`` runs a full update on the vectorized nav env + warm-started
     policy without raising, calling the held-out ``eval_fn`` once.

Requires flygym + the chemo controller JSON (WIN/sentry). Run from repo root:

    uv run python scripts/test_rl_navigation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cellular_gaits.rl.nav_env import NavRLConfig, NavRLEnv, make_nav_env  # noqa: E402
from cellular_gaits.rl.policies import NCAPolicy  # noqa: E402
from cellular_gaits.rl.ppo import PPOConfig, train  # noqa: E402

# Import the shared wiring helpers (warm start + held-out eval) so the smoke test
# exercises the SAME glue the calibration uses.
sys.path.insert(0, str(ROOT / "scripts"))
from run_rl_navigation import (  # noqa: E402
    CALIBRATION_W_COLLIDE,
    build_warm_started_policy,
    make_eval_fn,
)

NCON_BOUND = 25


def _ok(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def check_env(cfg: NavRLConfig) -> bool:
    print("\n[1] env builds + random rollout in-space, ncon bounded")
    env = NavRLEnv(cfg)
    obs, info = env.reset(seed=0)
    ok = True
    ok &= _ok("obs shape (6,8,8) float32", obs.shape == (6, 8, 8) and obs.dtype == np.float32,
              f"{obs.shape} {obs.dtype}")
    ok &= _ok("obs in observation_space", env.observation_space.contains(obs))
    ok &= _ok("action_space (42,)", env.action_space.shape == (42,))
    ok &= _ok("Newton cap dropped (solver=2/Newton, iters=100)",
              int(env.fly.sim.mj_model.opt.solver) == 2
              and int(env.fly.sim.mj_model.opt.iterations) == 100)
    rng = np.random.default_rng(0)
    ncon_peak = 0
    keys = None
    steps = 0
    for _ in range(60):
        a = env.action_space.sample().astype(np.float32) * 0.0  # zero action = stand
        a += rng.normal(0, 0.05, size=42).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        ncon_peak = max(ncon_peak, int(env.fly.sim.mj_data.ncon))
        keys = set(info.keys())
        steps += 1
        if term or trunc:
            break
    ok &= _ok("obs stays in space after stepping", env.observation_space.contains(obs))
    ok &= _ok("reward finite", np.isfinite(r), f"r={r:.4f}")
    expected_keys = {"collision", "dist_to_goal", "reached", "fell", "detour_perp",
                     "feeler_L", "feeler_R", "goal_bearing", "ever_collided", "approach"}
    ok &= _ok("info keys present", expected_keys.issubset(keys), str(sorted(keys or [])))
    ok &= _ok(f"ncon bounded <= {NCON_BOUND}", ncon_peak <= NCON_BOUND,
              f"ncon_peak={ncon_peak} over {steps} steps")
    env.close()
    return ok


def check_policy(cfg: NavRLConfig) -> bool:
    print("\n[2] NCA policy warm-start + agent contract on real obs")
    env = NavRLEnv(cfg)
    obs, _ = env.reset(seed=1)
    policy, ws = build_warm_started_policy(cfg)
    ok = True
    ok &= _ok("warm start nav params == 1524", ws["nav_params"] == 1524, str(ws))
    device = torch.device("cpu")
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    state = policy.initial_state(1, device)
    ok &= _ok("initial_state (1,4,8,8)", tuple(state.shape) == (1, 4, 8, 8), str(tuple(state.shape)))
    action, logprob, entropy, value, next_state = policy.get_action_and_value(obs_t, state)
    ok &= _ok("get_action_and_value shapes",
              tuple(action.shape) == (1, 42) and tuple(logprob.shape) == (1,)
              and tuple(next_state.shape) == (1, 4, 8, 8),
              f"act={tuple(action.shape)} lp={tuple(logprob.shape)} ns={tuple(next_state.shape)}")
    ok &= _ok("next_state detached", not next_state.requires_grad)
    mean_a, _ = policy.mean_action(obs_t, state)
    ok &= _ok("mean_action in [-1,1]", bool((mean_a.abs() <= 1.0 + 1e-6).all()))
    env.close()
    return ok


def check_ppo_update(cfg: NavRLConfig) -> bool:
    print("\n[3] one real PPO update on the vectorized nav env (eval_fn called once)")
    policy, _ = build_warm_started_policy(cfg)
    # Tiny budget: 2 envs x 16 steps = 32-sample batch, exactly one update, one eval.
    ppo = PPOConfig(
        exp_name="nrl_smoke",
        seed=0,
        device="cpu",
        total_steps=32,
        n_envs=2,
        n_steps=16,
        n_minibatch=2,
        update_epochs=1,
        eval_every=1,
        ckpt_every=0,
        log_every=1,
        wandb=False,
        behavior="nav",
        ckpt_dir=str(ROOT / "scratch" / "nrl" / "ckpt_smoke"),
    )
    eval_fn = make_eval_fn(cfg, n_episodes=2, device="cpu")
    try:
        train(ppo, lambda: make_nav_env(cfg), policy, eval_fn=eval_fn)
        return _ok("PPO update + eval ran clean", True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return _ok("PPO update + eval ran clean", False, str(exc))


def main() -> int:
    cfg = NavRLConfig(w_collide=CALIBRATION_W_COLLIDE, drop_solver_cap=True)
    print(f"N-RL-2 smoke test  (w_collide={cfg.w_collide}, drop_solver_cap={cfg.drop_solver_cap})")
    results = [check_env(cfg), check_policy(cfg), check_ppo_update(cfg)]
    passed = all(results)
    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {'ALL PASS' if passed else 'FAILURES'} ({sum(results)}/{len(results)})")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
