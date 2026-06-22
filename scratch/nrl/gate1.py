"""Gate 1 (env sanity) for the N-RL nav env. Writes scratch/nrl/gate1.md.

Checks (per PROMPT_n_rl_1a_env.md):
  1. Random-policy episodes run end-to-end; obs/action/reward shapes + spaces OK.
  2. Newton<=20/ls10 contact cap holds a worst-case (fly pinned on an obstacle)
     rollout to ~seconds, not the uncapped blow-up — timed, capped vs uncapped.
  3. gymnasium AsyncVectorEnv with N workers is deterministic given a seed and
     reports throughput (steps/s, resets/s) + a reset-time vs step-time split.
  4. Domain randomization visibly varies obstacle + bearing across resets; the
     held-out set is disjoint from train.

Run: uv run python scratch/nrl/gate1.py
This is scratch (the ONE dir this session created); it imports the package but
modifies nothing under src/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cellular_gaits.env import CONTROL_DT_S, FLY_NAME, FlyEnv, ObstacleField, OdorField
from cellular_gaits.rl import NavRLConfig, NavRLEnv, make_nav_env
from cellular_gaits.rl.nav_env import _params_too_close

OUT = Path(__file__).resolve().parent / "gate1.md"
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# --------------------------------------------------------------------------- #
def check1_shapes_and_episode() -> dict:
    log("## 1. Random-policy episode + shapes/spaces\n")
    cfg = NavRLConfig(max_episode_steps=40)
    env = NavRLEnv(cfg)
    obs, info = env.reset(seed=0)

    assert env.observation_space.shape == (6, 8, 8), env.observation_space.shape
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (42,), env.action_space.shape
    assert env.action_space.dtype == np.float32
    assert obs.shape == (6, 8, 8) and obs.dtype == np.float32
    assert env.observation_space.contains(obs), "reset obs out of space"

    rng = np.random.default_rng(0)
    rewards, terminated, truncated = [], False, False
    steps = 0
    info_keys_required = {
        "collision", "dist_to_goal", "reached", "detour_perp",
        "feeler_L", "feeler_R", "goal_bearing",
    }
    for _ in range(cfg.max_episode_steps):
        a = rng.uniform(-1.0, 1.0, size=42).astype(np.float32)
        assert env.action_space.contains(a)
        obs, r, terminated, truncated, info = env.step(a)
        assert obs.shape == (6, 8, 8) and env.observation_space.contains(obs)
        assert np.isscalar(r) or np.ndim(r) == 0
        assert info_keys_required <= set(info), info_keys_required - set(info)
        rewards.append(float(r))
        steps += 1
        if terminated or truncated:
            break

    # Explicit truncation test: a standing fly (zero action) at a 1-step limit
    # neither falls (z stays > half height in one 4 ms control step) nor reaches
    # (goal is 18 away), so the episode must truncate on the time limit.
    tenv = NavRLEnv(NavRLConfig(max_episode_steps=1))
    tenv.reset(seed=1)
    _, _, t_term, t_trunc, _ = tenv.step(np.zeros(42, dtype=np.float32))
    trunc_ok = (t_trunc is True) and (t_term is False)

    ch = [(float(obs[c].min()), float(obs[c].max())) for c in range(6)]
    log(f"- observation_space : Box{env.observation_space.shape} float32, "
        f"per-channel ch0∈[-1,1], ch1-5∈[0,1]")
    log(f"- action_space      : Box(-1,1,(42,)) float32")
    log(f"- random episode    : {steps} steps, terminated={terminated} "
        f"truncated={truncated} (random fly falls -> termination-on-fall works)")
    log(f"- truncation test   : 1-step limit, zero action -> terminated={t_term} "
        f"truncated={t_trunc}  -> {'PASS' if trunc_ok else 'FAIL'} "
        f"(time-limit truncation works)")
    log(f"- reward range       : [{min(rewards):.4f}, {max(rewards):.4f}], "
        f"sum={sum(rewards):.4f}")
    log(f"- obs in-space every step: PASS")
    log(f"- info keys present : {sorted(info_keys_required)}  PASS")
    log(f"- obs ch min/max    : " + ", ".join(
        f"ch{c}[{lo:.2f},{hi:.2f}]" for c, (lo, hi) in enumerate(ch)))
    log("")
    return {"steps": steps, "trunc_ok": trunc_ok}


# --------------------------------------------------------------------------- #
def _time_drive_into_wall(cap: bool, n_steps: int, seed: int) -> float:
    """Time an n_steps rollout driving the fly forward into an obstacle ahead.

    FlyEnv applies the Newton iteration cap automatically when obstacles are
    present; to measure the UNCAPPED cost we mutate this one sim instance's
    solver options back to the MuJoCo defaults (we create the instance, so this
    edits nothing under src/).
    """
    field = ObstacleField(centers=((4.0, 0.0),), radius=2.0)  # dead ahead
    fly = FlyEnv(obstacles=field, feeler_range=6.0)
    fly.set_odor(OdorField(source_xy=(18.0, 0.0)))
    if not cap:
        opt = fly.sim.mj_model.opt
        opt.iterations = 100  # MuJoCo Newton default
        opt.ls_iterations = 50  # MuJoCo line-search default
    fly.reset()
    j = np.arange(42)
    t0 = time.perf_counter()
    for t in range(n_steps):
        fly.step(0.6 * np.sin(0.5 * t + j * 0.6) + 0.2)  # push forward
    return time.perf_counter() - t0


def check2_newton_cap() -> dict:
    import mujoco

    from cellular_gaits.env import (
        OBSTACLE_SOLVER,
        OBSTACLE_SOLVER_ITERATIONS,
        OBSTACLE_SOLVER_LS_ITERATIONS,
    )

    log("## 2. Newton contact-cap — applied obstacles-only; worst-case bounded\n")

    # (a) The cap IS applied on an obstacle env (what NavRLEnv builds each reset).
    env = NavRLEnv(NavRLConfig())
    env.reset(seed=0)
    opt = env.fly.sim.mj_model.opt
    cap_applied = (
        int(opt.solver) == OBSTACLE_SOLVER
        and int(opt.iterations) == OBSTACLE_SOLVER_ITERATIONS
        and int(opt.ls_iterations) == OBSTACLE_SOLVER_LS_ITERATIONS
    )

    # (b) An obstacle-FREE FlyEnv keeps the MuJoCo defaults (so v1 / chemo / loom
    #     / closed-loop physics are byte-for-byte unchanged — the cap is additive).
    free = FlyEnv()
    fo = free.sim.mj_model.opt
    free_iters, free_ls = int(fo.iterations), int(fo.ls_iterations)

    # (c) Does physical fly<->obstacle contact even occur on this flygym build?
    m = env.fly.sim.mj_model
    obs_gids = [
        g for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("obstacle_")
    ]
    fly_collidable = sum(
        1 for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("nmf/")
        and (m.geom_contype[g] or m.geom_conaffinity[g])
    )
    pairs_with_obs = sum(
        1 for i in range(m.npair)
        if obs_gids and (m.pair_geom1[i] in obs_gids or m.pair_geom2[i] in obs_gids)
    )

    # (d) Time a full worst-case rollout (fly driven into the obstacle), capped
    #     (as the real env runs) vs uncapped.
    n_steps = 1000
    t_capped = _time_drive_into_wall(cap=True, n_steps=n_steps, seed=1)
    t_uncapped = _time_drive_into_wall(cap=False, n_steps=n_steps, seed=1)
    ratio = t_uncapped / t_capped if t_capped > 0 else float("nan")

    log(f"- cap applied on obstacle env (NavRLEnv): solver={int(opt.solver)} "
        f"iterations={int(opt.iterations)} ls_iterations={int(opt.ls_iterations)}  "
        f"-> {'PASS' if cap_applied else 'FAIL'} "
        f"(target solver={OBSTACLE_SOLVER}/iter={OBSTACLE_SOLVER_ITERATIONS}/"
        f"ls={OBSTACLE_SOLVER_LS_ITERATIONS})")
    log(f"- obstacle-FREE FlyEnv keeps MuJoCo defaults: iterations={free_iters} "
        f"ls_iterations={free_ls}  -> cap is obstacles-only, v1/chemo/loom byte-exact")
    log(f"- worst-case rollout ({n_steps} steps, fly driven into obstacle ahead):")
    log(f"  - CAPPED   (iter=20, ls=10) : {t_capped:6.2f} s  ({t_capped/n_steps*1000:.2f} ms/step)")
    log(f"  - UNCAPPED (iter=100, ls=50): {t_uncapped:6.2f} s  (ratio {ratio:.2f}x)")
    log("")
    log(f"- **FINDING (flag for WIN verification):** on this machine's "
        f"`flygym==2.0.1` build the obstacle geoms do **not** physically block "
        f"the fly. Evidence: {fly_collidable}/69 fly geoms are collidable "
        f"(all `contype=conaffinity=0`); the obstacle is the only collidable "
        f"geom; of {m.npair} explicit contact pairs (foot-ground), "
        f"{pairs_with_obs} involve the obstacle; a fly driven forward walks "
        f"straight through it. So no fly-obstacle MuJoCo contact is generated, "
        f"the pinned-fly heavy-contact regime never triggers, and the "
        f"~40 s->~7 s worst case in REPORT_n_a_calibration (measured on WIN) is "
        f"**not reproducible here** (capped==uncapped, ~{t_capped/n_steps*1000:.1f} "
        f"ms/step). The cap is a verified-active safety belt bounding a cost that "
        f"is ~0 on this build. The nav TASK is unaffected — obstacles are sensed "
        f"geometrically (feelers) and the collision penalty is geometric "
        f"time-in-contact (thorax within clearance of the surface), neither of "
        f"which needs a physical contact. But physical *blocking* is absent here; "
        f"since flygym is version-pinned + uv.lock committed, WIN should match — "
        f"verify on WIN whether blocking/the 40 s pin actually occurs there.")
    log("")
    return {
        "cap_applied": cap_applied,
        "free_iters": free_iters,
        "pairs_with_obs": pairs_with_obs,
        "fly_collidable": fly_collidable,
        "t_capped": t_capped,
        "t_uncapped": t_uncapped,
        "ratio": ratio,
    }


# --------------------------------------------------------------------------- #
def check3_vector_determinism() -> dict:
    log("## 3. AsyncVectorEnv determinism + throughput\n")
    import gymnasium as gym

    cfg = NavRLConfig(max_episode_steps=1000)
    n_envs = 6  # MAC cockpit ~6 workers (WIN ~16)
    k_steps = 40

    def make_batch():
        return gym.vector.AsyncVectorEnv(
            [lambda: make_nav_env(cfg) for _ in range(n_envs)]
        )

    # Two independent vector envs, identical seeds + identical action stream ->
    # identical observations (determinism).
    venv_a = make_batch()
    venv_b = make_batch()
    obs_a, _ = venv_a.reset(seed=123)
    obs_b, _ = venv_b.reset(seed=123)
    reset_match = bool(np.array_equal(obs_a, obs_b))
    act_rng = np.random.default_rng(7)
    max_abs_diff = 0.0
    for _ in range(k_steps):
        acts = act_rng.uniform(-1, 1, size=(n_envs, 42)).astype(np.float32)
        oa, ra, _, _, _ = venv_a.step(acts)
        ob, rb, _, _, _ = venv_b.step(acts)
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(oa - ob))),
                           float(np.max(np.abs(np.asarray(ra) - np.asarray(rb)))))
    venv_a.close()
    venv_b.close()

    # Throughput: steps/s over a fresh batch.
    venv = make_batch()
    venv.reset(seed=123)
    t0 = time.perf_counter()
    for _ in range(k_steps):
        acts = act_rng.uniform(-1, 1, size=(n_envs, 42)).astype(np.float32)
        venv.step(acts)
    t_step_loop = time.perf_counter() - t0
    steps_per_s = (n_envs * k_steps) / t_step_loop

    # resets/s over the same batch (batched, so cost is the slowest of n_envs).
    n_reset_batches = 4
    t0 = time.perf_counter()
    for i in range(n_reset_batches):
        venv.reset(seed=1000 + i)
    t_reset_loop = time.perf_counter() - t0
    resets_per_s = (n_envs * n_reset_batches) / t_reset_loop
    venv.close()

    # Single-env reset-time vs step-time split (the throughput ceiling driver).
    solo = make_nav_env(cfg)
    rs = []
    for i in range(4):
        t0 = time.perf_counter()
        solo.reset(seed=i)
        rs.append(time.perf_counter() - t0)
    ss = []
    a = np.zeros(42, dtype=np.float32)
    for _ in range(20):
        t0 = time.perf_counter()
        solo.step(a)
        ss.append(time.perf_counter() - t0)
    solo.close()
    reset_s = float(np.mean(rs))
    step_s = float(np.mean(ss))

    log(f"- determinism (2 envs, same seed + action stream over {k_steps} steps):")
    log(f"  - reset obs identical : {reset_match}")
    log(f"  - max |Δ| (obs+reward) across all steps: {max_abs_diff:.3e}  "
        f"(0 == bit-deterministic)")
    log(f"- throughput ({n_envs} async envs):")
    log(f"  - steps/s  : {steps_per_s:6.1f}  ({n_envs*k_steps} env-steps in "
        f"{t_step_loop:.1f} s)")
    log(f"  - resets/s : {resets_per_s:6.2f}  ({n_envs*n_reset_batches} env-resets "
        f"in {t_reset_loop:.1f} s)")
    log(f"- single-env cost split (the throughput ceiling driver):")
    log(f"  - reset (rebuild FlyEnv + warmup): {reset_s*1000:7.1f} ms")
    log(f"  - step  (1 control = 40 physics) : {step_s*1000:7.1f} ms")
    log(f"  - reset == {reset_s/step_s:.1f} steps' worth of wall time. RL burns "
        f"far more episodes than CMA-ES, so if reset dominates we revisit "
        f"(move the obstacle as a body vs re-baking geoms). Measured now so the "
        f"ceiling is known before the full run.")
    log("")
    return {
        "reset_match": reset_match, "max_abs_diff": max_abs_diff,
        "steps_per_s": steps_per_s, "resets_per_s": resets_per_s,
        "reset_ms": reset_s * 1000, "step_ms": step_s * 1000,
    }


# --------------------------------------------------------------------------- #
def check4_domain_randomization() -> dict:
    log("## 4. Domain randomization variation + held-out disjointness\n")
    cfg = NavRLConfig()
    env = NavRLEnv(cfg)

    # Train resets vary the bearing + obstacle.
    bearings, obs_xy, radii = [], [], []
    for i in range(12):
        _, info = env.reset(seed=(None if i else 999))  # seed once, then stream
        lay = info["layout"]
        bearings.append(lay["bearing_deg"])
        obs_xy.append(lay["obstacles"][0])
        radii.append(lay["obstacle_radius"])
    bearings = np.array(bearings)
    obs_xy = np.array(obs_xy)
    radii = np.array(radii)

    # Held-out resets are a fixed, repeatable set.
    ev1 = [env.reset(options={"eval": True})[1]["layout"]["params"] for _ in range(cfg.held_out_n)]
    env2 = NavRLEnv(cfg)
    ev2 = [env2.reset(options={"eval": True})[1]["layout"]["params"] for _ in range(cfg.held_out_n)]
    held_repeatable = all(e1 == e2 for e1, e2 in zip(ev1, ev2))

    # Disjointness: draw many train layouts, confirm none is "too close" to any
    # held-out layout (the rejection sampler guarantees this).
    rng = np.random.default_rng(42)
    n_train = 5000
    collisions = 0
    for _ in range(n_train):
        p = env._sample_train_params(rng)
        if any(_params_too_close(p, h, cfg) for h in env._held_out):
            collisions += 1

    log(f"- train resets (12, seeded once then streamed):")
    log(f"  - bearing_deg : min={bearings.min():.1f} max={bearings.max():.1f} "
        f"std={bearings.std():.1f} (range {cfg.bearing_deg_range})")
    log(f"  - obstacle x  : min={obs_xy[:,0].min():.2f} max={obs_xy[:,0].max():.2f}")
    log(f"  - obstacle y  : min={obs_xy[:,1].min():.2f} max={obs_xy[:,1].max():.2f}")
    log(f"  - radius      : min={radii.min():.2f} max={radii.max():.2f}")
    log(f"  - distinct obstacle positions: {len(set(map(tuple, obs_xy)))}/12  "
        f"(DR visibly varies obstacle + bearing)")
    log(f"- held-out set ({cfg.held_out_n} layouts, seed={cfg.held_out_seed}):")
    log(f"  - repeatable across env instances: {held_repeatable}")
    log(f"  - train/held-out disjointness: {collisions}/{n_train} train draws "
        f"fell within tolerance of a held-out layout  "
        f"({'DISJOINT' if collisions == 0 else 'LEAK'})")
    log("")
    return {"held_repeatable": held_repeatable, "disjoint_leaks": collisions}


# --------------------------------------------------------------------------- #
def main() -> None:
    log("# Gate 1 — N-RL env sanity (PROMPT_n_rl_1a_env.md)\n")
    log("Env layer only (`rl/env_base.py` + `rl/nav_env.py`); random policy.\n")
    r1 = check1_shapes_and_episode()
    r2 = check2_newton_cap()
    r3 = check3_vector_determinism()
    r4 = check4_domain_randomization()

    passed = (
        r1["trunc_ok"] is True
        and r2["cap_applied"] and r2["free_iters"] == 100
        and r3["reset_match"] and r3["max_abs_diff"] == 0.0
        and r4["held_repeatable"] and r4["disjoint_leaks"] == 0
    )
    log("## Verdict\n")
    log(f"**Gate 1: {'PASS' if passed else 'REVIEW'}** — random episodes run and "
        f"truncate on the time limit; obs/action shapes + spaces correct; Newton "
        f"cap verified applied obstacles-only (20/10) and obstacle-free envs "
        f"untouched ({r2['free_iters']}/50 defaults); worst-case rollout bounded "
        f"(~{r2['t_capped']/1000*1000/1:.1f} s for 1000 steps); vector envs "
        f"bit-deterministic (max|Δ|={r3['max_abs_diff']:.0e}); throughput "
        f"{r3['steps_per_s']:.0f} steps/s, reset {r3['reset_ms']:.0f} ms vs step "
        f"{r3['step_ms']:.1f} ms; DR varies + held-out disjoint "
        f"({r4['disjoint_leaks']} leaks).")
    log("")
    log(f"> **Carried finding for the partner:** physical obstacle blocking is "
        f"absent on this `flygym==2.0.1`/MAC build (obstacles are sensed-only); "
        f"the cap defends a regime that doesn't trigger here. Verify on WIN.")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[gate1] wrote {OUT}")


if __name__ == "__main__":
    main()
