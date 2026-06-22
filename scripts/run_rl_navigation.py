"""CLI for the RL obstacle-navigation run (N-RL): PPO + domain randomization.

This wires the three wave-1 modules into one runnable stack and mirrors the
``run_evolution_*`` CLIs:

    # Validation-first calibration: the four gates + REPORT, then STOP (no full run).
    uv run python scripts/run_rl_navigation.py --calibrate

    # The green-lit full run (proposed by the calibration; launch only on confirm).
    uv run python scripts/run_rl_navigation.py --full

The stack:
  * ``NavRLEnv`` (rl/nav_env.py)        — the obstacle-nav task on the physical FlyEnv,
                                           with ``w_collide`` and the Newton-cap drop set here.
  * ``NCAPolicy`` (rl/policies.py)      — NCA(nav=True) as a Gaussian PPO policy, warm-started
                                           from the chemo forager.
  * ``rl.ppo.train`` (rl/ppo.py)        — the policy-agnostic cleanrl-style PPO loop; nav's
                                           held-out detour/reach/collision metrics are wired in
                                           as the ``eval_fn``; W&B on.

Per the N-RL-PHYS carried findings, the calibration drops the obstacle-env CG solver cap
back to MuJoCo's default Newton (the cap was a safety belt for a regime that does not
occur; Gate 1 asserts ncon stays bounded so any real instability is still caught).

Rebalance pass (2b): the 2a calibration (w_collide=0.75) correctly failed Gate 4 — PPO
drove collisions to ~0 but reach/detour collapsed because the per-step contact penalty
dwarfed the per-step approach gain (an over-cautious "stall short of the obstacle" basin).
2b changes one experimental variable, the reward balance: ``w_collide`` 0.75 -> 0.25
(sweep band 0.2-0.4) and a new ``w_approach=2.0`` weighting the Δapproach term up so a
clean reach dominates the worst plausible per-episode contact cost. It also (a) wraps the
vec envs with ``RecordEpisodeStatistics`` (the 2a return logged ``nan``) and (b)
collision-gates ``reach``/``detour_success`` (contacts <= GATE_TAU) so the held-out
yardstick is honest. See ``ops/reports/REPORT_n_rl_calibration.md`` (Rebalance pass).

Outputs (calibrate):
    scratch/nrl/calibration_rl.json          (machine-readable gate results)
    ops/reports/REPORT_n_rl_calibration.md   (the report)
    scratch/nrl/ckpt/...                      (PPO checkpoints from the short Gate-4 run)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cellular_gaits.nca import CHEMO_N_PARAMS, NCA  # noqa: E402
from cellular_gaits.rl.nav_env import (  # noqa: E402
    NavRLConfig,
    NavRLEnv,
    make_nav_env,
)
from cellular_gaits.rl.policies import DEFAULT_CHEMO_JSON, NCAPolicy  # noqa: E402
from cellular_gaits.rl.ppo import PPOConfig, resolve_device, train  # noqa: E402

# --- carried calibration constants (N-RL-PHYS findings) --------------------- #
# 2b rebalance: w_collide 0.75 -> 0.25 (sweep band 0.2-0.4). At the real ~11-28
# in-contact steps/episode this integrates to ~3-7 of penalty — enough to make a clean
# detour the optimum without suppressing the final approach (see the Rebalance pass report).
CALIBRATION_W_COLLIDE = 0.25   # real-contact penalty per in-contact step (2b; was 2a's 0.75)
CALIBRATION_W_APPROACH = 2.0   # Δapproach gain weight (2b: homing dominates worst contact)
NCON_BOUND = 25                # assert the real contact set stays bounded (phys gate 3: <=23)

# --- held-out metric definitions (mirror outputs/web_data_n/navigation_metrics) #
COLLISION_OK_STEPS = 8         # an episode "avoids" if it has <= this many in-contact steps
DETOUR_THRESH = 0.3            # |perp offset| past which a detour counts as a real swerve
FEELER_HOT = 0.1               # feeler L+R above this == "near the obstacle" (matches N-A)
# 2b: collision-gate the success metric. reach/detour_success are "clean" only if the
# episode stayed at/under GATE_TAU in-contact steps — a small tolerance for an incidental
# single-step brush, distinct from clean_reach's strict zero. Gating reconciles Gate 2:
# N-A "succeeded" by grazing through (119.8 contacts/ep); under the gate it reads ~0.
GATE_TAU = 2                   # max in-contact steps for a "clean" (collision-gated) reach/detour
# 2c: the outer edge of the forager's homing cone. Gate 2 showed the CMA-ES forager
# clean-reaches bearings ~35-56° and falls off beyond that, so 60° is the boundary
# between "inside the cone" (where N-A is already competent — RL proves nothing) and
# "outside" (the omnidirectional generalization the curriculum must win). The held-out
# clean_reach is split on this so the LEADING EDGE (does reach extend past the cone?)
# is visible. Bearings are folded to (-180, 180] before the test so the omnidirectional
# full-run held-out buckets correctly (a 350° goal is a -10° goal = inside).
FORAGER_CONE_DEG = 60.0

NA_CONTROLLER_JSON = ROOT / "outputs" / "web_data_n" / "navigation_controller.json"
NA_CHECKPOINT = ROOT / "checkpoints" / "2026-06-21T05-22-48Z" / "gen_70.npz"
SCRATCH = ROOT / "scratch" / "nrl"
REPORT = ROOT / "ops" / "reports" / "REPORT_n_rl_calibration.md"
# Sentinel delimiting the appended Rebalance-pass section so re-runs replace it
# idempotently instead of stacking duplicates (the v1 report above it is preserved).
REBALANCE_MARKER = "<!-- REBALANCE_2B_START -->"
# 2c appends AFTER the preserved 2b section; re-runs replace from here on idempotently.
CURRICULUM_MARKER = "<!-- CURRICULUM_2C_START -->"


class _Tee:
    """Write to several streams at once (keep live stdout AND capture for parsing)."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


# --------------------------------------------------------------------------- #
# Policy construction
# --------------------------------------------------------------------------- #
def build_warm_started_policy(
    cfg: NavRLConfig, chemo_json: Path = DEFAULT_CHEMO_JSON, log_std_init: float = 0.0
) -> tuple[NCAPolicy, dict]:
    """A fresh ``NCAPolicy`` warm-started from the chemo forager (feelers zero-init).

    This is the actual training start: the pure forager, avoidance not yet learned.
    """
    policy = NCAPolicy(log_std_init=log_std_init)
    ws = policy.warm_start_from_chemo(chemo_json)
    return policy, ws


def load_na_policy(cfg: NavRLConfig, source: Path = NA_CONTROLLER_JSON) -> tuple[NCAPolicy, dict]:
    """Load the N-A CMA-ES nav controller (the overfit baseline) AS the policy.

    Reads the 1524-param nav ``flat_params`` from the N-A web export and sets them on
    an ``NCA(nav=True)`` wrapped in an ``NCAPolicy`` (value head / log_std are fresh, but
    eval uses only ``mean_action``, so they never matter here).
    """
    if source.suffix == ".json":
        vec = np.asarray(json.loads(source.read_text())["flat_params"], dtype=np.float64)
    else:  # checkpoint .npz fallback
        data = np.load(source)
        vec = np.asarray(data["best_params"], dtype=np.float64)
    nca = NCA(nav=True)
    if vec.size != nca.n_params:
        raise ValueError(f"N-A controller has {vec.size} params, expected {nca.n_params}")
    nca.set_params(vec)
    policy = NCAPolicy(nca)
    return policy, {"source": str(source.relative_to(ROOT)), "n_params": int(vec.size)}


# --------------------------------------------------------------------------- #
# Held-out evaluation (the yardstick vs N-A)
# --------------------------------------------------------------------------- #
def _bearing_split(rows: list[dict]) -> dict:
    """Split held-out rows into inside/outside the forager's cone and report the
    un-gameable bars (clean_reach, raw reach) per bucket. The OUTSIDE bucket is the
    omnidirectional frontier the curriculum must win; its clean_reach moving pre->post
    is the leading indicator for the full run (see ``FORAGER_CONE_DEG``)."""
    def fold(b: float) -> float:
        return ((b + 180.0) % 360.0) - 180.0

    inside = [r for r in rows if abs(fold(r["bearing_deg"])) <= FORAGER_CONE_DEG]
    outside = [r for r in rows if abs(fold(r["bearing_deg"])) > FORAGER_CONE_DEG]

    def mean(rs: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    return {
        "cone_deg": FORAGER_CONE_DEG,
        "inside_n": len(inside),
        "outside_n": len(outside),
        "inside_clean_reach": mean(inside, "clean_reach"),
        "outside_clean_reach": mean(outside, "clean_reach"),
        "inside_reach": mean(inside, "reached"),
        "outside_reach": mean(outside, "reached"),
    }


def evaluate_policy(
    policy: NCAPolicy,
    cfg: NavRLConfig,
    n_episodes: int | None = None,
    device: str | torch.device = "cpu",
    eval_env: NavRLEnv | None = None,
) -> dict:
    """Deterministic (``mean_action``) rollouts over the FROZEN held-out layout set.

    Returns aggregate held-out metrics + per-episode rows. Metric definitions mirror
    the N-A export (``navigation_metrics.json``):
      reached         : got within reach_radius of the goal at some step
      avoided         : <= COLLISION_OK_STEPS in-contact steps over the episode
      detour_correct  : swerved to the correct side (block left -> go right, etc.)
      detour_success  : reached AND avoided AND detour_correct  (raw/ungated, kept for transparency)
      clean_reach     : reached AND zero collisions — the HARDER, un-gameable bar (see note)
      reach_gated     : reached AND <= GATE_TAU in-contact steps (2b: collision-gated reach)
      detour_success_gated : reach_gated AND detour_correct (2b: the honest headline yardstick)

    2b also returns a per-episode REWARD DECOMPOSITION (mean Δapproach vs collision vs
    step-cost vs reach-bonus, under the live cfg weights) so the reward balance is legible
    pre- and post-PPO. Components mirror ``NavRLEnv._compute_reward`` exactly:
      r_approach  = w_approach * Σ Δapproach  (telescopes to w_approach*(d_start - d_final))
      r_collision = -w_collide * (#in-contact steps)
      r_stepcost  = -step_cost * steps
      r_reach_bonus = reach_bonus iff reached AND collision-free, else 0
    Their sum == the episode return the env actually paid.

    ``detour_perp`` is measured at the **peak-feeler** step (closest approach to the
    obstacle, where the dodge decision is committed), mirroring N-A's ``_signed_detour``
    feeler-active window rather than at the goal (the goal sits on the line, so a
    goal-approach offset is ~0 and meaningless). HONESTY: in the now-physical env the
    obstacle passively deflects a fly that walks into it toward the correct side, so a
    geometric ``detour_correct`` is partly contaminated by physics — ``clean_reach``
    (reach with NO contact) is the discriminator that physics cannot fake.
    """
    device = torch.device(device) if isinstance(device, str) else device
    n_episodes = n_episodes or cfg.held_out_n
    env = eval_env or NavRLEnv(cfg)
    policy_was_training = policy.training
    policy.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset(options={"eval": True})
            layout = info["layout"]
            block_side = float(layout["block_side"])
            state = policy.initial_state(1, device)
            reached = False
            ever_collided = False
            coll_steps = 0
            min_dist = float("inf")
            peak_feeler = -1.0
            detour_at_peak = 0.0
            steps = 0
            sum_approach = 0.0  # Σ Δapproach (telescopes to d_start - d_final)
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, state = policy.mean_action(obs_t, state)
                a = action.squeeze(0).cpu().numpy()
                obs, _r, term, trunc, info = env.step(a)
                steps += 1
                sum_approach += float(info["approach"])
                if info["collision"]:
                    coll_steps += 1
                    ever_collided = True
                if info["reached"]:
                    reached = True
                min_dist = min(min_dist, float(info["dist_to_goal"]))
                fmag = float(info["feeler_L"]) + float(info["feeler_R"])
                if fmag > FEELER_HOT and fmag > peak_feeler:
                    peak_feeler = fmag
                    detour_at_peak = float(info["detour_perp"])
                if term or trunc:
                    break
            avoided = coll_steps <= COLLISION_OK_STEPS
            detour_correct = bool(block_side * detour_at_peak < -DETOUR_THRESH)
            detour_success = bool(reached and avoided and detour_correct)  # raw/ungated
            clean_reach = bool(reached and not ever_collided)
            # 2b: collision-gated metrics (contacts <= GATE_TAU == "clean").
            reach_gated = bool(reached and coll_steps <= GATE_TAU)
            detour_success_gated = bool(reach_gated and detour_correct)
            # 2b: per-episode reward decomposition under the live cfg weights.
            r_approach = float(cfg.w_approach * sum_approach)
            r_collision = float(-cfg.w_collide * coll_steps)
            r_stepcost = float(-cfg.step_cost * steps)
            r_reach_bonus = float(cfg.reach_bonus if (reached and not ever_collided) else 0.0)
            rows.append({
                "bearing_deg": float(layout["bearing_deg"]),
                "block_side": block_side,
                "reached": bool(reached),
                "collision_count": int(coll_steps),
                "avoided": bool(avoided),
                "detour_perp": detour_at_peak,
                "detour_correct": detour_correct,
                "detour_success": detour_success,
                "detour_success_gated": detour_success_gated,
                "reach_gated": reach_gated,
                "clean_reach": clean_reach,
                "min_dist": min_dist,
                "steps": steps,
                "r_approach": r_approach,
                "r_collision": r_collision,
                "r_stepcost": r_stepcost,
                "r_reach_bonus": r_reach_bonus,
                "ep_return": r_approach + r_collision + r_stepcost + r_reach_bonus,
            })
    if policy_was_training:
        policy.train()
    n = len(rows)
    agg = {
        "detour_success_rate": float(np.mean([r["detour_success"] for r in rows])),  # raw/ungated
        "detour_success_gated_rate": float(np.mean([r["detour_success_gated"] for r in rows])),
        "clean_reach_rate": float(np.mean([r["clean_reach"] for r in rows])),
        "reach_rate": float(np.mean([r["reached"] for r in rows])),  # raw/ungated
        "reach_gated_rate": float(np.mean([r["reach_gated"] for r in rows])),
        "avoid_rate": float(np.mean([r["avoided"] for r in rows])),
        "detour_correct_rate": float(np.mean([r["detour_correct"] for r in rows])),
        "mean_collisions": float(np.mean([r["collision_count"] for r in rows])),
        "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
        "gate_tau": GATE_TAU,
        # 2c: held-out split by bearing bucket (inside vs outside the forager's cone).
        # The leading indicator for the full run — clean_reach trending up on the
        # OUTSIDE bucket = the curriculum is dissolving the narrow-cone overfit.
        "split": _bearing_split(rows),
        # 2b reward decomposition (per-episode means under the live cfg weights).
        "decomp": {
            "r_approach": float(np.mean([r["r_approach"] for r in rows])),
            "r_collision": float(np.mean([r["r_collision"] for r in rows])),
            "r_stepcost": float(np.mean([r["r_stepcost"] for r in rows])),
            "r_reach_bonus": float(np.mean([r["r_reach_bonus"] for r in rows])),
            "ep_return": float(np.mean([r["ep_return"] for r in rows])),
        },
        "n_episodes": n,
    }
    return {"aggregate": agg, "rows": rows}


def make_eval_fn(cfg: NavRLConfig, n_episodes: int | None = None, device: str = "cpu"):
    """A cached ``eval_fn(agent) -> {metric: float}`` for the PPO loop.

    Builds one persistent held-out eval env (reused across calls) and returns the flat
    aggregate dict; keys land in W&B under ``eval/*``.
    """
    cache: dict[str, NavRLEnv] = {}

    def eval_fn(agent) -> dict:
        if "env" not in cache:
            cache["env"] = NavRLEnv(cfg)
        out = evaluate_policy(agent, cfg, n_episodes=n_episodes, device=device,
                              eval_env=cache["env"])
        return out["aggregate"]

    return eval_fn


# --------------------------------------------------------------------------- #
# Gate 0 — curriculum self-check (the 2c levers + default-off bit-exactness)
# --------------------------------------------------------------------------- #
def gate0_curriculum_selfcheck(cfg: NavRLConfig) -> dict:
    """Validate the four wired 2c levers without spending PPO budget.

    (a) Default-OFF must be a no-op: a fresh ``NavRLConfig()`` env's schedules return
        the base values (so 2b stays bit-exact — Gate 3 / prior behavior preserved).
    (b) Curriculum-ON sweeps as configured: the bearing band widens to omnidirectional
        over ``bearing_widen_frac``, ``w_collide`` anneals base->max over its window,
        and the obstacle slides from off-path (extra lateral) onto the path.
    """
    print("\n[gate0] curriculum self-check: schedules + default-off bit-exactness ...")
    off = NavRLEnv(NavRLConfig())
    off_exact = bool(
        off.effective_w_collide() == off.cfg.w_collide
        and tuple(off._train_bearing_range(0.5)) == tuple(off.cfg.bearing_deg_range)
        and off._apply_obstacle_curriculum({"lateral": 1.0}, 0.0)["lateral"] == 1.0
    )

    on = NavRLEnv(NavRLConfig(
        curriculum=True, curriculum_horizon_steps=1000,
        bearing_deg_range=(20.0, 60.0), bearing_deg_target=(0.0, 360.0),
        bearing_widen_frac=0.6, w_collide=0.25, w_collide_max=0.75,
        w_collide_anneal_start=0.2, w_collide_anneal_end=0.7,
        obstacle_far_extra_lateral=2.0, obstacle_near_frac=0.5,
    ))
    bands = {p: tuple(round(x, 1) for x in on._train_bearing_range(p))
             for p in (0.0, 0.3, 0.6, 1.0)}
    wcol = {p: round(on.effective_w_collide(p), 3) for p in (0.0, 0.2, 0.45, 0.7, 1.0)}
    lat = {p: round(on._apply_obstacle_curriculum({"lateral": 1.0}, p)["lateral"], 2)
           for p in (0.0, 0.25, 0.5)}

    widened = bands[0.0] == (20.0, 60.0) and bands[0.6] == (0.0, 360.0)
    annealed = wcol[0.0] == 0.25 and wcol[1.0] == 0.75 and 0.25 < wcol[0.45] < 0.75
    slid = lat[0.0] > lat[0.5] and lat[0.5] == 1.0
    ok = bool(off_exact and widened and annealed and slid)

    print(f"[gate0] default-off bit-exact={off_exact}  band 0->.6={bands[0.0]}->{bands[0.6]}  "
          f"w_collide 0->1={wcol[0.0]}->{wcol[1.0]} (mid {wcol[0.45]})  "
          f"obstacle lat 0->.5={lat[0.0]}->{lat[0.5]}  pass={ok}")
    return {
        "off_bit_exact": off_exact,
        "bands": {str(k): list(v) for k, v in bands.items()},
        "w_collide": {str(k): v for k, v in wcol.items()},
        "obstacle_lateral": {str(k): v for k, v in lat.items()},
        "pass": ok,
    }


# --------------------------------------------------------------------------- #
# Gate 1 — env sanity on the now-physical task (+ ncon bounded)
# --------------------------------------------------------------------------- #
def gate1_env_sanity(cfg: NavRLConfig, n_envs: int) -> dict:
    import gymnasium as gym

    print("\n[gate1] env sanity on the physical task (Newton cap dropped) ...")
    env = NavRLEnv(cfg)
    obs, info = env.reset(seed=0)
    shapes_ok = (
        env.observation_space.shape == (6, 8, 8)
        and env.observation_space.dtype == np.float32
        and env.action_space.shape == (42,)
        and obs.shape == (6, 8, 8)
        and env.observation_space.contains(obs)
    )
    opt = env.fly.sim.mj_model.opt
    cap_dropped = int(opt.solver) == 2 and int(opt.iterations) == 100  # Newton, uncapped

    # ncon bounded across a real rollout: drive the warm-start forager into the layout
    # so contacts actually occur, and track the peak contact count.
    policy, _ = build_warm_started_policy(cfg)
    policy.eval()
    state = policy.initial_state(1, "cpu")
    ncon_peak = 0
    in_space = True
    steps = 0
    with torch.no_grad():
        for _ in range(cfg.max_episode_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a, state = policy.mean_action(obs_t, state)
            obs, r, term, trunc, info = env.step(a.squeeze(0).numpy())
            ncon_peak = max(ncon_peak, int(env.fly.sim.mj_data.ncon))
            in_space = in_space and env.observation_space.contains(obs)
            steps += 1
            if term or trunc:
                break
    ncon_bounded = ncon_peak <= NCON_BOUND
    env.close()

    # Vectorized determinism + throughput.
    def make_batch():
        return gym.vector.AsyncVectorEnv([lambda: make_nav_env(cfg) for _ in range(n_envs)])

    va, vb = make_batch(), make_batch()
    oa, _ = va.reset(seed=123)
    ob, _ = vb.reset(seed=123)
    reset_match = bool(np.array_equal(oa, ob))
    rng = np.random.default_rng(7)
    max_abs_diff = 0.0
    k = 20
    for _ in range(k):
        acts = rng.uniform(-1, 1, size=(n_envs, 42)).astype(np.float32)
        ra_o, ra_r, _, _, _ = va.step(acts)
        rb_o, rb_r, _, _, _ = vb.step(acts)
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(ra_o - rb_o))),
                           float(np.max(np.abs(np.asarray(ra_r) - np.asarray(rb_r)))))
    va.close()
    vb.close()

    # Throughput: parallel vs single-env steps/s. Both use a STANDING (zero) action so
    # the comparison measures pure stepping — a random action makes the fly fall, which
    # triggers an autoreset (~0.5 s rebuild) mid-measurement and unfairly tanks whichever
    # loop hits more falls. Reset cost is reported separately below.
    a0_batch = np.zeros((n_envs, 42), dtype=np.float32)
    venv = make_batch()
    venv.reset(seed=5)
    t0 = time.perf_counter()
    for _ in range(k):
        venv.step(a0_batch)
    par_sps = (n_envs * k) / (time.perf_counter() - t0)
    venv.close()
    solo = make_nav_env(cfg)
    solo.reset(seed=5)
    a0 = np.zeros(42, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(k):
        solo.step(a0)
    seq_sps = k / (time.perf_counter() - t0)
    # Reset cost (the throughput ceiling driver for RL, which burns many episodes).
    rs = []
    for i in range(3):
        t0 = time.perf_counter()
        solo.reset(seed=100 + i)
        rs.append(time.perf_counter() - t0)
    reset_ms = float(np.mean(rs)) * 1000
    solo.close()

    result = {
        "shapes_ok": shapes_ok,
        "cap_dropped": cap_dropped,
        "solver": int(opt.solver),
        "iterations": int(opt.iterations),
        "ncon_peak": ncon_peak,
        "ncon_bounded": ncon_bounded,
        "rollout_steps": steps,
        "obs_in_space_every_step": in_space,
        "vec_reset_match": reset_match,
        "vec_max_abs_diff": max_abs_diff,
        "n_envs": n_envs,
        "parallel_steps_per_s": par_sps,
        "sequential_steps_per_s": seq_sps,
        "speedup": par_sps / seq_sps if seq_sps else float("nan"),
        "reset_ms": reset_ms,
    }
    print(f"[gate1] shapes_ok={shapes_ok} cap_dropped={cap_dropped} (solver={int(opt.solver)}/"
          f"iters={int(opt.iterations)})  ncon_peak={ncon_peak} (<= {NCON_BOUND}: {ncon_bounded})")
    print(f"[gate1] vec determinism: reset_match={reset_match} max|Δ|={max_abs_diff:.2e}  "
          f"throughput: {par_sps:.0f} steps/s ({n_envs} envs) vs {seq_sps:.0f} solo "
          f"-> {result['speedup']:.1f}x  reset={reset_ms:.0f}ms")
    return result


# --------------------------------------------------------------------------- #
# Gate 2 — the N-A overfit baseline, measured in THIS env
# --------------------------------------------------------------------------- #
def gate2_na_baseline(cfg: NavRLConfig) -> dict:
    print("\n[gate2] N-A overfit baseline, measured on the held-out set in THIS env ...")
    policy, meta = load_na_policy(cfg)
    out = evaluate_policy(policy, cfg, device="cpu")
    agg = out["aggregate"]
    agg["source"] = meta["source"]
    print(f"[gate2] N-A on held-out: detour_success(raw)={agg['detour_success_rate']:.2f}  "
          f"detour_success(gated τ={GATE_TAU})={agg['detour_success_gated_rate']:.2f}  "
          f"reach(raw)={agg['reach_rate']:.2f}  reach(gated)={agg['reach_gated_rate']:.2f}  "
          f"clean_reach={agg['clean_reach_rate']:.2f}  mean_coll={agg['mean_collisions']:.1f}  "
          f"(source {meta['source']})")
    return out


# --------------------------------------------------------------------------- #
# Gate 3 — A/B integrity (warm-start feelers-off still == chemo forward, bit-exact)
# --------------------------------------------------------------------------- #
def gate3_ab_integrity(cfg: NavRLConfig) -> dict:
    print("\n[gate3] A/B integrity: feeler-zeroed warm-start == chemo forward (bit-exact) ...")
    # Load the trained chemo forward as the reference.
    chemo_vec = np.asarray(json.loads(DEFAULT_CHEMO_JSON.read_text())["flat_params"],
                           dtype=np.float64)
    assert chemo_vec.size == CHEMO_N_PARAMS, chemo_vec.size
    chemo = NCA(chemo=True)
    chemo.set_params(chemo_vec)
    policy, ws = build_warm_started_policy(cfg)
    policy.eval()

    rng = np.random.RandomState(0)
    ca = NCA.init_state(seed=0)  # (1,4,8,8) zeros == policy.initial_state
    ja = rng.uniform(-1, 1, 42)
    fc = (rng.uniform(0, 1, 6) > 0.5).astype(float)
    # chemo 8-input map vs nav 10-input map with feelers ZERO.
    sc = NCA.build_chemo_sensor_map(ja, fc, 0.3, 0.7)
    sn0 = NCA.build_nav_sensor_map(ja, fc, 0.3, 0.7, 0.0, 0.0)  # (1,6,8,8)
    with torch.no_grad():
        chemo_out = chemo.step(ca, sc)  # the 8-input chemo forward (motor block)
        nav_a0, _ = policy.mean_action(sn0.to(torch.float32), ca.to(torch.float32))
        # chemo.step returns the full (1,4,8,8) next state; compare the motor block.
        chemo_motor = chemo_out[0, 0, :7, :6].reshape(-1)
        d_feeler_off = float((chemo_motor - nav_a0.reshape(-1)).abs().max())
        # feelers ON (raw 1.0/0.6) must STILL match (feeler weights are zero at warm start).
        sn1 = NCA.build_nav_sensor_map(ja, fc, 0.3, 0.7, 1.0, 0.6)
        nav_a1, _ = policy.mean_action(sn1.to(torch.float32), ca.to(torch.float32))
        d_feeler_on = float((chemo_motor - nav_a1.reshape(-1)).abs().max())

    result = {
        "feeler_off_max_abs_delta": d_feeler_off,
        "feeler_on_max_abs_delta": d_feeler_on,
        "chemo_params": int(chemo_vec.size),
        "nav_params": int(ws["nav_params"]),
        "ab_bit_exact": bool(d_feeler_off == 0.0 and d_feeler_on == 0.0),
    }
    print(f"[gate3] feeler-off Δ={d_feeler_off:.2e}  feeler-on Δ={d_feeler_on:.2e}  "
          f"bit-exact={result['ab_bit_exact']}")
    return result


# --------------------------------------------------------------------------- #
# Gate 4 — short PPO run shows held-out generalization signal
# --------------------------------------------------------------------------- #
def gate4_short_ppo(cfg: NavRLConfig, args, na_baseline: dict) -> dict:
    print(f"\n[gate4] short PPO run ({args.cal_steps} steps, {args.n_envs} envs) ...")
    policy, ws = build_warm_started_policy(cfg)
    device = resolve_device(args.device)
    policy = policy.to(device)  # eval moves obs to this device; weights must match

    # Pre-training held-out eval (the warm-start forager, avoidance not yet learned).
    pre = evaluate_policy(policy, cfg, device=device)["aggregate"]
    print(f"[gate4] pre-PPO held-out: detour_success={pre['detour_success_rate']:.2f}  "
          f"reach={pre['reach_rate']:.2f}  mean_coll={pre['mean_collisions']:.1f}")

    ppo = PPOConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        device=args.device,
        total_steps=args.cal_steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_minibatch=args.n_minibatch,
        update_epochs=args.update_epochs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        eval_every=args.eval_every,
        ckpt_every=0,
        log_every=1,
        wandb=args.wandb,
        behavior="nav",
        wandb_tags=("nrl", "calibration"),
        ckpt_dir=str(SCRATCH / "ckpt"),
    )
    eval_fn = make_eval_fn(cfg, device=str(device))
    t0 = time.perf_counter()
    # Tee train()'s stdout so we keep the live log AND can parse the per-update
    # `return=` curve afterward — the 2b RecordEpisodeStatistics fix must make it finite.
    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        train(ppo, lambda: make_nav_env(cfg), policy, eval_fn=eval_fn)
    wall_s = time.perf_counter() - t0
    policy = policy.to(device)  # train() leaves it on device; keep eval consistent

    return_curve = [float(x) for x in re.findall(r"return=(nan|-?\d+\.\d+)", buf.getvalue())]
    n_finite = sum(1 for x in return_curve if np.isfinite(x))
    returns_finite = bool(return_curve) and n_finite == len(return_curve)
    print(f"[gate4] episodic_return curve ({len(return_curve)} updates): "
          f"{'FINITE' if returns_finite else 'has nan/empty'} "
          f"-> {[round(x, 2) for x in return_curve]}")

    post = evaluate_policy(policy, cfg, device=device)["aggregate"]
    print(f"[gate4] post-PPO held-out: detour_success(gated)={post['detour_success_gated_rate']:.2f}  "
          f"clean_reach={post['clean_reach_rate']:.2f}  reach(raw)={post['reach_rate']:.2f}  "
          f"mean_coll={post['mean_collisions']:.1f}")

    na = na_baseline["aggregate"]
    # Generalization signal: the harder, un-gameable bars move the right way vs the N-A
    # overfit — held-out clean_reach or the collision-GATED detour_success up, and/or
    # collisions down with reach held (2b: gate the detour bar so grazing can't fake it).
    signal = (
        post["clean_reach_rate"] > na["clean_reach_rate"]
        or post["detour_success_gated_rate"] > na["detour_success_gated_rate"]
        or (post["reach_rate"] >= na["reach_rate"] and post["mean_collisions"] < na["mean_collisions"])
    )
    return {
        "pre": pre,
        "post": post,
        "na_baseline": na,
        "wall_s": wall_s,
        "num_updates": ppo.num_updates,
        "return_curve": return_curve,
        "returns_finite": returns_finite,
        "config": {
            "total_steps": ppo.total_steps, "n_envs": ppo.n_envs, "n_steps": ppo.n_steps,
            "n_minibatch": ppo.n_minibatch, "update_epochs": ppo.update_epochs,
            "learning_rate": ppo.learning_rate, "ent_coef": ppo.ent_coef,
        },
        "generalization_signal": bool(signal),
    }


# --------------------------------------------------------------------------- #
# Calibration driver + report
# --------------------------------------------------------------------------- #
def calibrate(args) -> None:
    # 2c re-calibrates on a HARDER, wider band. Gate 2 (2a/2b) found the forager's
    # ~40° cone band too easy, so this run uses a FIXED moderately-wide band
    # [cal_bearing_lo, cal_bearing_hi] (inside + just-beyond the cone) with a matching
    # widened held-out — the cleanest cheap test of "does clean_reach extend OUTSIDE
    # the cone?". The progressive curriculum SCHEDULE (which compresses badly into a
    # cheap budget) is exercised + validated by Gate 0 and baked into --full instead.
    band = (args.cal_bearing_lo, args.cal_bearing_hi)
    # 2d homing-extension probe (--obstacle-free): push the obstacle 20 world-units off the
    # start->goal line (>> radius + feeler_range=6), so the straight path never clips it (no
    # collisions possible) and the feelers read ~0. This strips out the avoidance confound that
    # collapsed the 2c probe and asks the crux question alone: can PPO extend the warm-start
    # forager's ~40° homing cone toward omnidirectional?
    obstacle_kw = dict(obstacle_lateral_range=(20.0, 20.0)) if args.obstacle_free else {}
    cfg = NavRLConfig(
        w_collide=args.w_collide, w_approach=args.w_approach, drop_solver_cap=True,
        bearing_deg_range=band, held_out_bearing_deg_range=band,
        held_out_n=args.held_out_n,
        **obstacle_kw,
    )
    mode = "2d OBSTACLE-FREE homing-extension probe" if args.obstacle_free else "2c harder-band calibration"
    print(f"[cal] {mode}: bearing_deg_range={cfg.bearing_deg_range} "
          f"held_out_n={cfg.held_out_n}  forager_cone={FORAGER_CONE_DEG:.0f}°  "
          f"obstacle_lateral_range={cfg.obstacle_lateral_range}"
          + ("  (obstacle far off-path → pure homing)" if args.obstacle_free
             else "  (fixed moderately-wide band; curriculum validated by Gate 0 + baked into --full)"))
    print(f"[cal] NavRLConfig: w_collide={cfg.w_collide} w_approach={cfg.w_approach} "
          f"drop_solver_cap={cfg.drop_solver_cap} gate_tau={GATE_TAU}")

    g0 = gate0_curriculum_selfcheck(cfg)
    g1 = gate1_env_sanity(cfg, n_envs=args.n_envs)
    g2 = gate2_na_baseline(cfg)
    g3 = gate3_ab_integrity(cfg)
    g4 = gate4_short_ppo(cfg, args, na_baseline=g2)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "w_collide": cfg.w_collide, "w_approach": cfg.w_approach,
            "drop_solver_cap": cfg.drop_solver_cap,
            "bearing_deg_range": list(cfg.bearing_deg_range),
            "held_out_bearing_deg_range": list(cfg.held_out_bearing_deg_range),
            "forager_cone_deg": FORAGER_CONE_DEG,
            "obstacle_frac_range": list(cfg.obstacle_frac_range),
            "obstacle_lateral_range": list(cfg.obstacle_lateral_range),
            "obstacle_radius_range": list(cfg.obstacle_radius_range),
            "reach_bonus": cfg.reach_bonus, "step_cost": cfg.step_cost,
            "gate_tau": GATE_TAU,
            "held_out_n": cfg.held_out_n, "max_episode_steps": cfg.max_episode_steps,
        },
        "gate0_curriculum_selfcheck": g0,
        "gate1_env_sanity": g1,
        "gate2_na_baseline": g2,
        "gate3_ab_integrity": g3,
        "gate4_short_ppo": g4,
    }
    out = SCRATCH / ("calibration_homing_probe.json" if args.obstacle_free else "calibration_rl.json")
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[cal] wrote {out.relative_to(ROOT)}")
    if args.obstacle_free:
        # The probe's deliverable is the streamed gate-4 pre/post inside-vs-outside split;
        # don't clobber the committed 2c "Curriculum pass" report section.
        sp = g4["post"]["split"]
        print(f"[cal] 2d homing probe DONE — held-out clean_reach  inside(<=60°)="
              f"{sp['inside_clean_reach']}  outside(>60°)={sp['outside_clean_reach']}  "
              f"(pre outside={g4['pre']['split']['outside_clean_reach']}). "
              f"Outside lifting off 0 = omnidirectional homing is learnable. Report append skipped.")
    else:
        _append_curriculum_report(args, cfg, payload)


def _fmt(x: float, nd: int = 2) -> str:
    """Format a float, surfacing nan/inf legibly in the markdown."""
    return f"{x:.{nd}f}" if np.isfinite(x) else str(x)


def _append_rebalance_report(args, cfg: NavRLConfig, p: dict) -> None:
    """Append (idempotently) the 'Rebalance pass (2b)' section to the v1 report.

    The first calibration's full report (generated by ``_write_report``) is preserved as
    history; 2b changes one experimental variable (the reward balance) and re-runs the
    same cheap calibration, so it appends a focused summary delimited by ``REBALANCE_MARKER``
    rather than regenerating. Re-running replaces everything from the marker on, so the
    section never stacks duplicates.
    """
    g1 = p["gate1_env_sanity"]
    g2 = p["gate2_na_baseline"]["aggregate"]
    g3 = p["gate3_ab_integrity"]
    g4 = p["gate4_short_ppo"]
    pre, post, na = g4["pre"], g4["post"], g4["na_baseline"]
    dec_na, dec_pre, dec_post = na["decomp"], pre["decomp"], post["decomp"]

    # Verdicts (same correctness bars as v1, plus the 2b honesty checks).
    g1_pass = (g1["shapes_ok"] and g1["cap_dropped"] and g1["ncon_bounded"]
               and g1["vec_reset_match"] and g1["vec_max_abs_diff"] == 0.0)
    g3_pass = g3["ab_bit_exact"]
    # Did the collision gate actually bite N-A's "detour success"? If yes, 2b's grazing
    # hypothesis held; if not, N-A's successes were genuinely clean and the band is simply
    # too easy (the 2a finding). Either way the gate is the right metric — report which.
    gate_reduced = g2["detour_success_gated_rate"] < g2["detour_success_rate"] - 1e-9
    band_too_easy = g2["clean_reach_rate"] > 0.25
    returns_finite = bool(g4.get("returns_finite", False))
    # Did the rebalance work? clean/gated reach or detour trends UP pre->post, not to 0.
    collapsed = post["reach_rate"] <= 0.01
    clean_trend = post["clean_reach_rate"] - pre["clean_reach_rate"]
    greach_trend = post["reach_gated_rate"] - pre["reach_gated_rate"]
    gdetour_trend = post["detour_success_gated_rate"] - pre["detour_success_gated_rate"]
    worked = (not collapsed) and (clean_trend > 0 or greach_trend > 0 or gdetour_trend > 0)

    v1 = "✅" if g1_pass else "❌"
    v2 = "⚠️" if band_too_easy else "✅"  # honest finding: too-easy band is still flagged
    v3 = "✅" if g3_pass else "❌"
    v4 = "✅" if worked else "⚠️"

    rc = g4.get("return_curve", [])
    rc_str = ", ".join(_fmt(x) for x in rc) if rc else "(none captured)"

    L: list[str] = []
    L.append(REBALANCE_MARKER + "\n")
    L.append("\n---\n\n## Rebalance pass (2b) — targeted reward rebalance + re-run\n")
    L.append(
        f"The 2a calibration above **correctly failed Gate 4**: a short PPO run drove "
        f"collisions to ~0 but reach/detour collapsed because at `w_collide=0.75` the per-step "
        f"contact penalty dwarfed the per-step approach gain (an over-cautious *stall short of "
        f"the obstacle* basin). 2b changes **one experimental variable — the reward balance — "
        f"and re-runs the same cheap calibration** (`--cal-steps {g4['config']['total_steps']}`, "
        f"`--n-envs {g4['config']['n_envs']}`). DR / bearing range / curriculum are deliberately "
        f"left untouched so the signal is attributable. **The full run has NOT been launched.**\n"
    )

    L.append("\n### The four changes\n")
    L.append(
        f"1. **Collision penalty down:** `w_collide` 0.75 → **{cfg.w_collide}** (NavRLConfig + "
        f"calibration default; `--w-collide` kept for sweeping, band **0.2–0.4**).\n"
        f"2. **Approach reward up:** new **`w_approach={cfg.w_approach}`** weights the Δapproach "
        f"term so a clean reach (Δapproach telescopes to ~`w_approach·15` + `reach_bonus="
        f"{cfg.reach_bonus:.0f}`) clearly dominates the worst plausible per-episode contact cost "
        f"(`w_collide·~28 ≈ {cfg.w_collide*28:.0f}`). Decomposition below makes the balance legible.\n"
        f"3. **`nan` return fixed:** the vec envs are wrapped with "
        f"`gymnasium.wrappers.RecordEpisodeStatistics` (via `make_nav_env`), so finished-episode "
        f"stats reach the PPO loop and `charts/episodic_return` is finite.\n"
        f"4. **Success metric collision-gated:** `reach`/`detour_success` now require contacts "
        f"≤ **τ={GATE_TAU}** in-contact steps (raw/ungated values kept alongside for transparency).\n"
    )

    L.append("\n### Reward decomposition (held-out, per-episode means under the 2b weights)\n")
    L.append(
        f"| reward component | N-A | pre-PPO (warm start) | post-PPO |\n|---|---|---|---|\n"
        f"| Δapproach (`w_approach·ΣΔ`) | {dec_na['r_approach']:.2f} | {dec_pre['r_approach']:.2f} "
        f"| {dec_post['r_approach']:.2f} |\n"
        f"| collision (`−w_collide·steps`) | {dec_na['r_collision']:.2f} | "
        f"{dec_pre['r_collision']:.2f} | {dec_post['r_collision']:.2f} |\n"
        f"| step-cost (`−step_cost·steps`) | {dec_na['r_stepcost']:.2f} | "
        f"{dec_pre['r_stepcost']:.2f} | {dec_post['r_stepcost']:.2f} |\n"
        f"| reach-bonus (clean arrival) | {dec_na['r_reach_bonus']:.2f} | "
        f"{dec_pre['r_reach_bonus']:.2f} | {dec_post['r_reach_bonus']:.2f} |\n"
        f"| **episode return** | **{dec_na['ep_return']:.2f}** | **{dec_pre['ep_return']:.2f}** "
        f"| **{dec_post['ep_return']:.2f}** |\n\n"
        f"The approach term now carries the return: a clean homing episode is worth "
        f"~`w_approach·15 + {cfg.reach_bonus:.0f}`, while the worst plausible contact cost is only "
        f"~{cfg.w_collide*28:.0f}, so grazing is no longer the cheaper option the way it was at "
        f"`w_collide=0.75`.\n"
    )

    g2_oneliner = ("gate collapses grazed 'success' |\n" if gate_reduced
                   else "successes already clean; band still too easy |\n")
    g4_oneliner = ("clean/gated reach/detour trend UP pre→post (rebalance moved it right) |\n"
                   if worked else "reach/detour still weak at this budget — diagnosed below |\n")
    L.append("\n### Gate results under the new weights / metric\n")
    L.append(
        "| gate | verdict | one-line |\n|---|---|---|\n"
        f"| 1 env sanity + ncon bounded | {v1} | ncon_peak={g1['ncon_peak']}≤{NCON_BOUND}, "
        f"deterministic (max\\|Δ\\|={g1['vec_max_abs_diff']:.0e}), {g1['speedup']:.1f}× parallel |\n"
        f"| 2 N-A baseline, gated | {v2} | gated detour {g2['detour_success_gated_rate']:.2f} vs raw "
        f"{g2['detour_success_rate']:.2f}; clean_reach {g2['clean_reach_rate']:.2f} — " + g2_oneliner
        + f"| 3 A/B bit-exact | {v3} | warm-start == chemo forward, max\\|Δ\\|="
        f"{g3['feeler_off_max_abs_delta']:.0e} |\n"
        f"| 4 rebalance trend | {v4} | " + g4_oneliner
    )
    if gate_reduced:
        gate2_note = (
            f"\n- **Gate 2 (gate reconciles the yardstick).** Under the collision gate (τ={GATE_TAU}) "
            f"the N-A controller's held-out detour drops to **{g2['detour_success_gated_rate']:.2f}** "
            f"from raw/ungated {g2['detour_success_rate']:.2f} (clean_reach "
            f"{g2['clean_reach_rate']:.2f}, mean {g2['mean_collisions']:.1f} contacts/ep) — the 2a "
            f"figure was N-A *grazing through*, and the gate makes the yardstick honest as 2b "
            f"predicted.\n"
        )
    else:
        gate2_note = (
            f"\n- **Gate 2 (honest correction to the 2b hypothesis).** The collision gate (τ={GATE_TAU}) "
            f"leaves N-A's held-out detour **unchanged at {g2['detour_success_gated_rate']:.2f}** "
            f"(= raw {g2['detour_success_rate']:.2f}; clean_reach {g2['clean_reach_rate']:.2f}). So on "
            f"THIS held-out band N-A's detour successes are genuinely (near) collision-free (≤τ "
            f"contacts) — the {g2['mean_collisions']:.1f} mean contacts/ep come from the FAILED "
            f"low-bearing episodes that pin against the obstacle, not from grazing the successful "
            f"ones. The 2b premise ('N-A scored by grazing → gating sends it to ≈0') does NOT hold "
            f"here; the truthful finding is the 2a one — **this ~40° band is too easy** (within the "
            f"forager's competence), not that N-A cheats. The collision gate is still the correct "
            f"metric (it will bite once the band widens past the forager's envelope, and it already "
            f"governs the post-PPO numbers below). Reported straight, not papered over.\n"
        )
    L.append(gate2_note)
    L.append(
        f"- **Gate 4 (the headline trend).** "
        f"clean_reach {pre['clean_reach_rate']:.2f}→**{post['clean_reach_rate']:.2f}**, "
        f"gated reach {pre['reach_gated_rate']:.2f}→**{post['reach_gated_rate']:.2f}**, "
        f"gated detour {pre['detour_success_gated_rate']:.2f}→**{post['detour_success_gated_rate']:.2f}**, "
        f"raw reach {pre['reach_rate']:.2f}→**{post['reach_rate']:.2f}**, "
        f"mean_coll {pre['mean_collisions']:.1f}→**{post['mean_collisions']:.1f}** "
        f"over {g4['num_updates']} updates ({g4['wall_s']:.0f}s).\n"
        f"- **Return curve (no more `nan`):** `charts/episodic_return` = [{rc_str}] — "
        f"{'finite across all logged updates.' if returns_finite else 'WARNING: still non-finite/empty.'}\n"
    )

    L.append("\n### Verdict + recommendation\n")
    if worked:
        L.append(
            f"**The rebalance moved the trend the right way.** With `w_collide={cfg.w_collide}` + "
            f"`w_approach={cfg.w_approach}`, the collision-gated reach/detour no longer collapse to "
            f"0 pre→post (clean_reach Δ={clean_trend:+.2f}, gated reach Δ={greach_trend:+.2f}, gated "
            f"detour Δ={gdetour_trend:+.2f}) while collisions stay low. That is the signal 2b was "
            f"looking for. **Recommendation:** proceed to the full run with the *next* levers from "
            f"the 2a diagnosis — anneal `w_collide` (0.25→~0.75 once homing is stable), the far→near "
            f"obstacle curriculum, the widened bearing band + matching held-out set, and a much "
            f"larger budget — layered on top of this balance. Confirm and we'll wire those, then "
            f"launch.\n"
        )
    else:
        L.append(
            f"**The rebalance helped the balance but reach/detour are still weak at this "
            f"{g4['config']['total_steps']:,}-step budget** (post raw reach {post['reach_rate']:.2f}, "
            f"gated reach {post['reach_gated_rate']:.2f}; clean_reach Δ={clean_trend:+.2f}). Honest "
            f"diagnosis: (a) the 48k/~{g4['num_updates']}-update budget is ~8 updates — likely too "
            f"small to *show* joint avoid+home even with the right reward; (b) the warm-start forager "
            f"may need the far→near obstacle curriculum to discover the detour before the obstacle "
            f"clips its path; (c) τ={GATE_TAU} is strict on a physical env where a single brush is "
            f"common. The decomposition confirms the *balance* is now right (approach return ≫ worst "
            f"contact cost), so the next lever is **budget + curriculum**, not more reward tuning. "
            f"**Recommendation:** do not over-tune the reward; wire the curriculum + larger budget and "
            f"re-evaluate. Reported straight, not papered over.\n"
        )

    L.append(
        f"\n_Re-run: `uv run python scripts/run_rl_navigation.py --calibrate "
        f"--w-collide {cfg.w_collide} --w-approach {cfg.w_approach}`. Machine-readable results: "
        f"`scratch/nrl/calibration_rl.json`._\n"
    )

    # Append idempotently: keep everything before the marker, replace the rest.
    existing = REPORT.read_text() if REPORT.exists() else ""
    base = existing.split(REBALANCE_MARKER)[0].rstrip() + "\n"
    REPORT.write_text(base + "\n" + "".join(L))
    print(f"[cal] appended Rebalance pass (2b) to {REPORT.relative_to(ROOT)}")
    print("\n" + "=" * 70)
    print("N-RL REBALANCE (2b) GATES:")
    print(f"  Gate 1 (env sanity + ncon bounded):          {v1}")
    print(f"  Gate 2 (N-A gated yardstick honest):         {v2}")
    print(f"  Gate 3 (A/B bit-exact):                      {v3}")
    print(f"  Gate 4 (rebalance trend):                    {v4}")
    print(f"  episodic_return finite:                      {returns_finite}")
    print("=" * 70)


def _full_cmd(args) -> str:
    """The proposed --full command with the curriculum / anneal / wide-band config baked in."""
    return (
        "uv run python scripts/run_rl_navigation.py --full "
        f"--full-steps {args.full_steps} --full-n-envs {args.full_n_envs} "
        f"--w-collide {args.w_collide} --w-approach {args.w_approach} "
        f"--w-collide-max {args.w_collide_max} "
        f"--w-collide-anneal-start {args.w_collide_anneal_start} "
        f"--w-collide-anneal-end {args.w_collide_anneal_end} "
        f"--bearing-lo {args.bearing_lo:g} --bearing-hi {args.bearing_hi:g} "
        f"--bearing-target-lo {args.bearing_target_lo:g} --bearing-target-hi {args.bearing_target_hi:g} "
        f"--bearing-widen-frac {args.bearing_widen_frac:g} "
        f"--held-out-bearing-lo {args.held_out_bearing_lo:g} "
        f"--held-out-bearing-hi {args.held_out_bearing_hi:g} --held-out-n {args.held_out_n} "
        f"--obstacle-far-extra-lateral {args.obstacle_far_extra_lateral:g} "
        f"--obstacle-near-frac {args.obstacle_near_frac:g}"
    )


def _split_line(label: str, na: dict, pre: dict, post: dict, key: str) -> str:
    """One markdown table row for a split metric (inside/outside) across N-A/pre/post."""
    return (f"| {label} | {_fmt(na[key])} | {_fmt(pre[key])} | {_fmt(post[key])} |\n")


def _append_curriculum_report(args, cfg: NavRLConfig, p: dict) -> None:
    """Append (idempotently) the 'Curriculum pass (2c)' section, preserving 2a + 2b.

    2c wires the four levers (bearing curriculum, ``w_collide`` anneal, far→near
    obstacle, widened/matching held-out) and re-calibrates on the HARDER band. The
    headline is the held-out clean_reach split INSIDE vs OUTSIDE the forager's cone,
    pre→post: an upward trend on the OUTSIDE bucket is the leading indicator that the
    curriculum will keep extending the homing envelope under the full budget.
    """
    g0 = p["gate0_curriculum_selfcheck"]
    g1 = p["gate1_env_sanity"]
    g2 = p["gate2_na_baseline"]["aggregate"]
    g3 = p["gate3_ab_integrity"]
    g4 = p["gate4_short_ppo"]
    pre, post, na = g4["pre"], g4["post"], g4["na_baseline"]
    nas, pres, posts = na["split"], pre["split"], post["split"]
    cone = nas["cone_deg"]

    # Verdicts.
    g0_pass = bool(g0["pass"] and g0["off_bit_exact"])
    g1_pass = (g1["shapes_ok"] and g1["cap_dropped"] and g1["ncon_bounded"]
               and g1["vec_reset_match"] and g1["vec_max_abs_diff"] == 0.0)
    g3_pass = g3["ab_bit_exact"]
    returns_finite = bool(g4.get("returns_finite", False))
    # Gate 2 on the harder band: the widened held-out should now reach OUTSIDE the
    # forager cone, so N-A's OUTSIDE clean_reach should be ~0 (band no longer too easy).
    na_out = nas["outside_clean_reach"]
    band_now_hard = (not np.isfinite(na_out)) or na_out <= 0.25

    # The leading-edge signal: does post-PPO clean_reach (or raw reach) extend onto the
    # OUTSIDE bucket relative to the warm-start pre?
    def _up(a: float, b: float) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b) and b > a + 1e-9)
    out_cr_moved = _up(pres["outside_clean_reach"], posts["outside_clean_reach"])
    out_r_moved = _up(pres["outside_reach"], posts["outside_reach"])
    leading_edge_moved = out_cr_moved or out_r_moved
    # Dead-flat 0 outside (and no movement) is the documented no-go.
    post_out_cr = posts["outside_clean_reach"]
    post_out_r = posts["outside_reach"]
    flat_zero_outside = (
        (not np.isfinite(post_out_cr) or post_out_cr == 0.0)
        and (not np.isfinite(post_out_r) or post_out_r == 0.0)
        and not leading_edge_moved
    )

    v0 = "✅" if g0_pass else "❌"
    v1 = "✅" if g1_pass else "❌"
    v2 = "✅" if band_now_hard else "⚠️"  # ✅ = band is now genuinely hard (good for 2c)
    v3 = "✅" if g3_pass else "❌"
    v4 = "✅" if leading_edge_moved else ("⚠️" if not flat_zero_outside else "❌")

    rc = g4.get("return_curve", [])
    rc_str = ", ".join(_fmt(x) for x in rc) if rc else "(none captured)"
    full_cmd = _full_cmd(args)
    band = list(cfg.bearing_deg_range)
    horizon = max(1, args.full_steps // max(1, args.full_n_envs))

    L: list[str] = []
    L.append(CURRICULUM_MARKER + "\n")
    L.append("\n---\n\n## Curriculum pass (2c) — bearing curriculum + `w_collide` anneal + "
             "far→near obstacle + widened held-out (re-calibrate on the harder band)\n")
    L.append(
        f"2b fixed the reward balance; its real finding was that the `[20–60°]` held-out band sits "
        f"INSIDE the forager's ~40° homing cone (N-A clean-reaches {na['clean_reach_rate']:.0%} there), "
        f"so beating N-A there proves nothing. 2c wires the four levers that drive the full run toward "
        f"**omnidirectional** homing, then re-calibrates on a **harder, wider band** to confirm a "
        f"learnable signal beyond the cone. All four levers are config-gated and **default OFF → 2b "
        f"bit-exact** (Gate 0 + Gate 3). **The full run has NOT been launched.**\n"
    )

    L.append("\n### The four wired levers (schedules; Gate-0 validated)\n")
    L.append(
        f"1. **Progressive bearing curriculum** — the *training* band widens linearly from "
        f"`bearing_deg_range` → `bearing_deg_target` over the first `bearing_widen_frac` of training, "
        f"then holds. Full-run default: `[{args.bearing_lo:g}, {args.bearing_hi:g}]` → "
        f"`[{args.bearing_target_lo:g}, {args.bearing_target_hi:g}]` over the first "
        f"{args.bearing_widen_frac:g} of steps. **Schedule type: step-fraction** (progress = this "
        f"env's cumulative train steps / `curriculum_horizon_steps`, the per-env budget "
        f"`total_steps / n_envs`); not held-out-gated. Validated band sweep (20→omni example): "
        f"`{g0['bands']['0.0']}`@0 → `{g0['bands']['0.3']}`@.3 → `{g0['bands']['0.6']}`@.6.\n"
        f"2. **`w_collide` anneal** — ramps `{args.w_collide:g}` → `{args.w_collide_max:g}` over progress "
        f"`[{args.w_collide_anneal_start:g}, {args.w_collide_anneal_end:g}]`, so homing is learned "
        f"before avoidance tightens (defuses the 2a stall basin). Validated: "
        f"{g0['w_collide']['0.0']} → {g0['w_collide']['0.45']} → {g0['w_collide']['1.0']} "
        f"across progress 0/.45/1.\n"
        f"3. **Far→near obstacle curriculum** — the obstacle starts `obstacle_far_extra_lateral`="
        f"{args.obstacle_far_extra_lateral:g} world-units further OFF the path (straight line misses "
        f"the disk → homing unobstructed), sliding onto the path over the first "
        f"`obstacle_near_frac`={args.obstacle_near_frac:g} of progress. Validated lateral (base 1.0): "
        f"{g0['obstacle_lateral']['0.0']}@0 → {g0['obstacle_lateral']['0.25']}@.25 → "
        f"{g0['obstacle_lateral']['0.5']}@.5.\n"
        f"4. **Widened + matching held-out** — the frozen held-out is drawn from "
        f"`held_out_bearing_deg_range` (full run: `[{args.held_out_bearing_lo:g}, "
        f"{args.held_out_bearing_hi:g}]`, omnidirectional), still rejection-sampled **disjoint** from "
        f"training at every stage. Reported split by bearing below.\n"
        f"\n- **Gate 0 (curriculum self-check): {v0}** — default-off bit-exact="
        f"`{g0['off_bit_exact']}` (schedules collapse to base when off → 2b preserved); on-sweeps "
        f"widen/anneal/slide as configured.\n"
    )

    L.append("\n### Re-calibration setup (the harder band)\n")
    L.append(
        f"- **Band:** fixed moderately-wide `[{band[0]:g}, {band[1]:g}]` for BOTH training and the "
        f"matching frozen held-out (`held_out_n={cfg.held_out_n}`), rejection-disjoint. Chosen over a "
        f"budget-compressed curriculum because a cheap run can't widen far enough to populate the "
        f"outside bearings — a fixed wide band trains directly on them, the cleanest leading-edge "
        f"test. The curriculum *schedule* itself is validated by Gate 0 and runs in `--full`.\n"
        f"- **Budget:** `--cal-steps {g4['config']['total_steps']:,}`, `--n-envs "
        f"{g4['config']['n_envs']}`, `n_steps={g4['config']['n_steps']}` → {g4['num_updates']} updates "
        f"in {g4['wall_s']:.0f}s (larger than 2b's 48k because the wider task is harder). Reward "
        f"balance held at the 2b optimum (`w_collide={cfg.w_collide:g}`, `w_approach={cfg.w_approach:g}`; "
        f"anneal OFF for this cheap homing-first run).\n"
        f"- **Split boundary:** `FORAGER_CONE_DEG={cone:.0f}°` — held-out bearings ≤{cone:.0f}° are "
        f"INSIDE the forager's competence (Gate-2: clean ~35–56°), >{cone:.0f}° are the OUTSIDE "
        f"frontier (≈60–90°, just beyond the cone) the curriculum must win.\n"
    )

    L.append("\n### Gate results (harder band)\n")
    L.append(
        "| gate | verdict | one-line |\n|---|---|---|\n"
        f"| 0 curriculum self-check | {v0} | default-off bit-exact + on-sweeps validated |\n"
        f"| 1 env sanity + ncon bounded | {v1} | ncon_peak={g1['ncon_peak']}≤{NCON_BOUND}, "
        f"deterministic (max\\|Δ\\|={g1['vec_max_abs_diff']:.0e}), {g1['speedup']:.1f}× parallel |\n"
        f"| 2 N-A on widened held-out | {v2} | N-A clean_reach OUTSIDE {cone:.0f}° = "
        f"{_fmt(na_out)} → band is "
        + ("now genuinely hard (good) |\n" if band_now_hard else "still too easy |\n")
        + f"| 3 A/B bit-exact | {v3} | warm-start == chemo forward, max\\|Δ\\|="
        f"{g3['feeler_off_max_abs_delta']:.0e} |\n"
        f"| 4 leading edge (OUTSIDE clean_reach) | {v4} | "
        + ("moves UP pre→post (curriculum will extend it) |\n" if leading_edge_moved
           else ("non-zero but not yet rising — see diagnosis |\n" if not flat_zero_outside
                 else "dead-flat 0 outside — no-go, diagnosed below |\n"))
    )

    L.append(f"\n### Held-out clean_reach split — INSIDE vs OUTSIDE the {cone:.0f}° cone (pre→post)\n")
    L.append(
        f"| metric | N-A | pre-PPO (warm start) | post-PPO |\n|---|---|---|---|\n"
        + _split_line(f"clean_reach INSIDE ≤{cone:.0f}° (n={nas['inside_n']})",
                      nas, pres, posts, "inside_clean_reach")
        + _split_line(f"**clean_reach OUTSIDE >{cone:.0f}° (n={nas['outside_n']})**",
                      nas, pres, posts, "outside_clean_reach")
        + _split_line(f"reach (raw) INSIDE ≤{cone:.0f}°", nas, pres, posts, "inside_reach")
        + _split_line(f"reach (raw) OUTSIDE >{cone:.0f}°", nas, pres, posts, "outside_reach")
        + f"| clean_reach (all held-out) | {na['clean_reach_rate']:.2f} | "
        f"{pre['clean_reach_rate']:.2f} | {post['clean_reach_rate']:.2f} |\n"
        f"| mean_collisions | {na['mean_collisions']:.1f} | {pre['mean_collisions']:.1f} | "
        f"{post['mean_collisions']:.1f} |\n\n"
        f"The OUTSIDE bucket is the whole point: N-A clean-reaches {_fmt(nas['outside_clean_reach'])} "
        f"there (the forager cannot home past its cone), so any upward pre→post movement is signal the "
        f"narrow-cone overfit is dissolving.\n"
        f"- **Return curve (finite):** `charts/episodic_return` = [{rc_str}] — "
        f"{'finite across all logged updates.' if returns_finite else 'WARNING: non-finite/empty.'}\n"
    )

    L.append("\n### Verdict\n")
    if leading_edge_moved:
        L.append(
            f"**Green — the leading edge moves.** Post-PPO clean_reach/raw-reach on the OUTSIDE "
            f"(>{cone:.0f}°) bucket rises vs the warm-start pre (clean_reach "
            f"{_fmt(pres['outside_clean_reach'])}→**{_fmt(posts['outside_clean_reach'])}**, raw reach "
            f"{_fmt(pres['outside_reach'])}→**{_fmt(posts['outside_reach'])}**) while N-A sits at "
            f"{_fmt(nas['outside_clean_reach'])}. Even at this cheap budget PPO is extending homing "
            f"PAST the forager cone — exactly the leading indicator that the progressive curriculum "
            f"will keep widening the envelope toward omnidirectional under the full budget. "
            f"**Recommendation: launch the full run** with the curriculum/anneal/wide config baked in "
            f"(below).\n"
        )
    elif not flat_zero_outside:
        L.append(
            f"**Amber — outside bearings are reachable but not yet rising at this "
            f"{g4['config']['total_steps']:,}-step budget** (post OUTSIDE clean_reach "
            f"{_fmt(posts['outside_clean_reach'])}, raw reach {_fmt(posts['outside_reach'])}; "
            f"pre {_fmt(pres['outside_clean_reach'])}/{_fmt(pres['outside_reach'])}). There is a "
            f"non-zero base to build on; the cheap budget (~{g4['num_updates']} updates) is likely too "
            f"small to *show* the extension. The full curriculum (gradual widening from the cone, "
            f"homing-first anneal, far→near obstacle) is the mechanism to grow it. **Recommendation: "
            f"launch the full run**; the wide-band base is reachable.\n"
        )
    else:
        L.append(
            f"**Red — dead-flat 0 on the OUTSIDE bucket** (post clean_reach "
            f"{_fmt(posts['outside_clean_reach'])}, raw reach {_fmt(posts['outside_reach'])}). The "
            f"warm-start forager cannot home outside its cone and the cheap budget did not move it. "
            f"Before the full run, diagnose: (a) the warm start is insufficient at wide bearings — add "
            f"a **homing-only, obstacle-free warm-up phase** (curriculum stage 0: widen the bearing "
            f"band with NO obstacle until outside-cone reach is non-zero, THEN slide the obstacle in); "
            f"(b) raise `ent_coef` to force exploration past the cone; (c) widen more gradually. "
            f"**Do NOT launch the full run yet** — the wide-band homing signal must be non-zero first.\n"
        )

    L.append("\n### Proposed full run (NOT launched — curriculum/anneal/wide baked in)\n")
    L.append(
        f"```\n{full_cmd}\n```\n"
        f"- `--full` enables the curriculum by construction; `curriculum_horizon_steps` is set to "
        f"`full_steps / full_n_envs` = {horizon:,} per-env steps, so each async env's progress tracks "
        f"the global training fraction. Bearing widens "
        f"`[{args.bearing_lo:g},{args.bearing_hi:g}]`→`[{args.bearing_target_lo:g},"
        f"{args.bearing_target_hi:g}]`, `w_collide` anneals {args.w_collide:g}→{args.w_collide_max:g}, "
        f"obstacle slides far→near, held-out is omnidirectional "
        f"`[{args.held_out_bearing_lo:g},{args.held_out_bearing_hi:g}]` (n={args.held_out_n}).\n"
        f"- Checkpointed every {args.ckpt_every} updates and resumable, so it can span sessions.\n"
    )

    # Append idempotently: keep everything before the 2c marker, replace the rest.
    existing = REPORT.read_text() if REPORT.exists() else ""
    base = existing.split(CURRICULUM_MARKER)[0].rstrip() + "\n"
    REPORT.write_text(base + "\n" + "".join(L))
    print(f"[cal] appended Curriculum pass (2c) to {REPORT.relative_to(ROOT)}")
    print("\n" + "=" * 70)
    print("N-RL CURRICULUM (2c) GATES:")
    print(f"  Gate 0 (curriculum self-check / default-off):  {v0}")
    print(f"  Gate 1 (env sanity + ncon bounded):            {v1}")
    print(f"  Gate 2 (N-A baseline, band now hard):          {v2}")
    print(f"  Gate 3 (A/B bit-exact):                        {v3}")
    print(f"  Gate 4 (leading edge OUTSIDE cone moves):      {v4}")
    print(f"  episodic_return finite:                        {returns_finite}")
    print("=" * 70)


def _write_report(args, cfg: NavRLConfig, p: dict) -> None:
    # NOTE: this generated the v1 (2a) report. The 2b rebalance pass preserves that report
    # as history and appends a focused section via ``_append_rebalance_report`` instead of
    # regenerating, so ``calibrate()`` no longer calls this. Kept for first-time generation.
    g1, g2, g3, g4 = (p["gate1_env_sanity"], p["gate2_na_baseline"]["aggregate"],
                      p["gate3_ab_integrity"], p["gate4_short_ppo"])
    pre, post, na = g4["pre"], g4["post"], g4["na_baseline"]
    tests = p.get("tests", {})

    # Verdicts. Gates 1 & 3 are hard pass/fail (correctness). Gates 2 & 4 are
    # *findings* (⚠️) when the calibration surfaces something the full run must act on,
    # rather than a clean pass — that is the whole point of validation-first.
    g1_pass = (g1["shapes_ok"] and g1["cap_dropped"] and g1["ncon_bounded"]
               and g1["vec_reset_match"] and g1["vec_max_abs_diff"] == 0.0)
    g3_pass = g3["ab_bit_exact"]
    # Gate 2 is honest iff the OVERFIT N-A controller does NOT cleanly solve the held-out
    # set. clean_reach (reach with zero contact) is the un-gameable bar.
    held_out_too_easy = g2["clean_reach_rate"] > 0.25
    # Gate 4 net win = a harder bar moved the right way vs N-A.
    g4_win = g4["generalization_signal"]
    reach_collapsed = post["reach_rate"] < max(pre["reach_rate"], 0.01)

    v1 = "✅" if g1_pass else "❌"
    v2 = "⚠️" if held_out_too_easy else "✅"
    v3 = "✅" if g3_pass else "❌"
    v4 = "✅" if g4_win else "⚠️"

    full_steps = args.full_steps
    full_cmd = (
        f"uv run python scripts/run_rl_navigation.py --full "
        f"--full-steps {full_steps} --n-envs {args.full_n_envs} "
        f"--w-collide {cfg.w_collide}"
    )

    L: list[str] = []
    L.append("# REPORT — N-RL · Obstacle navigation via PPO: integration + calibration gate\n")
    L.append(
        "Validation-first calibration for the RL navigation stack (env + NCA policy + PPO), "
        "run on **sentry (WIN, 5900X + 3080 Ti)** with the obstacles **physically real** "
        "(N-RL-PHYS). `w_collide=%.2f` and the Newton/CG contact-solver cap dropped back to "
        "MuJoCo's default Newton (Gate 1 asserts `ncon` stays bounded). The stack wires cleanly "
        "and the harness is correct (Gates 1, 3 ✅); the calibration also surfaced two real "
        "findings the full run must act on (Gates 2, 4 ⚠️) — detailed below, not papered over. "
        "**The full run has NOT been launched.**\n" % cfg.w_collide
    )

    L.append("\n## Verdict at a glance\n")
    L.append(
        f"| gate | verdict | one-line |\n|---|---|---|\n"
        f"| 1 env sanity + ncon bounded | {v1} | shapes OK, ncon_peak={g1['ncon_peak']}≤{NCON_BOUND}, "
        f"deterministic, {g1['speedup']:.1f}× parallel |\n"
        f"| 2 N-A baseline in this env | {v2} | N-A clean-reaches {g2['clean_reach_rate']:.2f} of "
        f"held-out → **this band is too easy; widen it** |\n"
        f"| 3 A/B bit-exact | {v3} | warm-start == chemo forward, max\\|Δ\\|=0 |\n"
        f"| 4 short PPO signal | {v4} | collisions {na['mean_collisions']:.0f}→{post['mean_collisions']:.0f} "
        f"but reach →{post['reach_rate']:.2f} (**over-cautious at this budget**) |\n"
    )

    # ---- files ----
    L.append("\n## Files added / changed\n")
    L.append(
        "- **`scripts/run_rl_navigation.py`** (new) — the `--calibrate` / `--full` CLI that "
        "wires `NavRLEnv` + warm-started `NCAPolicy` + `rl.ppo.train`, with nav's held-out "
        "detour/reach/collision metrics as the PPO `eval_fn` and W&B on.\n"
        "- **`scripts/test_rl_navigation.py`** (new) — smoke test: env + policy + one real PPO "
        "update step run clean.\n"
        "- **`src/cellular_gaits/rl/nav_env.py`** (additive) — `NavRLConfig.drop_solver_cap` "
        "knob (default `False` → prior obstacle-env physics bit-exact); when `True`, "
        "`_build_fly_env` restores MuJoCo's default Newton solver (the same values the "
        "no-obstacle / chemo / loom path already uses, so those byte-exact paths are unaffected). "
        "`w_collide` was already a config field (set to %.2f here, no code change). No other "
        "source touched; `nca.py` / `env.py` / `evolve*.py` unchanged.\n" % cfg.w_collide
    )

    # ---- gates ----
    L.append("\n## The four gates\n")

    L.append(f"### {v1} Gate 1 — env sanity on the now-physical task (ncon bounded)\n")
    L.append(
        f"- obs/action spaces + in-space rollout: `{g1['shapes_ok']}`; obs in-space every step "
        f"of a {g1['rollout_steps']}-step warm-start rollout: `{g1['obs_in_space_every_step']}`.\n"
        f"- **Newton cap dropped** (per the carried finding): solver=`{g1['solver']}` (2 = Newton), "
        f"iters=`{g1['iterations']}` (uncapped). With the cap gone, **`ncon_peak = {g1['ncon_peak']}` "
        f"≤ {NCON_BOUND}** over the rollout → the contact set stays bounded, so dropping the cap "
        f"introduces no instability (matches N-RL-PHYS gate 3's ≤23). The assertion replaces the "
        f"old 'confirm the cap' check.\n"
        f"- vectorized determinism ({g1['n_envs']} async envs, same seed + action stream): reset "
        f"obs identical = `{g1['vec_reset_match']}`, max|Δ| (obs+reward) = "
        f"`{g1['vec_max_abs_diff']:.2e}` (0 = bit-deterministic).\n"
        f"- throughput (standing action, pure stepping): **{g1['parallel_steps_per_s']:.0f} steps/s** "
        f"with {g1['n_envs']} async envs vs {g1['sequential_steps_per_s']:.0f} solo → "
        f"**{g1['speedup']:.1f}×**. Reset (rebuild FlyEnv + warmup) ≈ {g1['reset_ms']:.0f} ms is the "
        f"per-episode throughput ceiling RL pays (the obstacle geoms are baked per layout).\n"
    )

    L.append(f"### {v2} Gate 2 — N-A overfit baseline, measured in THIS env (FINDING)\n")
    L.append(
        f"The N-A CMA-ES nav controller (`{g2.get('source','')}`) loaded as the policy and "
        f"evaluated deterministically on this env's frozen held-out set (n={g2['n_episodes']}):\n\n"
        f"| metric | N-A on held-out |\n|---|---|\n"
        f"| clean_reach_rate (reach, **zero** contact) | **{g2['clean_reach_rate']:.2f}** |\n"
        f"| reach_rate | {g2['reach_rate']:.2f} |\n"
        f"| detour_success_rate | {g2['detour_success_rate']:.2f} |\n"
        f"| avoid_rate (≤ {COLLISION_OK_STEPS} contacts) | {g2['avoid_rate']:.2f} |\n"
        f"| detour_correct_rate | {g2['detour_correct_rate']:.2f} |\n"
        f"| mean_collisions | {g2['mean_collisions']:.1f} |\n"
        f"| mean_min_dist | {g2['mean_min_dist']:.2f} |\n\n"
        f"- **The expected ≈0 did NOT reproduce on this held-out set — and that is itself the "
        f"finding.** N-A clean-reaches **{g2['clean_reach_rate']:.0%}** of these layouts. The "
        f"per-episode rows show why: N-A succeeds at bearings ≈35–56° and fails only at the "
        f"low-bearing edge (≈23°, where one episode pinned for ~960 steps). This env's held-out "
        f"band (`bearing_deg_range={list(cfg.bearing_deg_range)}`) **overlaps the ~40° cone the "
        f"forager (and hence N-A) is already competent in**, so it is not a hard generalization "
        f"test. The original export's held-out 0.00 came from a *harder* set (offset/near/radius "
        f"perturbations) in the *non-physical* env.\n"
        f"- **Consequence for the yardstick (acted on below):** to make 'RL beats N-A' a real "
        f"claim, the full run's held-out set must reach **outside** the forager's competence — "
        f"i.e. **widen the bearing band toward omnidirectional**. Within ~40°, CMA-ES already "
        f"does fine and there is little for RL to prove.\n"
        f"- **Physical-deflection caveat:** obstacles are now physical, so a fly that walks into "
        f"one is *passively deflected toward the open side* — a purely geometric `detour_correct` "
        f"is partly produced by physics, not skill. `clean_reach` (reach with NO contact) is the "
        f"discriminator physics cannot fake, so it is the headline metric here and below.\n"
    )

    L.append(f"### {v3} Gate 3 — A/B integrity (warm-start == chemo forward, bit-exact)\n")
    L.append(
        f"- Feeler-zeroed warm-start `mean_action` vs the trained 8-input chemo forward: "
        f"max|Δ| = `{g3['feeler_off_max_abs_delta']:.2e}`.\n"
        f"- Feelers ON (raw 1.0/0.6 × the policy's ×8 input gain) vs chemo forward: max|Δ| = "
        f"`{g3['feeler_on_max_abs_delta']:.2e}` — amplifying a feeler reading through the "
        f"zero-initialized feeler weights is still exactly zero, so A/B holds **regardless of the "
        f"gain knob** until training moves those weights.\n"
        f"- warm start: {g3['chemo_params']} chemo params → {g3['nav_params']} nav params "
        f"(== `NCA(nav=True)`).\n"
        + (
            f"- **Existing tests / prior behaviors:** {tests.get('summary', 'run separately')}\n"
            if tests else
            "- **Existing tests / prior behaviors:** see the test appendix below.\n"
        )
    )

    L.append(f"### {v4} Gate 4 — short PPO run (FINDING: avoidance learns, reach collapses)\n")
    L.append(
        f"Short PPO: `total_steps={g4['config']['total_steps']}`, `n_envs={g4['config']['n_envs']}`, "
        f"`n_steps={g4['config']['n_steps']}`, `n_minibatch={g4['config']['n_minibatch']}`, "
        f"`update_epochs={g4['config']['update_epochs']}`, `lr={g4['config']['learning_rate']}`, "
        f"`ent_coef={g4['config']['ent_coef']}` → {g4['num_updates']} updates in "
        f"{g4['wall_s']:.0f}s.\n\n"
        f"| held-out metric | N-A baseline | pre-PPO (warm start) | post-PPO |\n|---|---|---|---|\n"
        f"| clean_reach_rate | {na['clean_reach_rate']:.2f} | {pre['clean_reach_rate']:.2f} "
        f"| **{post['clean_reach_rate']:.2f}** |\n"
        f"| reach_rate | {na['reach_rate']:.2f} | {pre['reach_rate']:.2f} | "
        f"**{post['reach_rate']:.2f}** |\n"
        f"| detour_success_rate | {na['detour_success_rate']:.2f} | {pre['detour_success_rate']:.2f} "
        f"| **{post['detour_success_rate']:.2f}** |\n"
        f"| mean_collisions | {na['mean_collisions']:.1f} | {pre['mean_collisions']:.1f} | "
        f"**{post['mean_collisions']:.1f}** |\n"
        f"| avoid_rate | {na['avoid_rate']:.2f} | {pre['avoid_rate']:.2f} | "
        f"**{post['avoid_rate']:.2f}** |\n\n"
        f"- **PPO is clearly learning — in the wrong direction for now.** Over {g4['num_updates']} "
        f"updates it drove held-out collisions **{pre['mean_collisions']:.0f}→{post['mean_collisions']:.0f}** "
        f"and avoid_rate to **{post['avoid_rate']:.2f}**, but reach fell to "
        f"**{post['reach_rate']:.2f}**: at `w_collide={cfg.w_collide}` and a tiny "
        f"{g4['config']['total_steps']:,}-step budget the fastest way to cut the penalty is to "
        f"**stop advancing into the obstacle at all** (an over-cautious 'stall' basin), not to "
        f"learn the harder detour-and-home. So there is a real PPO gradient on the avoid half; the "
        f"joint avoid+home behavior needs the diagnosis below.\n"
        f"- Net win over N-A on the un-gameable bar (clean_reach/detour up, collisions down with "
        f"reach held): **{g4_win}** — collisions collapsed but reach did too, so no net win at "
        f"this budget. Reported straight, not papered over (per the master prompt).\n"
    )

    # ---- diagnosis ----
    L.append("\n## Diagnosis (Gate 4) + reward balance against the real contact scale\n")
    L.append(
        f"- **`w_collide` balance (the Gate-4 ask).** Real contacts run ~11–28 in-contact "
        f"steps/episode (N-RL-PHYS), so `w_collide={cfg.w_collide}` integrates to ~"
        f"{cfg.w_collide*11:.0f}–{cfg.w_collide*28:.0f} of penalty on a colliding episode — "
        f"comparable to the full approach (~source_distance 18) plus the one-off "
        f"`reach_bonus={cfg.reach_bonus}`. That balance makes a *clean* detour the true optimum, "
        f"but it also makes **'stall short of the obstacle' (penalty 0, partial approach) "
        f"competitive with 'push through and graze' at a short budget** — exactly the basin PPO "
        f"fell into. N-A's old 0.2 never bit because its collision count was the inflated "
        f"geometric 100+; at the real ~20, 0.2 integrates to ~4 (negligible) — confirming 0.2 is "
        f"too low and {cfg.w_collide} is in range, but needs a *ramp*.\n"
        "- **What to try in the full run (in priority order):** (1) **anneal `w_collide`** — start "
        "low (~0.2–0.3) so homing is learned first, ramp to ~0.75 once reach is stable, so the "
        "policy never discovers the stall basin before it can home; (2) **far→near obstacle "
        "curriculum** — obstacle off-path / distant early, slid onto the path as reach stabilizes; "
        "(3) **much larger budget** (the 48k calibration is ~8 updates; the joint task needs the "
        "full run); (4) optionally a small `ent_coef` (e.g. 1e-3) to keep the policy exploring "
        "past the stall basin. (5) `w_collide` sweep 0.5–1.0 once homing survives.\n"
    )

    # ---- full run proposal ----
    L.append("\n## Proposed full run (NOT launched)\n")
    L.append(f"```\n{full_cmd}\n```\n")
    cal_sps = g4["config"]["total_steps"] / max(g4["wall_s"], 1)  # observed, eval-inflated
    L.append(
        f"- **Step budget:** **{full_steps:,} env steps**. The {g4['config']['total_steps']:,}-step "
        f"calibration took {g4['wall_s']:.0f}s (~{cal_sps:.0f} steps/s) — but that rate is dominated "
        f"by held-out eval (5 evals × {na['n_episodes']} episodes of up to {cfg.max_episode_steps} "
        f"steps each on a single env); raw training throughput is the {g1['parallel_steps_per_s']:.0f} "
        f"steps/s of Gate 1, with reset (≈{g1['reset_ms']:.0f} ms, the per-episode ceiling) the real "
        f"limiter. With `eval_every` amortized over a long run, ~{full_steps:,} steps is a "
        f"**~2–6 h** job — checkpointed every {args.ckpt_every} updates and resumable, so it can "
        f"span sessions.\n"
        f"- **Bearing-width decision — RECOMMENDATION: bet WIDE, on a curriculum ramp.** Gate 2 "
        f"shows the moderate band `{list(cfg.bearing_deg_range)}` is *too easy* (N-A already "
        f"clean-reaches {g2['clean_reach_rate']:.0%}), so narrow inherits the ~40° envelope and "
        f"wastes the reason to use RL. Widen toward omnidirectional over the run (start at the "
        f"calibrated band, widen as held-out reach stabilizes) — this bets PPO **relearns "
        f"omnidirectional homing** under the broader reward, the higher-value outcome (it subsumes "
        f"the separate 'omnidirectional forager' upgrade). Risk: homing breaks outside the band "
        f"before avoidance is learned — mitigated by the warm start + the ramp + annealed "
        f"`w_collide`. **The full run must ALSO widen the held-out set to match**, or the yardstick "
        f"stays uninformative (Gate 2). (These widenings are config/curriculum changes to wire "
        f"before the full run; flagged here as the go/no-go decision, not silently applied.)\n"
        f"- The `--full` path already implements warm-start + PPO + held-out eval + checkpointing; "
        f"the curriculum ramp + `w_collide` anneal are the deltas to add before launch.\n"
    )

    # ---- what transfers to the connectome ----
    L.append("\n## What transfers to the connectome endgame\n")
    L.append(
        "- **Task-agnostic (transfers verbatim):** `rl/ppo.py` (the whole PPO loop — rollout, "
        "GAE(λ), clipped surrogate, minibatch epochs, opaque per-env recurrent-state threading, "
        "checkpoint/resume, W&B, the `eval_fn` hook) and `rl/env_base.py` (`EmbodiedRLEnv` — the "
        "policy-agnostic FlyEnv→Gymnasium wrapper with its four pluggable hooks). A future "
        "`ConnectomePolicy` (a FlyWire sub-circuit, itself stateful) drops into the same "
        "`train(cfg, make_env, agent, eval_fn)` call unchanged — the explicit state threading was "
        "built for exactly that recurrent case. `scripts/run_rl_navigation.py`'s eval/“measure a "
        "policy on a frozen held-out set” harness is reusable too.\n"
        "- **Nav-specific (rebuilt per task):** `rl/nav_env.py` (the DR sampler, obs builder, "
        "reward, held-out set) and the warm-start path in `rl/policies.py` (`NCAPolicy` wraps "
        "`NCA(nav=True)`; the connectome task supplies its own policy module). The escape behavior "
        "is where the real connectome seam (LC4/LPLC2→DNp01) lives; **navigation has no clean "
        "real-circuit seam** — the RL *harness* is the bridge, not the nav behavior.\n"
    )

    # ---- honesty ----
    L.append("\n## Honesty caveats (carry into the export meta)\n")
    L.append(
        "1. **Feelers are a hand-built rangefinder abstraction**; real *Drosophila* avoid via "
        "vision / optic flow / looming. Navigation is a robotics-flavored capability demo, not a "
        "connectome bridge.\n"
        "2. **Reactive local avoidance + gradient homing, not path planning / spatial memory** — "
        "can trap in concave / dead-end layouts (a local minimum). Not hidden.\n"
        "3. If RL **widens the homing envelope**, that is the controller *relearning steering "
        "under a broader reward*, not new biology. The DR distributions and reward weights "
        "(`w_collide`, `reach_bonus`, `step_cost`, the bearing band) are **design choices**, "
        "stated here; the policy is tuned to them.\n"
        "4. Nav return / fitness is **not comparable** across behaviors.\n"
        "5. The Gate-2 baseline is N-A on a **different** frozen held-out set than the original "
        "export, in the **now-physical** env — an honest in-env yardstick, not a re-run of the old "
        "numbers; it revealed the band is too easy (a finding, not a failure).\n"
        "6. The Gate-4 short run is 8 updates on a tiny budget — it shows a *direction* "
        "(avoidance learns fast, homing needs the curriculum/anneal), not a converged controller.\n"
    )

    L.append("\n---\n\n**Confirm and I'll launch the full run.** (Recommended: first wire the "
             "bearing-curriculum + `w_collide` anneal + widened held-out set per the findings "
             "above — say the word and I'll add those before launching.)\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(L))
    print(f"[cal] wrote {REPORT.relative_to(ROOT)}")
    print("\n" + "=" * 70)
    print("N-RL CALIBRATION GATES:")
    print(f"  Gate 1 (env sanity + ncon bounded):          {v1}")
    print(f"  Gate 2 (N-A baseline / yardstick honesty):   {v2}")
    print(f"  Gate 3 (A/B bit-exact):                      {v3}")
    print(f"  Gate 4 (PPO generalization signal):          {v4}")
    print("=" * 70)
    print(f"\nProposed full run:\n  {full_cmd}\nConfirm and I'll launch the full run.")


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #
def build_full_cfg(args) -> NavRLConfig:
    """The full-run config: the 2c curriculum ON, horizon = per-env step budget.

    The N async envs each advance ~``full_steps / full_n_envs`` control steps, so setting
    ``curriculum_horizon_steps`` to that makes each env's own progress track the global
    training fraction (the schedules are measured per-env to survive subprocess isolation).
    """
    horizon = max(1, args.full_steps // max(1, args.full_n_envs))
    return NavRLConfig(
        w_collide=args.w_collide, w_approach=args.w_approach, drop_solver_cap=True,
        curriculum=True, curriculum_horizon_steps=horizon,
        bearing_deg_range=(args.bearing_lo, args.bearing_hi),
        bearing_deg_target=(args.bearing_target_lo, args.bearing_target_hi),
        bearing_widen_frac=args.bearing_widen_frac,
        held_out_bearing_deg_range=(args.held_out_bearing_lo, args.held_out_bearing_hi),
        held_out_n=args.held_out_n,
        w_collide_max=args.w_collide_max,
        w_collide_anneal_start=args.w_collide_anneal_start,
        w_collide_anneal_end=args.w_collide_anneal_end,
        obstacle_far_extra_lateral=args.obstacle_far_extra_lateral,
        obstacle_near_frac=args.obstacle_near_frac,
    )


def full_run(args) -> None:
    cfg = build_full_cfg(args)
    policy, ws = build_warm_started_policy(cfg)
    print(f"[full] warm start {ws}  w_collide={cfg.w_collide}→{cfg.w_collide_max} (anneal "
          f"{cfg.w_collide_anneal_start}-{cfg.w_collide_anneal_end})  w_approach={cfg.w_approach}  "
          f"bearing {cfg.bearing_deg_range}→{cfg.bearing_deg_target} over {cfg.bearing_widen_frac} "
          f"(horizon {cfg.curriculum_horizon_steps:,}/env)  held_out {cfg.held_out_bearing_deg_range} "
          f"n={cfg.held_out_n}  steps={args.full_steps}")
    ppo = PPOConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        device=args.device,
        total_steps=args.full_steps,
        n_envs=args.full_n_envs,
        n_steps=args.n_steps,
        n_minibatch=args.n_minibatch,
        update_epochs=args.update_epochs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        log_every=1,
        wandb=args.wandb,
        behavior="nav",
        wandb_tags=("nrl", "full"),
        ckpt_dir=str(ROOT / "checkpoints" / "nrl" / args.exp_name),
    )
    eval_fn = make_eval_fn(cfg, device=str(resolve_device(args.device)))
    train(ppo, lambda: make_nav_env(cfg), policy, eval_fn=eval_fn,
          resume_from=str(args.resume_from) if args.resume_from else None)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true", help="Run the four gates + report, then exit.")
    p.add_argument("--full", action="store_true", help="Launch the full PPO run (green-lit only).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--exp-name", type=str, default="nrl_nav")
    p.add_argument("--w-collide", type=float, default=CALIBRATION_W_COLLIDE,
                   help="per-in-contact-step penalty (2b default 0.25; sweep band 0.2-0.4)")
    p.add_argument("--w-approach", type=float, default=CALIBRATION_W_APPROACH,
                   help="Δapproach gain weight (2b default 2.0; homing must dominate worst contact)")
    # 2c calibration band (the harder, wider fixed band for --calibrate)
    p.add_argument("--cal-bearing-lo", type=float, default=20.0,
                   help="2c calibration training+held-out bearing band low (deg)")
    p.add_argument("--cal-bearing-hi", type=float, default=90.0,
                   help="2c calibration training+held-out bearing band high (deg, > forager cone)")
    p.add_argument("--held-out-n", type=int, default=16,
                   help="held-out episodes (2c: bumped from 2b's 8 so the inside/outside split is robust)")
    p.add_argument("--obstacle-free", action="store_true",
                   help="homing-extension probe (2d): push the obstacle far off-path (no collisions "
                        "possible, feelers read ~0) to ISOLATE whether PPO can extend the warm-start's "
                        "homing cone to wide bearings — without the avoidance confound that collapsed 2c")
    # 2c curriculum (used by --full; defaults reproduce the proposed schedule)
    p.add_argument("--bearing-lo", type=float, default=20.0, help="--full curriculum START band low")
    p.add_argument("--bearing-hi", type=float, default=60.0, help="--full curriculum START band high")
    p.add_argument("--bearing-target-lo", type=float, default=0.0, help="--full curriculum TARGET low")
    p.add_argument("--bearing-target-hi", type=float, default=360.0, help="--full curriculum TARGET high")
    p.add_argument("--bearing-widen-frac", type=float, default=0.6,
                   help="fraction of training over which the band widens to target")
    p.add_argument("--held-out-bearing-lo", type=float, default=0.0,
                   help="--full widened held-out band low (omnidirectional)")
    p.add_argument("--held-out-bearing-hi", type=float, default=360.0,
                   help="--full widened held-out band high (omnidirectional)")
    p.add_argument("--w-collide-max", type=float, default=0.75, help="w_collide anneal ceiling")
    p.add_argument("--w-collide-anneal-start", type=float, default=0.2,
                   help="progress at which w_collide starts ramping up")
    p.add_argument("--w-collide-anneal-end", type=float, default=0.7,
                   help="progress at which w_collide reaches its ceiling")
    p.add_argument("--obstacle-far-extra-lateral", type=float, default=2.0,
                   help="extra |lateral| pushing the obstacle off-path at progress 0 (far→near)")
    p.add_argument("--obstacle-near-frac", type=float, default=0.5,
                   help="progress over which the obstacle slides onto the path")
    # PPO budget / shape
    p.add_argument("--cal-steps", type=int, default=120000,
                   help="calibration PPO budget (2c: ~100-150k, larger than 2b's 48k for the wider task)")
    p.add_argument("--full-steps", type=int, default=3_000_000, help="proposed full-run budget")
    p.add_argument("--n-envs", type=int, default=16, help="calibration parallel envs")
    p.add_argument("--full-n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--n-minibatch", type=int, default=20)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--no-wandb", dest="wandb", action="store_false", help="disable W&B (on by default)")
    p.add_argument("--resume-from", type=Path, default=None)
    p.set_defaults(wandb=True)
    args = p.parse_args()

    if args.calibrate:
        calibrate(args)
    elif args.full:
        full_run(args)
    else:
        p.error("pass --calibrate (validation-first) or --full (green-lit run)")


if __name__ == "__main__":
    main()
