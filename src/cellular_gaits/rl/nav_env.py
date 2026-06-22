"""``NavRLEnv`` — the obstacle-navigation task on top of ``EmbodiedRLEnv``.

This is the nav-specific half of the N-RL harness: it supplies the four
pluggable hooks (DR sampler, FlyEnv builder, observation builder, reward fn)
that ``EmbodiedRLEnv`` calls. Geometry mirrors the CMA-ES-era ``NavConfig``
(``cellular_gaits.evolve_navigation``) so this env trains the *same* task the
N-A controller was evolved on — only now per-step, over thousands of randomized
episodes instead of four fixed layouts.

THE CONTRACT (sessions 1b/1c code against this — keep it exact)
==============================================================
Observation : ``Box(float32, shape=(6, 8, 8))`` — the nav sensor grid from
    ``NCA.build_nav_sensor_map``, rebuilt every step from the *current* body
    state. Channel layout (each is an 8x8 plane; only a sub-block is non-zero):
        ch0  proprio   : 42 joint angles in the 7x6 motor block (rows 0-6,
                         cols 0-5), normalized ~[-1, 1] by ctrlrange.
        ch1  proprio   : 6 foot-contact booleans in row 7, cols 0-5.
        ch2  odor  L   : left-antenna goal-beacon concentration over the left
                         half of the motor block (cols 0-2), in [0, 1].
        ch3  odor  R   : right-antenna concentration over the right half
                         (cols 3-5), in [0, 1].
        ch4  feeler L  : left obstacle-proximity over the left half, RAW [0, 1].
        ch5  feeler R  : right obstacle-proximity over the right half, RAW [0, 1].
    Box bounds are per-channel: ch0 in [-1, 1], ch1-ch5 in [0, 1].

    NOTE (decision, flagged loud): the feelers are delivered RAW in [0, 1]. The
    CMA-ES nav rollout multiplied the feeler reading by ``feeler_input_gain``
    (8.0) *before* ``build_nav_sensor_map`` to give small feeler weights leverage
    on the bang-bang gait. That x8 is a first-layer input-weight / encoding
    choice, NOT a property of the sensory signal, so it lives on the POLICY side
    here (1b's ``NCAPolicy``, as a documented knob defaulting to 8.0), keeping the
    observation physical and bounded. Warm-start bit-exactness is unaffected
    either way (the feeler channels/weights are zero on a chemo warm-start).

Action      : ``Box(float32, low=-1, high=1, shape=(42,))`` — the motor-cell
    vector the NCA emits. It is clipped to [-1, 1] and handed straight to
    ``FlyEnv.step``, whose existing clip+rescale to the ctrlrange IS the actuator
    map (reused verbatim, so the body dynamics match the CMA-ES rollouts).

reset(seed, options) : applies domain randomization — samples the goal bearing,
    the obstacle position (a fraction along the straight start->goal line, offset
    laterally so it clips the path), the obstacle radius. ``options={"eval":
    True}`` draws from a FROZEN, fixed-seeded held-out set (cycled
    deterministically); training resets rejection-sample AWAY from that set, so
    the held-out layouts are genuinely disjoint from training while sharing the
    same distributions.

step(action) -> (obs, reward, terminated, truncated, info):
    reward = w_approach * Δapproach - w_collide * in_contact - step_cost
             (+ reach_bonus  iff  reached AND the episode has been collision-free)
    The collision-free gate on the reach bonus is the fix for N-A's "reach by
    grazing" exploit. ``terminated`` on reach or fall; ``truncated`` on the time
    limit. ``info`` carries: ``collision``, ``dist_to_goal``, ``reached``,
    ``detour_perp`` (signed perpendicular offset from the start->goal line, + =
    body-left), ``feeler_L``, ``feeler_R``, ``goal_bearing`` (body-frame, rad).

HONESTY: the feeler front-end is a hand-built rangefinder abstraction; navigation
has no clean real-circuit seam (real flies avoid via vision/optic flow). The
obstacle is placed on the *straight* start->goal line (not the warm-start fly's
curved natural path as the CMA-ES evolver did) — the cheap/honest choice under
continuous DR; ``detour_perp`` is measured against the same straight line for
consistency. DR distributions and reward weights are design choices, stated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from ..env import (
    ANTENNA_FORWARD_OFFSET,
    ANTENNA_LATERAL_OFFSET,
    DEFAULT_ODOR_LAMBDA,
    FEELER_HALF_ANGLE_DEG,
    FEELER_RANGE,
    N_ACTUATORS,
    OBSTACLE_CONTACT_CLEARANCE,
    FlyEnv,
    ObstacleField,
    OdorField,
)
from ..nca import GRID_H, GRID_W, NCA
from .env_base import EmbodiedRLEnv

# Nav episodes run ~4 s at 250 Hz (same as NavConfig.rollout_steps): time to
# home, detour, and re-home. Defined here to keep the env layer free of the
# heavy CMA-ES import (cma) that evolve_navigation pulls in.
NAV_ROLLOUT_STEPS = 1000

OBS_CHANNELS = 6
OBS_SHAPE = (OBS_CHANNELS, GRID_H, GRID_W)  # (6, 8, 8)


@dataclass
class NavRLConfig:
    """Observation / reward / domain-randomization parameters for ``NavRLEnv``.

    Geometry defaults mirror ``cellular_gaits.evolve_navigation.NavConfig`` so
    the RL task matches N-A's: ``source_distance=18``, ``obstacle_radius~2``,
    ``feeler_range=6``, ``reach_radius=3``.
    """

    # --- task geometry (mirror NavConfig) ---
    source_distance: float = 18.0
    odor_lambda: float = DEFAULT_ODOR_LAMBDA
    reach_radius: float = 3.0
    feeler_range: float = FEELER_RANGE
    feeler_half_angle_deg: float = FEELER_HALF_ANGLE_DEG
    antenna_forward: float = ANTENNA_FORWARD_OFFSET
    antenna_lateral: float = ANTENNA_LATERAL_OFFSET
    max_episode_steps: int = NAV_ROLLOUT_STEPS

    # --- domain randomization ranges (sampled fresh every reset) ---
    # Goal bearing, CCW from the +x spawn heading (deg). Default band is the
    # forager's roughly-reachable cone around the clean N-A azimuth (40 deg);
    # 1c's curriculum widens it toward omnidirectional.
    bearing_deg_range: tuple[float, float] = (20.0, 60.0)
    # Obstacle center = frac of the way along the straight start->goal line,
    # then offset |lateral| world-units to body-left (+side) / body-right
    # (-side). |lateral| < radius keeps the line clipping the disk (the fly
    # collides if it homes straight), while the side biases the dodge direction.
    obstacle_frac_range: tuple[float, float] = (0.30, 0.55)
    obstacle_lateral_range: tuple[float, float] = (0.8, 1.5)  # magnitude
    obstacle_radius_range: tuple[float, float] = (1.6, 2.4)  # centered on 2.0

    # --- held-out eval set (frozen, disjoint from train) ---
    held_out_n: int = 8
    held_out_seed: int = 20259  # fixed; reserved from the training RNG stream
    # Bearing band the FROZEN held-out is drawn from. ``None`` -> bearing_deg_range
    # (2b: held-out == the train band). 2c's curriculum sets this to the WIDENED /
    # omnidirectional range so the yardstick reaches OUTSIDE the forager's ~40° cone
    # (the Gate-2 finding: the narrow band is too easy). The held-out stays frozen +
    # rejection-sampled disjoint from training at every curriculum stage.
    held_out_bearing_deg_range: tuple[float, float] | None = None
    # A training layout within ALL of these tolerances of a held-out layout
    # (and on the same block side) is redrawn -> genuine disjointness.
    bearing_tol_deg: float = 4.0
    frac_tol: float = 0.04
    lateral_tol: float = 0.15
    radius_tol: float = 0.1
    max_reject_tries: int = 200

    # --- reward shaping (per-step decomposition of N-A's episode fitness) ---
    # Δapproach gain weight. The first calibration (2a) over-suppressed the final
    # approach: at w_collide=0.75 the per-step contact penalty dwarfed the per-step
    # approach gain, so PPO fell into a "stall short of the obstacle" basin. 2b
    # rebalances by (a) lowering w_collide and (b) weighting Δapproach UP so a
    # clean reach clearly dominates the worst plausible per-episode contact cost.
    # Over a clean reach Δapproach telescopes to (d_start - reach_radius) ~= 15, so
    # the homing return is w_approach*15 + reach_bonus; the worst plausible contact
    # cost is w_collide * ~28 in-contact steps ~= 7 at w_collide=0.25. w_approach=2.0
    # makes homing (~38) dominate the worst contact (~7) by >5x. Sweepable via --w-approach.
    w_approach: float = 2.0
    # Penalty per in-contact step (time-in-contact). 2b lowers this from the 2a
    # 0.75 to 0.25 (sweep band 0.2-0.4): real fly<->obstacle contacts run ~11-28
    # steps/episode, so 0.25 integrates to ~3-7 of penalty — enough to make a clean
    # detour the optimum, but no longer large enough to suppress the final approach.
    w_collide: float = 0.25
    # Per-step analog of NavConfig.time_penalty (4.0 * reach_frac): a constant
    # 4.0 / (max_episode_steps - 1) ~= 0.004 integrates to the same ~4 over a
    # full non-reaching episode. Encourages reaching sooner.
    step_cost: float = 0.004
    reach_bonus: float = 8.0  # paid once on collision-free arrival

    # --- progressive curriculum (2c; default OFF -> reproduces 2b bit-exact) ---
    # The full run dissolves the forager's narrow-cone overfit into OMNIDIRECTIONAL
    # homing by widening the *training* bearing band over training, annealing the
    # collision penalty up only after homing stabilizes, and sliding the obstacle
    # from off-path onto the path. All four levers are gated behind ``curriculum``:
    # with it ``False`` every schedule is a no-op and the env behaves bit-exactly
    # like 2b (same sampling RNG stream, same per-step reward, same held-out set).
    curriculum: bool = False
    # Per-env control-step horizon the schedules are measured against:
    # progress = clip(cum_train_steps / curriculum_horizon_steps, 0, 1). The driver
    # sets this to (total_steps / n_envs) so each async env's own progress tracks the
    # global training fraction (the N envs advance ~1/N of the budget apiece).
    curriculum_horizon_steps: int = 0
    # Bearing curriculum: widen the TRAINING band from ``bearing_deg_range`` (narrow,
    # the forager cone) to ``bearing_deg_target`` (omnidirectional) linearly over the
    # first ``bearing_widen_frac`` of progress, then hold at the target.
    bearing_deg_target: tuple[float, float] = (0.0, 360.0)
    bearing_widen_frac: float = 0.6
    # w_collide anneal: ramp w_collide (its base value is the homing-phase penalty)
    # up to ``w_collide_max`` linearly over progress in [anneal_start, anneal_end], so
    # the policy learns to home BEFORE avoidance tightens (avoids the 2a stall basin).
    w_collide_max: float = 0.75
    w_collide_anneal_start: float = 0.2
    w_collide_anneal_end: float = 0.7
    # Far->near obstacle curriculum: at progress 0 the obstacle is pushed
    # ``obstacle_far_extra_lateral`` world-units further off the path (so the straight
    # start->goal line misses the disk -> homing is unobstructed); the extra offset
    # ramps to 0 over the first ``obstacle_near_frac`` of progress, sliding the
    # obstacle onto the path (the real detour task) as reach stabilizes.
    obstacle_far_extra_lateral: float = 2.0
    obstacle_near_frac: float = 0.5

    # --- solver cap (obstacle envs only) ---
    # FlyEnv.__init__ caps the contact solver to CG (iters 30 / ls 20) whenever
    # obstacles are present (OBSTACLE_SOLVER*). With real fly<->obstacle contacts
    # the contact set stays small (ncon_peak <= ~23, see scratch/nrlphys gate 3)
    # and uncapped Newton was measured FASTER, so the cap is a harmless safety
    # belt that may cost throughput. ``drop_solver_cap=True`` restores MuJoCo's
    # default Newton solver (the same values the no-obstacle / chemo / loom path
    # uses, so dropping it does NOT change those byte-exact paths) for the N-RL
    # calibration; default False keeps the prior obstacle-env physics bit-exact.
    drop_solver_cap: bool = False


def _params_too_close(p: dict, h: dict, cfg: NavRLConfig) -> bool:
    """True iff train layout ``p`` is within tolerance of held-out layout ``h``."""
    return (
        p["side"] == h["side"]
        and abs(p["bearing_deg"] - h["bearing_deg"]) < cfg.bearing_tol_deg
        and abs(p["frac"] - h["frac"]) < cfg.frac_tol
        and abs(p["lateral"] - h["lateral"]) < cfg.lateral_tol
        and abs(p["radius"] - h["radius"]) < cfg.radius_tol
    )


class NavRLEnv(EmbodiedRLEnv):
    """Obstacle navigation as a Gymnasium env (see module docstring for the
    contract). Supplies the four task hooks to ``EmbodiedRLEnv``."""

    def __init__(self, cfg: NavRLConfig | None = None) -> None:
        self.cfg = cfg or NavRLConfig()
        # Curriculum bookkeeping (no-ops when cfg.curriculum is False). Each env
        # instance counts its OWN cumulative TRAINING control steps; progress =
        # that / curriculum_horizon_steps. ``_cur_eval`` marks the live episode as
        # held-out (its steps don't advance the curriculum, and its reward uses the
        # base weights, not the annealed ones).
        self._train_steps: int = 0
        self._cur_eval: bool = False

        # Per-channel observation bounds: ch0 (joint angles) in [-1, 1], the
        # rest (contacts, odor, feeler) in [0, 1]. Zeros (the unused cells) lie
        # inside every channel's range, so a built map is always in-space.
        low = np.zeros(OBS_SHAPE, dtype=np.float32)
        low[0, :, :] = -1.0
        high = np.ones(OBS_SHAPE, dtype=np.float32)
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(N_ACTUATORS,), dtype=np.float32
        )

        # Freeze the held-out layout set once (fixed seed, independent of the
        # training RNG stream). Training layouts are rejection-sampled away from
        # these in _sample_train_params.
        self._held_out: list[dict] = self._make_held_out(self.cfg)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            sample_layout=self._sample_layout,
            build_fly_env=self._build_fly_env,
            build_obs=self._build_obs,
            map_action=self._map_action,
            compute_reward=self._compute_reward,
            init_ctx=self._init_ctx,
            max_episode_steps=self.cfg.max_episode_steps,
        )

    # ---- curriculum schedules (all no-ops when cfg.curriculum is False) -- #
    def _progress(self) -> float:
        """Training progress in [0, 1] from THIS env's cumulative train steps.

        0.0 whenever the curriculum is off or the horizon is unset, so every
        schedule below collapses to its 2b base value.
        """
        h = self.cfg.curriculum_horizon_steps
        if not self.cfg.curriculum or h <= 0:
            return 0.0
        return min(1.0, self._train_steps / float(h))

    def _train_bearing_range(self, progress: float) -> tuple[float, float]:
        """The widening training bearing band at this progress (base band if off)."""
        cfg = self.cfg
        if not cfg.curriculum:
            return cfg.bearing_deg_range
        w = cfg.bearing_widen_frac
        t = 1.0 if w <= 0.0 else min(progress / w, 1.0)
        lo0, hi0 = cfg.bearing_deg_range
        lo1, hi1 = cfg.bearing_deg_target
        return (lo0 + (lo1 - lo0) * t, hi0 + (hi1 - hi0) * t)

    def effective_w_collide(self, progress: float | None = None) -> float:
        """Annealed collision penalty at this progress (base ``w_collide`` if off)."""
        cfg = self.cfg
        if not cfg.curriculum:
            return cfg.w_collide
        p = self._progress() if progress is None else progress
        s, e = cfg.w_collide_anneal_start, cfg.w_collide_anneal_end
        t = 1.0 if e <= s else min(max((p - s) / (e - s), 0.0), 1.0)
        return cfg.w_collide + (cfg.w_collide_max - cfg.w_collide) * t

    def _apply_obstacle_curriculum(self, params: dict, progress: float) -> dict:
        """Push the obstacle off-path early, slide it onto the path as reach grows."""
        cfg = self.cfg
        if not cfg.curriculum or cfg.obstacle_far_extra_lateral <= 0.0:
            return params
        n = cfg.obstacle_near_frac
        t = 1.0 if n <= 0.0 else min(progress / n, 1.0)
        extra = cfg.obstacle_far_extra_lateral * (1.0 - t)
        if extra <= 0.0:
            return params
        p = dict(params)
        p["lateral"] = params["lateral"] + extra
        return p

    # ---- domain randomization ------------------------------------------- #
    def _draw_params(
        self, rng: np.random.Generator, bearing_range: tuple[float, float] | None = None
    ) -> dict:
        cfg = self.cfg
        br = bearing_range if bearing_range is not None else cfg.bearing_deg_range
        return {
            "bearing_deg": float(rng.uniform(*br)),
            "frac": float(rng.uniform(*cfg.obstacle_frac_range)),
            "side": float(rng.choice((-1.0, 1.0))),
            "lateral": float(rng.uniform(*cfg.obstacle_lateral_range)),
            "radius": float(rng.uniform(*cfg.obstacle_radius_range)),
        }

    def _make_held_out(self, cfg: NavRLConfig) -> list[dict]:
        rng = np.random.default_rng(cfg.held_out_seed)
        # held_out_bearing_deg_range is None for 2b -> bearing_deg_range, so the draw
        # sequence (and thus the frozen set) is byte-identical to 2b when off.
        br = cfg.held_out_bearing_deg_range or cfg.bearing_deg_range
        return [self._draw_params(rng, br) for _ in range(cfg.held_out_n)]

    def _sample_train_params(self, rng: np.random.Generator) -> dict:
        """Draw a training layout, rejecting any too close to a held-out one.

        With the curriculum on, the bearing band widens with progress and the
        obstacle slides from off-path onto the path; with it off this is the exact
        2b draw (``_train_bearing_range`` returns the base band, the obstacle hook
        is a no-op), so the training RNG stream is unchanged.
        """
        progress = self._progress()
        br = self._train_bearing_range(progress)
        for _ in range(self.cfg.max_reject_tries):
            p = self._draw_params(rng, br)
            if not any(_params_too_close(p, h, self.cfg) for h in self._held_out):
                return self._apply_obstacle_curriculum(p, progress)
        # Continuous ranges make exhaustion astronomically unlikely; if it ever
        # happens, fall back to the last draw rather than loop forever.
        return self._apply_obstacle_curriculum(p, progress)

    def _layout_from_params(self, params: dict) -> dict:
        """Build the concrete world (goal + obstacle) from sampled params."""
        cfg = self.cfg
        d = cfg.source_distance
        th = np.radians(params["bearing_deg"])
        u = np.array([np.cos(th), np.sin(th)])  # straight start->goal direction
        left = np.array([-np.sin(th), np.cos(th)])  # path-left unit vector
        goal = d * u
        center = params["frac"] * d * u + params["side"] * params["lateral"] * left
        return {
            "goal_xy": (float(goal[0]), float(goal[1])),
            "obstacles": ((float(center[0]), float(center[1])),),
            "obstacle_radius": float(params["radius"]),
            "bearing_deg": float(params["bearing_deg"]),
            "block_side": float(params["side"]),
            "params": params,
        }

    def _sample_layout(
        self, rng: np.random.Generator, eval_mode: bool, eval_index: int
    ) -> dict:
        # Remember whether THIS episode is held-out: its steps must not advance the
        # curriculum and its reward must use the base (un-annealed) weights.
        self._cur_eval = bool(eval_mode)
        if eval_mode:
            params = self._held_out[eval_index % len(self._held_out)]
        else:
            params = self._sample_train_params(rng)
        return self._layout_from_params(params)

    # ---- FlyEnv construction (per episode) ------------------------------ #
    def _build_fly_env(self, layout: dict) -> FlyEnv:
        cfg = self.cfg
        fly = FlyEnv(
            obstacles=ObstacleField(
                centers=layout["obstacles"], radius=layout["obstacle_radius"]
            ),
            feeler_range=cfg.feeler_range,
            feeler_half_angle_deg=cfg.feeler_half_angle_deg,
            antenna_forward=cfg.antenna_forward,
            antenna_lateral=cfg.antenna_lateral,
        )
        # Obstacles present -> FlyEnv.__init__ already applied the CG contact-solver
        # cap. Optionally drop it back to MuJoCo's default Newton solver (matching the
        # no-obstacle path exactly) for the calibration; Gate 1 then asserts ncon stays
        # bounded so any real instability is still caught. ``opt`` is read live by
        # mj_step, so setting it on the compiled model takes effect immediately.
        if cfg.drop_solver_cap and fly._obstacle_geom_ids:
            opt = fly.sim.mj_model.opt
            opt.solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
            opt.iterations = 100
            opt.ls_iterations = 50
        fly.set_odor(OdorField(source_xy=layout["goal_xy"], lam=cfg.odor_lambda))
        return fly

    # ---- observation ---------------------------------------------------- #
    def _build_obs(self, fly: FlyEnv, layout: dict, ctx: dict) -> np.ndarray:
        ja, fc = fly.read_sensors()
        c_left, c_right = fly.read_chemo()  # odor goal-beacon at the antennae
        feeler_left, feeler_right = fly.read_feelers()  # RAW [0, 1]
        smap = NCA.build_nav_sensor_map(
            ja, fc, c_left, c_right, feeler_left, feeler_right
        )  # (1, 6, 8, 8)
        return smap.squeeze(0).numpy().astype(np.float32)

    # ---- action -> actuator map ----------------------------------------- #
    def _map_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        return np.clip(a, -1.0, 1.0)

    # ---- per-episode context -------------------------------------------- #
    def _init_ctx(self, fly: FlyEnv, layout: dict) -> dict:
        start_xy = np.asarray(fly._thorax_xyz()[:2], dtype=np.float64).copy()
        goal = np.asarray(layout["goal_xy"], dtype=np.float64)
        d_start = float(np.linalg.norm(start_xy - goal))
        return {
            "start_xy": start_xy,
            "goal_xy": goal,
            "d_start": d_start,
            "prev_dist": d_start,
            "ever_collided": False,
        }

    # ---- reward + diagnostics ------------------------------------------- #
    def _compute_reward(
        self, fly, layout, ctx, step_reward, fly_obs, action
    ) -> tuple[float, bool, dict]:
        cfg = self.cfg
        thorax_xy = np.asarray(fly_obs["thorax_xyz"][:2], dtype=np.float64)
        goal = ctx["goal_xy"]
        dist_now = float(np.linalg.norm(thorax_xy - goal))
        approach = ctx["prev_dist"] - dist_now  # Δapproach this step
        ctx["prev_dist"] = dist_now

        # In-contact: thorax within the contact clearance of an obstacle surface
        # (same definition as the CMA-ES rollout's time-in-contact collision).
        field = ObstacleField(
            centers=layout["obstacles"], radius=layout["obstacle_radius"]
        )
        in_contact = bool(
            field.min_surface_distance(thorax_xy) < OBSTACLE_CONTACT_CLEARANCE
        )
        if in_contact:
            ctx["ever_collided"] = True

        reached = bool(dist_now < cfg.reach_radius)
        fell = bool(step_reward.below_threshold)

        # Advance this env's curriculum clock on TRAINING steps only, then read the
        # annealed collision penalty at the current progress (both no-ops with the
        # curriculum off: w_collide stays cfg.w_collide, so the reward is bit-exact).
        if not self._cur_eval:
            self._train_steps += 1
        w_collide = self.effective_w_collide()
        reward = cfg.w_approach * approach - w_collide * float(in_contact) - cfg.step_cost
        collision_free = not ctx["ever_collided"]
        if reached and collision_free:
            reward += cfg.reach_bonus  # gated on collision-free arrival
        terminated = bool(reached or fell)

        # Diagnostics. goal_bearing is the goal direction in the fly's body
        # frame (0 = dead ahead, + = to the fly's left). detour_perp is the
        # signed perpendicular offset of the body from the straight start->goal
        # line (+ = body-left of the line).
        feeler_left, feeler_right = fly.read_feelers()
        yaw = fly._thorax_yaw()
        rel = goal - thorax_xy
        ang = np.arctan2(rel[1], rel[0]) - yaw
        goal_bearing = float(np.arctan2(np.sin(ang), np.cos(ang)))

        start = ctx["start_xy"]
        line = goal - start
        line_norm = float(np.linalg.norm(line))
        if line_norm > 0.0:
            u = line / line_norm
            left = np.array([-u[1], u[0]])
            detour_perp = float((thorax_xy - start) @ left)
        else:
            detour_perp = 0.0

        info = {
            "collision": in_contact,
            "dist_to_goal": dist_now,
            "reached": reached,
            "fell": fell,
            "detour_perp": detour_perp,
            "feeler_L": float(feeler_left),
            "feeler_R": float(feeler_right),
            "goal_bearing": goal_bearing,
            "ever_collided": ctx["ever_collided"],
            "approach": float(approach),
            "curriculum_progress": float(self._progress()),
            "w_collide_eff": float(w_collide),
        }
        return float(reward), terminated, info


def make_nav_env(cfg: NavRLConfig | None = None) -> gym.Env:
    """Factory for one ``NavRLEnv`` (a no-arg-friendly thunk for vector envs).

    ``gymnasium.vector.AsyncVectorEnv([lambda: make_nav_env(cfg) for _ in ...])``
    gets N independent copies; each builds its own FlyEnv per reset.

    The env is wrapped in ``RecordEpisodeStatistics`` so finished-episode return /
    length surface in ``info["episode"]``; ``AsyncVectorEnv`` then aggregates these
    into the ``infos["episode"]`` + ``_episode``-mask format the PPO loop's
    ``_collect_episode_stats`` parses (2b fix: without it ``charts/episodic_return``
    logged ``nan`` because no episode stats ever reached the loop). Eval uses
    ``NavRLEnv`` directly, so the wrapper is training-only and never touches the
    held-out metrics.
    """
    return gym.wrappers.RecordEpisodeStatistics(NavRLEnv(cfg))
