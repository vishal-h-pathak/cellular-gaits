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
    reward = Δapproach - w_collide * in_contact - step_cost
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
    # A training layout within ALL of these tolerances of a held-out layout
    # (and on the same block side) is redrawn -> genuine disjointness.
    bearing_tol_deg: float = 4.0
    frac_tol: float = 0.04
    lateral_tol: float = 0.15
    radius_tol: float = 0.1
    max_reject_tries: int = 200

    # --- reward shaping (per-step decomposition of N-A's episode fitness) ---
    w_collide: float = 0.2  # penalty per in-contact step (time-in-contact)
    # Per-step analog of NavConfig.time_penalty (4.0 * reach_frac): a constant
    # 4.0 / (max_episode_steps - 1) ~= 0.004 integrates to the same ~4 over a
    # full non-reaching episode. Encourages reaching sooner.
    step_cost: float = 0.004
    reach_bonus: float = 8.0  # paid once on collision-free arrival

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

    # ---- domain randomization ------------------------------------------- #
    def _draw_params(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        return {
            "bearing_deg": float(rng.uniform(*cfg.bearing_deg_range)),
            "frac": float(rng.uniform(*cfg.obstacle_frac_range)),
            "side": float(rng.choice((-1.0, 1.0))),
            "lateral": float(rng.uniform(*cfg.obstacle_lateral_range)),
            "radius": float(rng.uniform(*cfg.obstacle_radius_range)),
        }

    def _make_held_out(self, cfg: NavRLConfig) -> list[dict]:
        rng = np.random.default_rng(cfg.held_out_seed)
        return [self._draw_params(rng) for _ in range(cfg.held_out_n)]

    def _sample_train_params(self, rng: np.random.Generator) -> dict:
        """Draw a training layout, rejecting any too close to a held-out one."""
        for _ in range(self.cfg.max_reject_tries):
            p = self._draw_params(rng)
            if not any(_params_too_close(p, h, self.cfg) for h in self._held_out):
                return p
        # Continuous ranges make exhaustion astronomically unlikely; if it ever
        # happens, fall back to the last draw rather than loop forever.
        return p

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

        reward = approach - cfg.w_collide * float(in_contact) - cfg.step_cost
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
        }
        return float(reward), terminated, info


def make_nav_env(cfg: NavRLConfig | None = None) -> NavRLEnv:
    """Factory for one ``NavRLEnv`` (a no-arg-friendly thunk for vector envs).

    ``gymnasium.vector.AsyncVectorEnv([lambda: make_nav_env(cfg) for _ in ...])``
    gets N independent copies; each builds its own FlyEnv per reset.
    """
    return NavRLEnv(cfg)
