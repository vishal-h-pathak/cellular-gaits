"""Flygym environment wrapper for NCA-driven walking experiments.

Public surface:
    - reset() -> obs
    - step(joint_targets_unit) -> (obs, reward_terms, done, info)
    - rollout(policy, n_steps) -> (fitness, trajectory_log)
    - read_sensors() -> (joint_angles_unit[42], foot_contacts[6])   (C2-A)

Joint target convention:
    Controllers emit a 42-vector in ``[-1, 1]``. The env clips and rescales to
    the actuator ctrlrange ``(-3.14, 3.14)`` rad, then issues one
    set_actuator_inputs call followed by ``PHYSICS_PER_CONTROL`` physics ticks
    per control step (250 Hz control over 10 kHz physics).

Sensors (C2-A — closing the loop):
    ``read_sensors`` returns the 42 actuated joint angles (read from
    ``mj_data.actuator_length`` in the same order as ``set_actuator_inputs``,
    normalized to ~[-1, 1] by ctrlrange) and 6 per-leg foot-contact booleans.
    A closed-loop ``rollout(..., pass_sensors=True)`` feeds these to the policy
    each control step.

Perturbation (C2-A — robustness):
    A seedable lateral impulse is applied to the thorax via ``xfrc_applied``
    during a short window at a random time in a mid-rollout band. Because the
    onset / sign / magnitude are derived purely from the perturbation seed,
    open-loop and closed-loop rollouts at the same seed see *identical* shoves.
    An optional uneven-ground variant (seeded "pebble" geoms) is available
    behind ``uneven_ground=True``.

Fitness (v1, unchanged default):
    fitness = forward_dx - STABILITY_PENALTY_PER_STEP * (# steps below z thresh)
    The closed-loop evolver adds a heading-retention term and a fall penalty;
    see evolve_closed_loop.py. ``rollout`` always logs yaw and thorax_z so the
    richer fitness can be computed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from flygym import Simulation
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    Fly,
    KinematicPosePreset,
)
from flygym.utils.math import Rotation3D

FLY_NAME = "nmf"
THORAX_IDX = 0  # c_thorax is index 0 in fly.get_bodysegs_order()
N_ACTUATORS = 42  # LEGS_ACTIVE_ONLY: 7 DoFs/leg x 6 legs
N_LEGS = 6

PHYSICS_PER_CONTROL = 40  # 0.0001s physics * 40 = 0.004s control = 250 Hz
CONTROL_DT_S = 0.004
DEFAULT_ROLLOUT_S = 3.0
DEFAULT_N_STEPS = int(round(DEFAULT_ROLLOUT_S / CONTROL_DT_S))  # 750

CTRLRANGE_RAD = 3.14
Z_FALL_FRACTION = 0.5
STABILITY_PENALTY_PER_STEP = 0.05

# Chemotaxis (CH-A) — bilateral antenna geometry, in world units (~mm at fly
# scale). The two antennae sit ahead of the thorax and offset left/right; the
# left/right concentration difference is the only steering cue. Defaults are
# calibrated in the validation run and can be overridden per FlyEnv.
ANTENNA_FORWARD_OFFSET = 1.0
ANTENNA_LATERAL_OFFSET = 0.6
DEFAULT_ODOR_LAMBDA = 12.0  # decay length λ of the concentration field

# Escape / looming (X-A) — a virtual threat on a collision course generates a
# bilateral looming signal at the two eyes. The looming magnitude blends an
# *angular-size* term (a stand-in for size-tuned LPLC2) and an *expansion-rate*
# term (a stand-in for edge/expansion-tuned LC4); it is then split between the
# two eyes by the threat's bearing so a left-vs-right looming asymmetry (the
# whole point — it makes the escape *directed*) is available to the controller.
# This front-end is HAND-BUILT: it stands in for the real LC4/LPLC2 -> DNp01
# (Giant Fiber) circuit, which is the connectome endgame and is not built here.
# Defaults are calibrated in the validation run and can be overridden per FlyEnv.
LOOM_SIZE_GAIN = 1.0  # weight on the normalized angular-size term (LPLC2-like)
LOOM_EXP_GAIN = 1.0  # weight on the normalized expansion-rate term (LC4-like)
LOOM_EXP_REF = 6.0  # expansion-rate (rad/s of angular size) that normalizes to 1
# Threat stimulus defaults (world units; speed in units/s). Calibrated so an
# untrained walker gets hit but escape is achievable.
THREAT_RADIUS = 2.0  # physical radius of the looming object (sets angular size)
THREAT_SPEED = 45.0  # approach speed along the collision course (calibrated)
THREAT_START_DISTANCE = 18.0  # distance from the fly when the threat launches
THREAT_HIT_RADIUS = 3.0  # survival radius: closer than this == hit
# Target leading: the threat aims not at the fly's onset position but at where a
# constant-velocity fly would be when the threat arrives (aim = onset_pos +
# lead_distance * heading). This makes the threat a genuine collision course
# from ANY azimuth — a fly that keeps walking straight is hit, and only an
# escape maneuver (turning/fleeing off-course) breaks the intercept. Roughly
# fly_speed * (start_distance / speed); calibrated in the validation run.
THREAT_LEAD_DISTANCE = 15.0

# Navigation (N-A) — hand-built bilateral "feeler" rangefinder + physical
# obstacles. Obstacles are vertical cylinders that the fly *physically* collides
# with (built into the world at construction, like the pebbles) AND that the
# feelers sense. The feeler is a short-range proximity sensor split left/right by
# bearing exactly like the looming eye-split, so the left-vs-right feeler
# asymmetry carries which way to dodge.
#
# HONESTY: the feeler is a robotics-flavoured LIDAR-like distance sensor. Real
# Drosophila avoid obstacles via vision / optic flow / visual looming, NOT a
# rangefinder. Unlike escape (LC4/LPLC2 -> DNp01), navigation has NO clean
# real-circuit seam — it is a capability demo, not a connectome bridge.
FEELER_RANGE = 6.0  # max sensing distance to an obstacle SURFACE; beyond -> 0
FEELER_HALF_ANGLE_DEG = 100.0  # forward field half-angle; obstacles outside -> ignored
OBSTACLE_RADIUS = 2.0  # default physical radius of an obstacle cylinder (world units)
OBSTACLE_HEIGHT = 3.0  # full height of the cylinder (tall enough the fly can't climb over)
# Contact-solver cap for obstacle envs ONLY. A fly that gets pinned against a
# physical obstacle generates a large contact set; MuJoCo's default Newton solver
# then does a dense Cholesky factorization per iteration for up to 100 iterations,
# which can blow a single rollout from ~2 s to ~40 s and stall the whole evolution.
# Keeping the Newton solver (so the contact dynamics — and the calibrated feeler
# cue signs / collision headroom — are preserved) but capping the iteration count
# bounds the per-step cost: worst-case rollout drops to ~7 s with the warm-start
# cue signs unchanged. Applied ONLY when obstacles are present, so chemo/loom/
# closed-loop/v1 physics (no obstacles) are byte-for-byte unchanged.
OBSTACLE_SOLVER = 2  # mujoco.mjtSolver.mjSOL_NEWTON (the default; we only cap iterations)
OBSTACLE_SOLVER_ITERATIONS = 20
OBSTACLE_SOLVER_LS_ITERATIONS = 10
# The fly is "in contact" with an obstacle when its thorax is within this
# distance of the obstacle SURFACE (center distance - radius). Used for the
# collision count / time-in-contact penalty; the physical geom does the actual
# blocking. Roughly the fly's body half-length so pressing against a wall counts.
OBSTACLE_CONTACT_CLEARANCE = 1.0


@dataclass
class Perturbation:
    """Seedable lateral impulse on the thorax.

    magnitude   : force magnitude in MuJoCo force units (mN-scale here).
    direction   : (dx, dy) in the world plane; normalized internally. Default
                  is lateral (+y). The seed may flip the sign (see
                  randomize_sign) so a population sees shoves from both sides.
    window      : (lo, hi) fractions of the rollout; the impulse onset is drawn
                  uniformly from [lo, hi) * n_steps.
    duration_s  : how long the force is held (a short push, not a permanent
                  bias).
    seed        : fully determines onset, sign, magnitude jitter -> identical
                  shoves across controllers.
    randomize_sign : if True, the seed picks +/- on the lateral axis.
    """

    magnitude: float = 3.0
    direction: tuple[float, float] = (0.0, 1.0)
    window: tuple[float, float] = (0.4, 0.6)
    duration_s: float = 0.05
    seed: int = 0
    randomize_sign: bool = True

    def schedule(self, n_steps: int) -> "ImpulseSchedule":
        rng = np.random.default_rng(self.seed)
        lo = int(round(self.window[0] * n_steps))
        hi = max(lo + 1, int(round(self.window[1] * n_steps)))
        onset = int(rng.integers(lo, hi))
        dur_steps = max(1, int(round(self.duration_s / CONTROL_DT_S)))
        d = np.asarray(self.direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        d = d / norm if norm > 0 else np.array([0.0, 1.0])
        sign = float(rng.choice([-1.0, 1.0])) if self.randomize_sign else 1.0
        force_xy = self.magnitude * sign * d
        force = np.array([force_xy[0], force_xy[1], 0.0], dtype=np.float64)
        return ImpulseSchedule(
            onset_step=onset, end_step=onset + dur_steps, force=force
        )


@dataclass
class ImpulseSchedule:
    onset_step: int
    end_step: int  # exclusive
    force: np.ndarray  # (3,) world-frame force on thorax

    def active(self, step_idx: int) -> bool:
        return self.onset_step <= step_idx < self.end_step


def _build_default_world(
    uneven_ground: bool = False,
    terrain_seed: int = 0,
    obstacles: Optional["ObstacleField"] = None,
) -> FlatGroundWorld:
    fly = Fly()
    skel = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.YAW_PITCH_ROLL,
    )
    neutral = KinematicPosePreset.NEUTRAL
    fly.add_joints(skel, neutral_pose=neutral)
    dofs = skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        dofs,
        ActuatorType.POSITION,
        neutral_input=neutral,
        kp=50.0,
        ctrlrange=(-CTRLRANGE_RAD, CTRLRANGE_RAD),
    )
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")
    world = FlatGroundWorld()
    if uneven_ground:
        _add_pebbles(world, terrain_seed)
    if obstacles is not None and obstacles.centers:
        _add_obstacles(world, obstacles)
    world.add_fly(
        fly,
        spawn_position=(0.0, 0.0, 0.7),
        spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
    )
    return world


def _add_pebbles(
    world: FlatGroundWorld,
    seed: int,
    n_x: int = 24,
    n_y: int = 9,
    spacing: float = 0.6,
    max_height: float = 0.06,
) -> None:
    """Inject a seeded grid of small box "pebbles" to make the floor uneven.

    The fly walks roughly +x; pebbles cover a band ahead of and around the
    spawn. Heights are small (sub-millimetre at fly scale) so the task is
    "rough terrain" rather than "obstacle course". Heights/positions are fully
    determined by ``seed`` for reproducibility across controllers.
    """
    rng = np.random.default_rng(seed)
    wb = world.mjcf_root.worldbody
    x0 = -2.0 * spacing
    y0 = -((n_y - 1) / 2.0) * spacing
    for ix in range(n_x):
        for iy in range(n_y):
            h = float(rng.uniform(0.2, 1.0) * max_height)
            jx = float(rng.uniform(-0.2, 0.2) * spacing)
            jy = float(rng.uniform(-0.2, 0.2) * spacing)
            px = x0 + ix * spacing + jx
            py = y0 + iy * spacing + jy
            wb.add(
                "geom",
                name=f"pebble_{ix}_{iy}",
                type="box",
                size=[spacing * 0.45, spacing * 0.45, h],
                pos=[px, py, h],  # rests with top at z = 2h, base near floor
                rgba=[0.55, 0.52, 0.5, 1.0],
            )


def _add_obstacles(world: FlatGroundWorld, obstacles: "ObstacleField") -> None:
    """Inject physical vertical-cylinder obstacles into the world.

    Each obstacle is a collidable cylinder of radius ``obstacles.radius`` and
    height ``OBSTACLE_HEIGHT``, standing on the floor at its world ``(x, y)``.
    These are real MuJoCo geoms with default contact settings (like the pebbles),
    so the fly actually collides with them; the matching ``ObstacleField`` on the
    env senses them via the feelers.
    """
    wb = world.mjcf_root.worldbody
    half_h = 0.5 * OBSTACLE_HEIGHT
    for i, (cx, cy) in enumerate(obstacles.centers):
        wb.add(
            "geom",
            name=f"obstacle_{i}",
            type="cylinder",
            size=[float(obstacles.radius), half_h],
            pos=[float(cx), float(cy), half_h],  # base on the floor
            rgba=[0.35, 0.30, 0.45, 1.0],
        )


@dataclass
class OdorField:
    """Smooth odor/taste concentration field around a point source.

    ``C(p) = exp(-||p - source|| / lam)`` on the flat ground (x, y only). The
    field is 1 at the source and decays with characteristic length ``lam``
    (λ): the gradient is steepest within ~λ of the source and goes flat far
    away, so λ vs. start-distance is what makes the task learnable-but-non-
    trivial (calibrated in the validation run). Values are already in (0, 1],
    so they double as the normalized chemo-channel input.
    """

    source_xy: tuple[float, float]
    lam: float = DEFAULT_ODOR_LAMBDA

    def concentration(self, xy) -> float:
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        s = np.asarray(self.source_xy, dtype=np.float64).reshape(-1)[:2]
        d = float(np.linalg.norm(p - s))
        return float(np.exp(-d / self.lam))

    def distance(self, xy) -> float:
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        s = np.asarray(self.source_xy, dtype=np.float64).reshape(-1)[:2]
        return float(np.linalg.norm(p - s))


@dataclass
class ObstacleField:
    """A set of vertical-cylinder obstacles on the flat ground.

    Each obstacle is a disk of radius ``radius`` centered at a world ``(x, y)``;
    all obstacles in one field share the radius. The same field is used two ways:
    it is baked into the MuJoCo world as *physical* collidable cylinders (so the
    fly actually bumps into them) and it is *sensed* by :meth:`FlyEnv.read_feelers`
    (so the controller can steer around them). Distances are measured to the
    cylinder SURFACE (center distance - radius), clamped at 0 inside the disk.
    """

    centers: tuple[tuple[float, float], ...] = ()
    radius: float = OBSTACLE_RADIUS

    def surface_distances(self, xy) -> np.ndarray:
        """Distance from ``xy`` to each obstacle surface (>= 0; 0 inside)."""
        if not self.centers:
            return np.zeros(0, dtype=np.float64)
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        c = np.asarray(self.centers, dtype=np.float64).reshape(-1, 2)
        d_center = np.linalg.norm(c - p[None, :], axis=1)
        return np.maximum(0.0, d_center - self.radius)

    def min_surface_distance(self, xy) -> float:
        d = self.surface_distances(xy)
        return float(d.min()) if d.size else float("inf")


@dataclass
class Threat:
    """A virtual looming object on a collision course with the fly.

    The threat appears at a seeded time mid-episode and flies straight at a fixed
    ``speed`` toward the fly's position *at the moment of onset* (a collision
    course), starting ``start_distance`` away along ``azimuth_deg`` (degrees CCW
    from the world +x axis = the fly's spawn heading). It does not home: once
    launched it travels a straight line through the aim point and past it, so a
    fly that steps off the collision axis is missed. ``radius`` sets the object's
    angular size; ``hit_radius`` is the survival threshold (the fly is "hit" if
    the threat ever passes within it).

    ``seed`` fully determines the onset step (drawn from ``window`` * n_steps),
    so a given (seed, n_steps) yields identical threats across controllers and
    sequential / parallel evaluation agree exactly. The aim point is captured
    live at onset, so the stimulus is genuinely closed-loop (it targets wherever
    the fly has walked to), and that too is deterministic given the controller.
    """

    azimuth_deg: float = 0.0
    speed: float = THREAT_SPEED
    radius: float = THREAT_RADIUS
    start_distance: float = THREAT_START_DISTANCE
    hit_radius: float = THREAT_HIT_RADIUS
    lead_distance: float = THREAT_LEAD_DISTANCE
    window: tuple[float, float] = (0.2, 0.4)
    seed: int = 0

    def onset_step(self, n_steps: int) -> int:
        rng = np.random.default_rng(self.seed)
        lo = int(round(self.window[0] * n_steps))
        hi = max(lo + 1, int(round(self.window[1] * n_steps)))
        return int(rng.integers(lo, hi))


@dataclass
class StepReward:
    fwd_dx: float
    below_threshold: bool
    thorax_z: float


class FlyEnv:
    def __init__(
        self,
        world: Optional[FlatGroundWorld] = None,
        renderer_camera: Optional[str] = None,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        uneven_ground: bool = False,
        terrain_seed: int = 0,
        antenna_forward: float = ANTENNA_FORWARD_OFFSET,
        antenna_lateral: float = ANTENNA_LATERAL_OFFSET,
        loom_size_gain: float = LOOM_SIZE_GAIN,
        loom_exp_gain: float = LOOM_EXP_GAIN,
        loom_exp_ref: float = LOOM_EXP_REF,
        obstacles: "ObstacleField | list | tuple | None" = None,
        obstacle_radius: float = OBSTACLE_RADIUS,
        feeler_range: float = FEELER_RANGE,
        feeler_half_angle_deg: float = FEELER_HALF_ANGLE_DEG,
    ) -> None:
        # Normalize the obstacle layout to an ObstacleField (the geoms baked into
        # the world AND the sensed field share one source of truth).
        if obstacles is not None and not isinstance(obstacles, ObstacleField):
            obstacles = ObstacleField(
                centers=tuple(tuple(c) for c in obstacles), radius=obstacle_radius
            )
        if world is not None:
            self.world = world
        else:
            self.world = _build_default_world(
                uneven_ground=uneven_ground,
                terrain_seed=terrain_seed,
                obstacles=obstacles,
            )
        self.sim = Simulation(self.world)
        # Obstacle envs only: cap the contact-constraint solver so a pinned fly
        # cannot blow up per-step cost (see OBSTACLE_SOLVER constants). opt is read
        # live by mj_step, so setting it on the compiled model takes effect.
        if obstacles is not None and obstacles.centers:
            opt = self.sim.mj_model.opt
            opt.solver = OBSTACLE_SOLVER
            opt.iterations = OBSTACLE_SOLVER_ITERATIONS
            opt.ls_iterations = OBSTACLE_SOLVER_LS_ITERATIONS
        if renderer_camera is not None:
            self.sim.set_renderer(
                renderer_camera,
                camera_res=camera_res,
                playback_speed=playback_speed,
                output_fps=output_fps,
            )
        self._initial_thorax_xyz: Optional[np.ndarray] = None
        self._prev_thorax_xyz: Optional[np.ndarray] = None
        self._z_threshold: float = 0.0

        # Cache the 42 position-actuator ids (ordered as set_actuator_inputs)
        # and the thorax body id for sensor reads and impulse application.
        self._pos_actuator_ids = np.asarray(
            self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][
                FLY_NAME
            ],
            dtype=np.int64,
        )
        self._thorax_body_id = int(
            self.sim._internal_bodyids_by_fly[FLY_NAME][THORAX_IDX]
        )

        # Perturbation state (set via set_perturbation / configured at rollout).
        self._perturbation: Optional[Perturbation] = None
        self._impulse: Optional[ImpulseSchedule] = None
        self._step_idx: int = 0

        # Chemotaxis state (set via set_odor).
        self._odor: Optional[OdorField] = None
        self._antenna_forward = float(antenna_forward)
        self._antenna_lateral = float(antenna_lateral)

        # Escape / looming state (set via set_threat). The aim point is captured
        # live at onset; _prev_theta tracks angular size for the expansion rate.
        self._threat: Optional[Threat] = None
        self._threat_onset: int = -1
        self._threat_aim: Optional[np.ndarray] = None  # lead-point captured at onset
        self._threat_dir: Optional[np.ndarray] = None  # unit vec fly->threat origin
        self._prev_theta: float = 0.0
        self._loom_size_gain = float(loom_size_gain)
        self._loom_exp_gain = float(loom_exp_gain)
        self._loom_exp_ref = float(loom_exp_ref)

        # Navigation state (obstacles + feelers). ``_obstacles`` is the SENSED
        # field; the physical geoms are fixed in the world at construction, so a
        # later set_obstacles only re-points the feelers (use it with a layout
        # matching the geoms that were built).
        self._obstacles: Optional[ObstacleField] = obstacles
        self._feeler_range = float(feeler_range)
        self._feeler_half_angle = float(np.radians(feeler_half_angle_deg))

    # ---- geometry / state helpers -------------------------------------------
    def _thorax_xyz(self) -> np.ndarray:
        return self.sim.get_body_positions(FLY_NAME)[THORAX_IDX].copy()

    def _thorax_yaw(self) -> float:
        """Yaw (rotation about world z) of the thorax, radians.

        Quaternion is (w, x, y, z). Heading retention measures how close yaw
        stays to its post-warmup value after the shove.
        """
        q = np.asarray(self.sim.get_body_rotations(FLY_NAME)[THORAX_IDX])
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def read_sensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_angles_unit[42], foot_contacts[6]).

        Joint angles come from actuator transmission length (= joint angle for
        a 1-DoF position actuator), in the same order as set_actuator_inputs,
        normalized to ~[-1, 1] by ctrlrange and clipped. Foot contacts are the
        6 per-leg booleans from get_ground_contact_info.
        """
        lengths = np.asarray(self.sim.mj_data.actuator_length, dtype=np.float64)
        ja = lengths[self._pos_actuator_ids] / CTRLRANGE_RAD
        ja = np.clip(ja, -1.0, 1.0)
        contacts = np.asarray(
            self.sim.get_ground_contact_info(FLY_NAME)[0], dtype=np.float64
        ).reshape(-1)[:N_LEGS]
        contacts = (contacts > 0.5).astype(np.float64)
        return ja, contacts

    # ---- chemotaxis ---------------------------------------------------------
    def set_odor(self, odor: Optional[OdorField]) -> None:
        self._odor = odor

    def antenna_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame (x, y) of the left and right antennae.

        Derived from the thorax position and yaw: body-forward is
        ``(cos yaw, sin yaw)`` and body-left is ``(-sin yaw, cos yaw)``. Each
        antenna sits ``antenna_forward`` ahead of the thorax and
        ``antenna_lateral`` to its left / right.
        """
        thorax = self._thorax_xyz()[:2]
        yaw = self._thorax_yaw()
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        base = thorax + self._antenna_forward * fwd
        pos_l = base + self._antenna_lateral * left
        pos_r = base - self._antenna_lateral * left
        return pos_l, pos_r

    def read_chemo(self) -> tuple[float, float]:
        """Return (c_left, c_right) odor concentrations at the two antennae.

        Returns (0, 0) when no odor field is set, so a chemo policy run without
        a source sees a flat (zero) cue.
        """
        if self._odor is None:
            return 0.0, 0.0
        pos_l, pos_r = self.antenna_positions()
        return self._odor.concentration(pos_l), self._odor.concentration(pos_r)

    # ---- escape / looming ---------------------------------------------------
    def set_threat(self, threat: Optional[Threat]) -> None:
        self._threat = threat

    def _arm_threat(self, n_steps: int) -> None:
        if self._threat is not None:
            self._threat_onset = self._threat.onset_step(n_steps)
        else:
            self._threat_onset = -1
        self._threat_aim = None
        self._threat_dir = None
        self._prev_theta = 0.0

    def _capture_threat_geometry(self) -> None:
        """At onset, fix the lead aim point and the launch direction.

        ``azimuth_deg`` is interpreted in the fly's onset heading frame (0 =
        straight ahead, +90 = the fly's left), so "left"/"right"/"front" are
        relative to where the fly is actually facing. The aim point leads the fly
        by ``lead_distance`` along its heading so the threat intercepts a fly that
        keeps walking straight.
        """
        fly_xy = self._thorax_xyz()[:2]
        yaw = self._thorax_yaw()
        fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        a = np.radians(self._threat.azimuth_deg)
        # direction from the (lead) aim point out to where the threat starts,
        # rotated into the onset heading frame.
        d_local = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
        c, s = np.cos(yaw), np.sin(yaw)
        self._threat_dir = np.array(
            [c * d_local[0] - s * d_local[1], s * d_local[0] + c * d_local[1]],
            dtype=np.float64,
        )
        self._threat_aim = fly_xy + self._threat.lead_distance * fwd

    def threat_position(self, step_idx: int) -> Optional[np.ndarray]:
        """World (x, y) of the threat at ``step_idx`` (None before onset).

        The threat launches ``start_distance`` away along the captured launch
        direction from the lead aim point and travels straight toward it at
        ``speed``, continuing past. Requires the geometry to have been captured
        (it is, on the first sensed step >= onset).
        """
        if (
            self._threat is None
            or self._threat_onset < 0
            or step_idx < self._threat_onset
            or self._threat_aim is None
            or self._threat_dir is None
        ):
            return None
        elapsed_s = (step_idx - self._threat_onset) * CONTROL_DT_S
        traveled = self._threat.speed * elapsed_s
        # launch = aim + start_distance * dir; moving toward aim along -dir.
        return self._threat_aim + (self._threat.start_distance - traveled) * self._threat_dir

    def read_loom(self) -> tuple[float, float, dict]:
        """Bilateral looming signal (loom_L, loom_R) plus diagnostics.

        Returns ``(loom_L, loom_R, info)``. With no threat, or before onset,
        returns ``(0, 0, ...)``. On the first sensed step at/after onset the aim
        point is captured (fly xy now) and the expansion rate is initialized to 0
        so there is no spurious onset spike.

        Looming magnitude ``m`` blends a normalized angular-size term and a
        normalized positive expansion-rate term:
            theta = 2*atan2(R, d)            angular size of the object
            size_norm = theta / pi
            rate = max(0, dtheta/dt)         only expansion (approach) counts
            exp_norm = min(1, rate / exp_ref)
            m = clip(size_gain*size_norm + exp_gain*exp_norm, 0, 1)
        and is split between the eyes by the threat's body-frame bearing ``phi``
        (CCW positive = to the fly's left):
            loom_L = m * 0.5*(1 + sin(phi))   loom_R = m * 0.5*(1 - sin(phi))
        A frontal threat (phi=0) excites both eyes equally; a threat directly to
        one side excites only that eye. The L/R difference is the directional cue.
        """
        zero_info = {"theta": 0.0, "d": float("inf"), "phi": 0.0, "m": 0.0,
                     "threat_xy": None, "present": False}
        if self._threat is None or self._threat_onset < 0:
            return 0.0, 0.0, zero_info
        step = self._step_idx
        if step < self._threat_onset:
            return 0.0, 0.0, zero_info
        just_armed = self._threat_aim is None
        if just_armed:
            self._capture_threat_geometry()
        fly_xy = self._thorax_xyz()[:2]
        tpos = self.threat_position(step)
        if tpos is None:
            return 0.0, 0.0, zero_info
        rel = tpos - fly_xy
        d = float(np.linalg.norm(rel))
        theta = 2.0 * float(np.arctan2(self._threat.radius, max(d, 1e-9)))
        if just_armed:
            self._prev_theta = theta  # no expansion spike on the onset frame
        rate = max(0.0, (theta - self._prev_theta) / CONTROL_DT_S)
        self._prev_theta = theta
        size_norm = theta / np.pi
        exp_norm = min(1.0, rate / self._loom_exp_ref) if self._loom_exp_ref > 0 else 0.0
        m = float(
            np.clip(
                self._loom_size_gain * size_norm + self._loom_exp_gain * exp_norm,
                0.0,
                1.0,
            )
        )
        yaw = self._thorax_yaw()
        phi = float(np.arctan2(rel[1], rel[0]) - yaw)
        phi = float(np.arctan2(np.sin(phi), np.cos(phi)))  # wrap to [-pi, pi]
        s = np.sin(phi)
        loom_l = m * 0.5 * (1.0 + s)
        loom_r = m * 0.5 * (1.0 - s)
        info = {
            "theta": theta,
            "d": d,
            "phi": phi,
            "m": m,
            "threat_xy": [float(tpos[0]), float(tpos[1])],
            "present": True,
        }
        return float(loom_l), float(loom_r), info

    # ---- navigation / feelers -----------------------------------------------
    def set_obstacles(self, obstacles: "ObstacleField | list | tuple | None") -> None:
        """Set the SENSED obstacle field (re-points the feelers).

        The physical collidable geoms are baked into the world at construction;
        this only updates what the feelers report, so the layout passed here
        should match the geoms that were built (use ``FlyEnv(obstacles=...)`` to
        get both). Passing a plain list of centers wraps it in an ObstacleField
        with the default radius.
        """
        if obstacles is not None and not isinstance(obstacles, ObstacleField):
            obstacles = ObstacleField(centers=tuple(tuple(c) for c in obstacles))
        self._obstacles = obstacles

    def read_feelers(self) -> tuple[float, float]:
        """Bilateral short-range feeler (obstacle proximity): (feeler_L, feeler_R).

        From the thorax, over the forward visual field, find the NEAREST obstacle
        within ``feeler_range`` of its surface and inside the forward half-angle,
        and report its proximity ``clip(1 - d_surface / feeler_range, 0, 1)`` (a
        close wall -> near 1, nothing in range -> 0). The proximity is split
        between the two feelers by the obstacle's body-frame bearing ``phi`` (CCW
        positive = to the fly's left), exactly like the looming eye-split:

            feeler_L = p * 0.5*(1 + sin(phi))   feeler_R = p * 0.5*(1 - sin(phi))

        An obstacle dead-ahead (phi=0) excites both feelers equally; an offset
        obstacle weights the near side, so the L-R feeler difference carries which
        way to dodge. Returns (0, 0) when no obstacles are set or none are in the
        forward field — so a nav policy run with no obstacles == the pure forager.
        """
        if self._obstacles is None or not self._obstacles.centers:
            return 0.0, 0.0
        fly_xy = self._thorax_xyz()[:2]
        yaw = self._thorax_yaw()
        c = np.asarray(self._obstacles.centers, dtype=np.float64).reshape(-1, 2)
        rel = c - fly_xy[None, :]
        d_surf = np.maximum(0.0, np.linalg.norm(rel, axis=1) - self._obstacles.radius)
        phi = np.arctan2(rel[:, 1], rel[:, 0]) - yaw
        phi = np.arctan2(np.sin(phi), np.cos(phi))  # wrap to [-pi, pi]
        in_field = (d_surf < self._feeler_range) & (np.abs(phi) <= self._feeler_half_angle)
        if not np.any(in_field):
            return 0.0, 0.0
        idx = np.where(in_field)[0]
        nearest = idx[int(np.argmin(d_surf[idx]))]
        prox = float(np.clip(1.0 - d_surf[nearest] / self._feeler_range, 0.0, 1.0))
        s = float(np.sin(phi[nearest]))
        return prox * 0.5 * (1.0 + s), prox * 0.5 * (1.0 - s)

    def _obs(self) -> dict:
        return {"thorax_xyz": self._thorax_xyz(), "time": float(self.sim.time)}

    # ---- perturbation -------------------------------------------------------
    def set_perturbation(self, pert: Optional[Perturbation]) -> None:
        self._perturbation = pert

    @property
    def impulse(self) -> Optional[ImpulseSchedule]:
        return self._impulse

    def _arm_impulse(self, n_steps: int) -> None:
        self._impulse = (
            self._perturbation.schedule(n_steps)
            if self._perturbation is not None
            else None
        )

    def _apply_xfrc(self) -> None:
        f = np.zeros(3)
        if self._impulse is not None and self._impulse.active(self._step_idx):
            f = self._impulse.force
        self.sim.mj_data.xfrc_applied[self._thorax_body_id, :3] = f

    # ---- core loop ----------------------------------------------------------
    def reset(self) -> dict:
        self.sim.reset()
        self.sim.warmup()
        self._initial_thorax_xyz = self._thorax_xyz()
        self._prev_thorax_xyz = self._initial_thorax_xyz.copy()
        self._z_threshold = Z_FALL_FRACTION * float(self._initial_thorax_xyz[2])
        self._step_idx = 0
        self.sim.mj_data.xfrc_applied[self._thorax_body_id, :3] = 0.0
        # Reset per-rollout threat run-state (onset is (re)armed by rollout).
        self._threat_aim = None
        self._threat_dir = None
        self._prev_theta = 0.0
        return self._obs()

    def step(self, joint_targets_unit: np.ndarray) -> tuple[dict, StepReward, bool, dict]:
        u = np.asarray(joint_targets_unit, dtype=np.float64).reshape(-1)
        if u.size != N_ACTUATORS:
            raise ValueError(f"step expected {N_ACTUATORS} targets, got {u.size}")
        ctrl = np.clip(u, -1.0, 1.0) * CTRLRANGE_RAD
        self.sim.set_actuator_inputs(FLY_NAME, ActuatorType.POSITION, ctrl)
        self._apply_xfrc()
        for _ in range(PHYSICS_PER_CONTROL):
            self.sim.step()
            if self.sim.renderer is not None:
                self.sim.render_as_needed()
        self._step_idx += 1
        thorax = self._thorax_xyz()
        prev = self._prev_thorax_xyz if self._prev_thorax_xyz is not None else thorax
        reward = StepReward(
            fwd_dx=float(thorax[0] - prev[0]),
            below_threshold=bool(thorax[2] < self._z_threshold),
            thorax_z=float(thorax[2]),
        )
        self._prev_thorax_xyz = thorax
        obs = {"thorax_xyz": thorax, "time": float(self.sim.time)}
        return obs, reward, False, {}

    def rollout(
        self,
        policy: Callable[..., np.ndarray],
        n_steps: int = DEFAULT_N_STEPS,
        on_step: Optional[Callable[[int, dict, np.ndarray], None]] = None,
        pass_sensors: bool = False,
    ) -> tuple[float, dict]:
        """Run a deterministic rollout.

        ``pass_sensors=False`` (default, v1-compatible): policy is called as
        ``policy(t)``. ``pass_sensors=True``: policy is called as
        ``policy(t, sensors)`` where ``sensors`` is
        ``{"joint_angles_unit": (42,), "foot_contacts": (6,)}`` read *before*
        the control step (so the policy reacts to the current body state).

        Always logs yaw and thorax_z so heading-retention / fall metrics can be
        computed downstream. If a perturbation is set, it is armed for this
        rollout (onset derived from its seed + n_steps).
        """
        obs = self.reset()
        self._arm_impulse(n_steps)
        self._arm_threat(n_steps)
        yaw0 = self._thorax_yaw()

        thorax_log: list[np.ndarray] = [obs["thorax_xyz"].copy()]
        targets_log: list[np.ndarray] = []
        below_log: list[bool] = []
        yaw_log: list[float] = [yaw0]
        z_log: list[float] = [float(obs["thorax_xyz"][2])]
        chemo_log: list[tuple[float, float]] = []
        loom_log: list[tuple[float, float]] = []  # (loom_L, loom_R) per step
        threat_log: list = []  # threat xy (or None before onset) per step
        threat_dist_log: list[float] = []  # fly<->threat distance per step
        feeler_log: list[tuple[float, float]] = []  # (feeler_L, feeler_R) per step

        for t in range(n_steps):
            if pass_sensors:
                ja, fc = self.read_sensors()
                sensors = {"joint_angles_unit": ja, "foot_contacts": fc}
                if self._odor is not None:
                    cl, cr = self.read_chemo()
                    sensors["c_left"] = cl
                    sensors["c_right"] = cr
                    chemo_log.append((cl, cr))
                if self._threat is not None:
                    lL, lR, linfo = self.read_loom()
                    sensors["loom_left"] = lL
                    sensors["loom_right"] = lR
                    loom_log.append((lL, lR))
                    threat_log.append(linfo["threat_xy"])
                    threat_dist_log.append(linfo["d"])
                if self._obstacles is not None:
                    fL, fR = self.read_feelers()
                    sensors["feeler_left"] = fL
                    sensors["feeler_right"] = fR
                    feeler_log.append((fL, fR))
                targets = np.asarray(
                    policy(t, sensors), dtype=np.float64
                ).reshape(-1)
            else:
                targets = np.asarray(policy(t), dtype=np.float64).reshape(-1)
            obs, reward, done, _ = self.step(targets)
            thorax_log.append(obs["thorax_xyz"].copy())
            targets_log.append(targets.copy())
            below_log.append(reward.below_threshold)
            yaw_log.append(self._thorax_yaw())
            z_log.append(reward.thorax_z)
            if on_step is not None:
                on_step(t, obs, targets)
            if done:
                break

        forward_dx = float(thorax_log[-1][0] - thorax_log[0][0])
        n_below = int(sum(below_log))
        fitness = forward_dx - STABILITY_PENALTY_PER_STEP * n_below

        # Chemotaxis bookkeeping: start/end distance to source and the cL/cR log.
        if self._odor is not None:
            dist_start = self._odor.distance(thorax_log[0][:2])
            dist_end = self._odor.distance(thorax_log[-1][:2])
            source_xy = list(self._odor.source_xy)
            chemo_arr = np.asarray(chemo_log, dtype=np.float64) if chemo_log else np.zeros((0, 2))
        else:
            dist_start = dist_end = float("nan")
            source_xy = None
            chemo_arr = np.zeros((0, 2))

        # Escape bookkeeping: per-step loom, threat path, closest threat approach,
        # and the hit / survival outcome.
        if self._threat is not None:
            loom_arr = np.asarray(loom_log, dtype=np.float64) if loom_log else np.zeros((0, 2))
            finite_d = [d for d in threat_dist_log if np.isfinite(d)]
            threat_min_dist = float(min(finite_d)) if finite_d else float("inf")
            hit = bool(threat_min_dist < self._threat.hit_radius)
            threat_path = [
                ([float(p[0]), float(p[1])] if p is not None else None)
                for p in threat_log
            ]
            threat_meta = {
                "azimuth_deg": float(self._threat.azimuth_deg),
                "speed": float(self._threat.speed),
                "radius": float(self._threat.radius),
                "start_distance": float(self._threat.start_distance),
                "hit_radius": float(self._threat.hit_radius),
                "onset_step": int(self._threat_onset),
                "aim_xy": (
                    [float(self._threat_aim[0]), float(self._threat_aim[1])]
                    if self._threat_aim is not None
                    else None
                ),
            }
        else:
            loom_arr = np.zeros((0, 2))
            threat_min_dist = float("nan")
            hit = False
            threat_path = None
            threat_meta = None

        # Navigation bookkeeping: per-step feeler log, collision count
        # (time-in-contact: steps the thorax is within the contact clearance of
        # an obstacle surface), and the closest approach to any obstacle.
        if self._obstacles is not None and self._obstacles.centers:
            feeler_arr = (
                np.asarray(feeler_log, dtype=np.float64) if feeler_log else np.zeros((0, 2))
            )
            post = np.stack(thorax_log[1:], axis=0)[:, :2] if len(thorax_log) > 1 else np.zeros((0, 2))
            surf = np.array(
                [self._obstacles.min_surface_distance(p) for p in post], dtype=np.float64
            )
            in_contact = surf < OBSTACLE_CONTACT_CLEARANCE
            collision_count = int(in_contact.sum())
            collided = bool(collision_count > 0)
            obstacle_min_surface = float(surf.min()) if surf.size else float("inf")
            obstacle_meta = {
                "centers": [[float(x), float(y)] for (x, y) in self._obstacles.centers],
                "radius": float(self._obstacles.radius),
                "feeler_range": float(self._feeler_range),
                "contact_clearance": float(OBSTACLE_CONTACT_CLEARANCE),
            }
        else:
            feeler_arr = np.zeros((0, 2))
            collision_count = 0
            collided = False
            obstacle_min_surface = float("nan")
            obstacle_meta = None

        traj = {
            "thorax_xyz": np.stack(thorax_log, axis=0),
            "joint_targets": np.stack(targets_log, axis=0)
            if targets_log
            else np.zeros((0, N_ACTUATORS)),
            "below_threshold": np.asarray(below_log, dtype=bool),
            "yaw": np.asarray(yaw_log, dtype=np.float64),
            "yaw0": yaw0,
            "thorax_z": np.asarray(z_log, dtype=np.float64),
            "forward_dx": forward_dx,
            "n_below": n_below,
            "z_threshold": self._z_threshold,
            "n_steps": len(targets_log),
            "impulse_onset_step": (
                int(self._impulse.onset_step) if self._impulse is not None else -1
            ),
            "impulse_end_step": (
                int(self._impulse.end_step) if self._impulse is not None else -1
            ),
            "impulse_force": (
                self._impulse.force.tolist() if self._impulse is not None else None
            ),
            "odor_source_xy": source_xy,
            "dist_start": dist_start,
            "dist_end": dist_end,
            "chemo": chemo_arr,
            "loom": loom_arr,
            "threat_path": threat_path,
            "threat_min_dist": threat_min_dist,
            "threat_hit": hit,
            "threat": threat_meta,
            "feeler": feeler_arr,
            "collision_count": collision_count,
            "collided": collided,
            "obstacle_min_surface": obstacle_min_surface,
            "obstacles": obstacle_meta,
        }
        return fitness, traj
