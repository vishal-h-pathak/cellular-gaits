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
    uneven_ground: bool = False, terrain_seed: int = 0
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
    ) -> None:
        if world is not None:
            self.world = world
        else:
            self.world = _build_default_world(
                uneven_ground=uneven_ground, terrain_seed=terrain_seed
            )
        self.sim = Simulation(self.world)
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
        yaw0 = self._thorax_yaw()

        thorax_log: list[np.ndarray] = [obs["thorax_xyz"].copy()]
        targets_log: list[np.ndarray] = []
        below_log: list[bool] = []
        yaw_log: list[float] = [yaw0]
        z_log: list[float] = [float(obs["thorax_xyz"][2])]
        chemo_log: list[tuple[float, float]] = []

        for t in range(n_steps):
            if pass_sensors:
                ja, fc = self.read_sensors()
                sensors = {"joint_angles_unit": ja, "foot_contacts": fc}
                if self._odor is not None:
                    cl, cr = self.read_chemo()
                    sensors["c_left"] = cl
                    sensors["c_right"] = cr
                    chemo_log.append((cl, cr))
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
        }
        return fitness, traj
