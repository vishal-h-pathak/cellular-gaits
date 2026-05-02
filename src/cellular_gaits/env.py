"""Flygym environment wrapper for NCA-driven walking experiments.

Public surface (matches the spec in cellular-gaits-prompt.md):
    - reset() -> obs
    - step(joint_targets_unit) -> (obs, reward_terms, done, info)
    - rollout(policy, n_steps) -> (fitness, trajectory_log)

Joint target convention:
    Controllers emit a 42-vector in ``[-1, 1]``. The env clips and rescales to
    the actuator ctrlrange ``(-3.14, 3.14)`` rad, then issues one
    set_actuator_inputs call followed by ``PHYSICS_PER_CONTROL`` physics
    ticks per control step (250 Hz control over 10 kHz physics).

Fitness formula (3-second rollout):
    fitness = (thorax.x_end - thorax.x_start)
              - STABILITY_PENALTY_PER_STEP * (# control steps where thorax.z < z_threshold)
    where z_threshold = Z_FALL_FRACTION * post_warmup_thorax_z
    (adaptive to whatever Flygym's actual standing height is — empirically
    ~0.07 in Flygym's units after warmup, far below the spawn z=0.7).

The rollout's first-step thorax position is captured AFTER warmup, so the
displacement reflects only motor control, not the gravity-settling phase.
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

PHYSICS_PER_CONTROL = 40  # 0.0001s physics * 40 = 0.004s control = 250 Hz
CONTROL_DT_S = 0.004
DEFAULT_ROLLOUT_S = 3.0
DEFAULT_N_STEPS = int(round(DEFAULT_ROLLOUT_S / CONTROL_DT_S))  # 750

CTRLRANGE_RAD = 3.14
Z_FALL_FRACTION = 0.5
STABILITY_PENALTY_PER_STEP = 0.05


def _build_default_world() -> FlatGroundWorld:
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
    world.add_fly(
        fly,
        spawn_position=(0.0, 0.0, 0.7),
        spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
    )
    return world


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
    ) -> None:
        self.world = world if world is not None else _build_default_world()
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

    def _thorax_xyz(self) -> np.ndarray:
        return self.sim.get_body_positions(FLY_NAME)[THORAX_IDX].copy()

    def _obs(self) -> dict:
        return {"thorax_xyz": self._thorax_xyz(), "time": float(self.sim.time)}

    def reset(self) -> dict:
        self.sim.reset()
        self.sim.warmup()
        self._initial_thorax_xyz = self._thorax_xyz()
        self._prev_thorax_xyz = self._initial_thorax_xyz.copy()
        self._z_threshold = Z_FALL_FRACTION * float(self._initial_thorax_xyz[2])
        return self._obs()

    def step(self, joint_targets_unit: np.ndarray) -> tuple[dict, StepReward, bool, dict]:
        u = np.asarray(joint_targets_unit, dtype=np.float64).reshape(-1)
        if u.size != N_ACTUATORS:
            raise ValueError(
                f"step expected {N_ACTUATORS} targets, got {u.size}"
            )
        ctrl = np.clip(u, -1.0, 1.0) * CTRLRANGE_RAD
        self.sim.set_actuator_inputs(FLY_NAME, ActuatorType.POSITION, ctrl)
        for _ in range(PHYSICS_PER_CONTROL):
            self.sim.step()
            if self.sim.renderer is not None:
                self.sim.render_as_needed()
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
        policy: Callable[[int], np.ndarray],
        n_steps: int = DEFAULT_N_STEPS,
        on_step: Optional[Callable[[int, dict, np.ndarray], None]] = None,
    ) -> tuple[float, dict]:
        obs = self.reset()
        thorax_log: list[np.ndarray] = [obs["thorax_xyz"].copy()]
        targets_log: list[np.ndarray] = []
        below_log: list[bool] = []
        for t in range(n_steps):
            targets = np.asarray(policy(t), dtype=np.float64).reshape(-1)
            obs, reward, done, _ = self.step(targets)
            thorax_log.append(obs["thorax_xyz"].copy())
            targets_log.append(targets.copy())
            below_log.append(reward.below_threshold)
            if on_step is not None:
                on_step(t, obs, targets)
            if done:
                break
        forward_dx = float(thorax_log[-1][0] - thorax_log[0][0])
        n_below = int(sum(below_log))
        fitness = forward_dx - STABILITY_PENALTY_PER_STEP * n_below
        traj = {
            "thorax_xyz": np.stack(thorax_log, axis=0),
            "joint_targets": np.stack(targets_log, axis=0) if targets_log else np.zeros((0, N_ACTUATORS)),
            "below_threshold": np.asarray(below_log, dtype=bool),
            "forward_dx": forward_dx,
            "n_below": n_below,
            "z_threshold": self._z_threshold,
            "n_steps": len(targets_log),
        }
        return fitness, traj
