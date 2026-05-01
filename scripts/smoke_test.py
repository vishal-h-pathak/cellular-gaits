"""Smoke test: prove Flygym install + offscreen render pipeline work on this machine.

Builds a default Flygym 2.x Fly on a FlatGroundWorld, holds it at the neutral
pose for 1 simulated second, renders the tracking camera, and writes
``outputs/videos/smoke.mp4``.

Run from repo root:

    uv run python scripts/smoke_test.py
"""

from pathlib import Path

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
from flygym import Simulation
from flygym.utils.math import Rotation3D


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "videos"
OUT_PATH = OUT_DIR / "smoke.mp4"

SIM_DURATION_S = 1.0
CAMERA_RES = (240, 320)
PLAYBACK_SPEED = 0.2
OUTPUT_FPS = 25
FLY_NAME = "nmf"  # default Fly() name; cameras compile as "{name}/trackingcam"


def build_world() -> FlatGroundWorld:
    fly = Fly()

    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.YAW_PITCH_ROLL,
    )
    neutral_pose = KinematicPosePreset.NEUTRAL
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=50.0,
        ctrlrange=(-3.14, 3.14),
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


def main() -> None:
    world = build_world()
    sim = Simulation(world)
    sim.set_renderer(
        f"{FLY_NAME}/trackingcam",
        camera_res=CAMERA_RES,
        playback_speed=PLAYBACK_SPEED,
        output_fps=OUTPUT_FPS,
    )

    n_steps = 0
    while sim.mj_data.time < SIM_DURATION_S:
        sim.step()
        sim.render_as_needed()
        n_steps += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sim.renderer.save_video(OUT_PATH)

    sim_time = sim.mj_data.time
    n_frames = sum(len(v) for v in sim.renderer.frames.values())
    print(
        f"smoke_test: stepped {n_steps} physics ticks "
        f"({sim_time:.3f}s simulated), rendered {n_frames} frames -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
