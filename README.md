# Cellular Gaits

A neural cellular automaton (NCA), evolved with CMA-ES, drives the leg
actuators of a biomechanically realistic fruit fly simulated in
[Flygym 2.x](https://neuromechfly.org/) (MuJoCo). v1 target: one good
forward-walking gait, rendered to mp4 and ready to embed in a separate
Next.js portfolio.

## Status

Work in progress. See `outputs/videos/` for the latest rendered gait and
`outputs/ca_states_best.json` for the CA state log of that rollout.

## Setup

Requires `uv`. Python 3.12 is provisioned automatically.

```sh
cd cellular-gaits
uv sync
```

CPU-only. No GPU / Warp dependencies are installed.

## Run

```sh
# 1. Sanity-check Flygym install + render pipeline (writes outputs/videos/smoke.mp4)
uv run python scripts/smoke_test.py

# 5. Tiny end-to-end CMA-ES test
uv run python scripts/run_evolution.py --pop 8 --gens 5

# 6. Full evolution
uv run python scripts/run_evolution.py --pop 32 --gens 50

# 7. Render the best individual from the latest checkpoint
uv run python scripts/render_best.py
```

## Decisions & assumptions

These resolve ambiguities in the original spec.

- **Python pin.** Spec said "3.11+", but `flygym==2.0.1` requires
  `>=3.12,<3.13`. Pinned to 3.12.
- **Flygym version.** Pinned to `flygym==2.0.1` (the FlyGym 2.x rewrite
  from March 2026, not the older `flygym-gymnasium` 1.x). Top-level fly
  class is `flygym.compose.Fly`; canonical wiring is
  `Fly → add_joints → add_actuators(position) → add_joint_sites → colorize → add_tracking_camera → FlatGroundWorld → world.add_fly → Simulation(world)`.
- **Actuator count: 42, not 18.** The spec assumes "3 DOF × 6 legs = 18
  motors", but Flygym 2.x's `LEGS_ACTIVE_ONLY` preset actuates 7 DoFs per
  leg = 42 motors. Rather than artificially down-select to 3 per leg
  (which forces an arbitrary choice and ignores actuator structure), v1
  drives all 42 actuators from the NCA. Motor cells are laid out as a
  contiguous 7×6 sub-grid of the 8×8 grid, deterministic by leg position
  and joint DoF order.
- **Position actuators @ kp=50.** Matching `launch_interactive_viewer.py`
  in the upstream flygym repo.
- **Spawn pose / contact bodies.** Pose `(0, 0, 0.7)` mm above ground,
  identity quaternion; contact preset `LEGS_THORAX_ABDOMEN_HEAD`.
- **macOS rendering.** Default GLFW/CGL backend works on Apple Silicon
  for offscreen rendering via `mujoco.Renderer`; we never invoke
  `mjpython` (only needed for the interactive viewer), so the
  `link_libpython_dylib_macos.sh` workaround is not required.
- **Fitness formula.** `forward_x_displacement(thorax) − α · (count of timesteps where thorax z drops below z_threshold)`
  with `α = 0.05` per timestep below threshold and `z_threshold = 0.3` mm
  (~half the neutral spawn height). Documented in `env.py`.
- **Rollout length.** 3 simulated seconds at the default control rate.

## Repository layout

```
cellular-gaits/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/cellular_gaits/
│   ├── __init__.py
│   ├── nca.py
│   ├── env.py
│   ├── evolve.py
│   ├── render.py
│   └── viz/
│       ├── __init__.py
│       └── state_log.py
├── scripts/
│   ├── smoke_test.py
│   ├── run_evolution.py
│   └── render_best.py
├── checkpoints/        # gitignored
└── outputs/videos/     # gitignored
```

## Output schema (stable)

`outputs/ca_states_<run_id>.json` is consumed by the React widget — keep
this stable.

```json
{
  "meta": {
    "run_id": "2026-05-01T19-37-00Z",
    "n_ticks": 750,
    "grid": [8, 8],
    "channels": 4,
    "control_dt_s": 0.004,
    "motor_cells": [[1, 1], [1, 2], "..."]
  },
  "frames": [
    [[[0.12, -0.31, 0.04, 0.55], "...64 cells per frame..."]]
  ]
}
```
