# Cellular Gaits

A neural cellular automaton (NCA), evolved with CMA-ES, drives the leg
actuators of a biomechanically realistic fruit fly simulated in
[Flygym 2.x](https://neuromechfly.org/) (MuJoCo). v1 target: one good
forward-walking gait, rendered to mp4 and ready to embed in a separate
Next.js portfolio.

## Status

All seven build steps implemented end-to-end:

1. ✅ Smoke test (`scripts/smoke_test.py`) — Flygym install + offscreen render
2. ✅ NCA module (`src/cellular_gaits/nca.py`) — 660-param 2-layer conv MLP
3. ✅ Env wrapper (`src/cellular_gaits/env.py`) — reset/step/rollout, 42 actuators
4. ✅ NCA ↔ Env integration (`scripts/test_nca_env.py`)
5. ✅ Evolution loop (`src/cellular_gaits/evolve.py`) — CMA-ES, checkpoints, CSV
6. ⏳ Full run (pop=32, gens=50) — runs in ~35 min on M5 CPU
7. ✅ Renderer (`src/cellular_gaits/render.py`) — mp4 + stable CA-state JSON

After the full run completes, render with
`uv run python scripts/render_best.py` — output lands in
`outputs/videos/best.mp4` and `outputs/ca_states_best.json` (the React
widget consumes the latter).

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

# 2. NCA module unit tests (no Flygym, no CMA-ES)
uv run python scripts/test_nca.py

# 3. Env wrapper smoke test (zeros + sinusoid dummy controllers)
uv run python scripts/test_env.py

# 4. Wire NCA into env, single rollout with random params
uv run python scripts/test_nca_env.py

# 5. Tiny end-to-end CMA-ES test (~1 minute)
uv run python scripts/run_evolution.py --pop 8 --gens 5

# 6. Full evolution (~35 minutes on M5 CPU)
uv run python scripts/run_evolution.py --pop 32 --gens 50

# 7. Render the best individual from the latest checkpoint
uv run python scripts/render_best.py             # writes outputs/videos/best.mp4
uv run python scripts/render_best.py --name g50  # custom name
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
- **Fitness formula.** `forward_x_displacement(thorax) − α · (# control steps where thorax z < z_threshold)`
  with `α = 0.05` per below-threshold step. `z_threshold` is set
  *adaptively* to half the post-warmup standing thorax z (rather than a
  hard-coded 0.3, which mismatched Flygym's actual units in this build —
  empirical standing z ≈ 1.08, so the threshold is ≈ 0.54). Documented
  in `env.py`. The first reading is taken AFTER `Simulation.warmup()` so
  the gravity-settling phase is excluded from the displacement signal.
- **Rollout length.** 3 simulated seconds at 250 Hz control rate (40
  physics ticks per control step → 750 control steps per rollout).
- **NCA architecture.** 8×8 grid, 4 channels, single shared 2-layer MLP
  implemented as a 3×3 zero-padded conv (Conv2d 4→16) followed by a 1×1
  conv (16→4). tanh between layers, output clamped to [-1, 1]. **660
  parameters total** (well under the 1000 cap). Random init draws
  state ∼ U(-0.1, 0.1).
- **NCA → joint mapping.** Channel 0 of the 7×6 motor sub-grid (rows
  0-6, cols 0-5; 42 cells, row-major) is read as targets in [-1, 1] and
  the env clips/rescales to (-3.14, 3.14) rad before
  `set_actuator_inputs`.

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
│   ├── test_nca.py         # standalone NCA unit tests
│   ├── test_env.py         # env wrapper smoke test
│   ├── test_nca_env.py     # NCA <-> env integration
│   ├── run_evolution.py
│   └── render_best.py
├── checkpoints/<run_id>/   # gitignored — gen_NN.npz per checkpoint
└── outputs/                # gitignored
    ├── videos/             # mp4
    ├── ca_states_<name>.json
    └── fitness_log_<run_id>.csv
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
