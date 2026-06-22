# Claude Code prompt — A · Export the fly MJCF for the browser

> **Run from the `cellular-gaits` repo root** (uv project, Python 3.12, FlyGym 2.0.1, the
> `.venv` has flygym + mujoco installed). Wave 1 of the Cellular Gaits page redesign;
> independent, no upstream dependencies. This BLOCKS prompt B (the in-browser MuJoCo-WASM
> substrate). Keep it small and verified. Use a todo list. Work on a branch
> `feat/web-export-mjcf`; don't merge.

## Why

The redesigned portfolio page will run the **real fly model in the real MuJoCo engine
compiled to WebAssembly**, in the browser. DeepMind's official MuJoCo WASM bindings load a
standard MJCF model + its asset files. FlyGym builds the fly from MJCF under
`flygym/data/mjcf/`. Your job: produce a **self-contained, validated MJCF bundle** of the
walking fly (the same body our controller drives) that a browser can load without any
Python.

## What to produce

1. **Locate the source model.** Find the fly MJCF that FlyGym's `Simulation` actually
   compiles for our setup (see `src/cellular_gaits/env.py` — `_build_default_world`, the
   `LEGS_ACTIVE_ONLY` actuated DoFs, position actuators `kp=50`, ctrlrange `(-3.14,
   3.14)`, flat ground). Inspect `flygym/data/mjcf/` and the FlyGym source to identify the
   exact XML(s) and referenced assets (meshes/textures).

2. **Emit a flattened, self-contained bundle.** Build the model the way `env.py` does, get
   the compiled `mujoco.MjModel`, and export a standalone MJCF with all assets resolved.
   Prefer the route that yields a single loadable model: e.g. compile via FlyGym, then
   `mujoco.mj_saveLastXML(...)` (or save the `MjSpec`/XML and copy every referenced mesh
   and texture next to it). The result must load with vanilla
   `mujoco.MjModel.from_xml_path(...)` in a **fresh** Python process with **no FlyGym
   imported**. Include a flat-ground plane so the fly can stand/walk.

3. **Drop anything WASM can't run, and document it.** If the model uses FlyGym-specific
   plugins/sensors/adhesion that the stock engine or the WASM build won't support, produce
   a walking-capable variant with those removed, and write down exactly what you removed
   and why. The non-negotiable requirement: the **42 leg position actuators must be
   present and in a known order** (7 DoFs × 6 legs, matching `env.py`'s
   `set_actuator_inputs` order), because the JS controller indexes them.

4. **Record the actuator + joint contract.** Emit a small `manifest.json` alongside the
   model: ordered list of the 42 actuator names, their joint names, ctrlrange, `kp`, the
   body name used for the thorax, control/physics rates (250 Hz control, 40 physics steps
   @ 10 kHz), and any simplifications. The browser side depends on this contract.

## Verify (and report the numbers)

- A fresh-process `mujoco.MjModel.from_xml_path` load succeeds with **zero** missing-asset
  or compile errors (no FlyGym in the interpreter).
- `model.nu == 42`; print the actuator names in order and confirm they match `env.py`.
- Step the model ~100 ticks in plain MuJoCo applying a constant mid-range ctrl; confirm it
  doesn't NaN/explode and the thorax body exists and reports a position.
- Report the bundle's total size (meshes can be large — note it; B may need to decimate).

## Output & handoff

- Write the bundle to `outputs/web_export/` in this repo:
  `fly.xml`, the asset files it references, and `manifest.json`.
- **Cross-repo note:** these files ultimately belong at
  `portfolio/public/cellular-gaits/model/`. You do NOT have access to the portfolio repo
  from here. Leave the bundle in `outputs/web_export/` and end your report with the exact
  copy command the user should run (or that prompt B will run with `/add-dir`).
- Update the status of row **A** to `done` in `portfolio/docs/cellular-gaits/build-plan.md`
  if (and only if) you've been given access to that repo; otherwise note it for the user.

## Deliverable

Branch `feat/web-export-mjcf`: the validated `outputs/web_export/` bundle + a SHORT report
(load-test numbers, actuator order, bundle size, anything removed, the copy command).
Then stop.
