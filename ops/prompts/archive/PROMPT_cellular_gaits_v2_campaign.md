# Claude Code prompt — Cellular Gaits v2: a staged experiment campaign

> Run from the `cellular-gaits` repo root (uv project, Python 3.12, FlyGym 2.0.1).
> This is a 3-stage campaign. **Execute Stage 1 fully, then STOP and report — do
> not start Stage 2 until I confirm.** Stages 2–3 are specced so you have the
> whole arc, but Stage 1's findings may change how we set them up. Use a todo
> list. Keep v1 intact; each stage gets its own branch. Don't merge.

## What exists (v1)
A neural cellular automaton drives a FlyGym fly. The NCA is an 8×8 grid, 4
channels/cell, one shared 660-param rule (`src/cellular_gaits/nca.py`: 3×3 conv
4→16, tanh, 1×1 conv 16→4, output clamped to [-1,1]). Channel 0 of a 7×6 motor
sub-grid → 42 leg actuators (`env.py`). Optimized with CMA-ES (`evolve.py`)
against forward walking distance minus a stability penalty (`env.py`). Best v1
walks ~86.6 mm / 3 s. Render with `render.py` (mp4 + CA-state JSON).

**Change ONE thing per stage so results are attributable. After each stage:
commit to its branch, render artifacts to `outputs/`, and write a SHORT report
(metrics + 3–5 sentence interpretation).**

---

## Stage 1 — Criticality sweep  (branch `feat/cg-v2-criticality`) — FAST, do this now
Add a scalar `gain` in `nca.py` that multiplies the pre-activation (the input to
`tanh`); default `1.0` = current behavior. Take the v1 best controller and sweep
gain over `{0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}`. For each gain, run a
rollout and record:
- **a chaos/criticality metric on the grid:** (a) mean per-tick state-change
  magnitude, and (b) a poor-man's Lyapunov — run two rollouts whose grids start
  ε apart and measure how fast they diverge over ticks.
- **gait quality:** forward distance + stability (the existing fitness).

Produce: a plot (gain → chaos metric, gain → fitness) in `outputs/`; 3 short
rendered clips at low / critical / high gain showing the grid orderly →
edge-of-chaos → chaotic; and a writeup on **where the best gaits sit relative to
the order→chaos transition.** This is analysis on the existing controller — no
big re-evolution needed (minutes).

**Then STOP and report. Wait for my go-ahead before Stage 2.**

---

## Stage 2 — Close the sensory loop  (branch `feat/cg-v2-closed-loop`) — ~one evolution run
v1 is open-loop (the grid drives the legs but never feels them). Modify `env.py`
+ `nca.py` so the grid **receives proprioceptive feedback** each control step:
read the fly's current joint angles (and foot-contact booleans if available) and
write them into the NCA — as extra input channels or into dedicated "sensor
cells" of the grid (document the choice). Everything else identical (same
fitness, CMA-ES, gain=1). Re-evolve. Compare **open-loop (v1) vs closed-loop**:
- walking distance + stability;
- a **robustness test the open loop can't pass:** perturb mid-rollout (a lateral
  shove, or a small step/uneven ground) and measure recovery — closed-loop
  should adapt, open-loop shouldn't.

Produce: closed-loop best controller, rendered video, the open-vs-closed metrics
+ the perturbation result + writeup.

---

## Stage 3 — A gallery of gaits via MAP-Elites  (branch `feat/cg-v2-gallery`) — HEAVY
On the better architecture from Stage 2, replace CMA-ES with **MAP-Elites**
(a QD library is fine). Define **2 behavior descriptors** — e.g. average speed,
and a gait-pattern descriptor (per-leg duty factor / stance fraction, or
front-vs-rear leg usage); document the choice. Evolve an **archive** of diverse,
high-performing controllers.

Produce: the archive, the descriptor-map visualization (2D grid of elites colored
by fitness) in `outputs/`, rendered clips of **4–6 distinct gaits** from
different archive cells (a fast one, a slow one, a weird one), + writeup. MAP-
Elites needs many more evaluations than CMA-ES — make it **checkpointed/
resumable** like `evolve.py`, cap it sensibly, and expect it to run long.

---

## Portfolio artifacts
Each stage should also emit web-embeddable assets in v1's spirit (mp4 + a stable
JSON of the relevant data) so the `/projects/cellular-gaits` page can later grow
a "v2" section — the criticality sweep, the open-vs-closed comparison, the gait
gallery. **Don't build the page; just emit the assets and note their paths.**

## Deliverable
Stage 1 only for now: branch `feat/cg-v2-criticality`, the sweep plot + 3 clips +
report. Then stop. Don't merge anything.
