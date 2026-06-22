# Cellular Gaits / FlyGym — context handoff

Paste this into a fresh chat to get up to speed fast. It captures where the
`cellular-gaits` project stands, the questions I was working through, and the
direction we picked.

## The project, factually

`cellular-gaits` is a **Neural Cellular Automaton (NCA)** that drives a
simulated fruit fly to walk.

- **Grid:** 8×8, **4 channels** per cell (think 4 stacked 8×8 heatmaps, values
  blue↔red, that flow/pulse over time — not Conway's black/white bits).
- **Rule:** one shared 2-layer conv "MLP", ~**660 params** — `Conv2d 4→16`
  (3×3, zero-padded) → **tanh** → `1×1 conv 16→4`, output clamped to [-1, 1].
  Every cell updates from its 3×3 neighborhood using this same rule each tick.
- **Body:** FlyGym / MuJoCo biomechanical fly, **42 leg actuators**. Channel 0
  of a 7×6 motor sub-grid is read as joint targets in [-1,1], rescaled to
  (-π, π) rad.
- **Training:** **CMA-ES** (not backprop). Fitness = forward walking distance −
  0.05·(stability penalty). Pop 32, ~50 gens. Best individual walks **~86.6 mm
  in 3 s (~29 mm/s)** — real-Drosophila range.
- **Stack:** `flygym==2.0.1`, Python 3.12, uv project. Key files:
  `src/cellular_gaits/nca.py`, `env.py`, `evolve.py`, `render.py` (emits mp4 +
  CA-state JSON). It's NOT Conway's GoL — it's GoL's *skeleton* (grid, locality,
  one shared rule) with continuous multi-channel cells and a *learnable* rule.

It's also live on the portfolio at `/projects/cellular-gaits` (hero video + a
canvas that renders the CA state synced to the video).

## The three questions I was chewing on (and the answers we landed)

1. **"What does the GoL actually look like here?"** — Four little 8×8 heatmaps,
   coupled, flowing continuously. Each cell holds a 4-vector; each tick it
   updates from its 3×3 neighborhood via the shared rule. A 7×6 corner of
   channel 0 is wired to the legs.

2. **"Why CMA-ES instead of backprop?"** — Because the objective (distance
   walked in MuJoCo) is **non-differentiable** — you'd have to differentiate
   through the physics/contact solver, which has discontinuities. CMA-ES is a
   **black-box optimizer**: it only needs to *evaluate* (run sim → score), not
   differentiate. (Aside: differentiable physics like MuJoCo MJX/Brax *would*
   let you backprop — a possible future thread.)

3. **"Why tanh — is it normalization?"** — Partly, but the smaller half. tanh
   does two jobs:
   - **It's the nonlinearity** — the whole ballgame. Without it, two stacked
     linear layers collapse to one linear map and the CA can produce *no*
     emergent patterns/rhythms. (It's the smooth cousin of a spiking threshold —
     ties to my SNN background.)
   - **It bounds state to (-1, 1)** — stabilizes the recurrent dynamics so values
     don't blow up over many ticks; also convenient for the motor readout.
   - **The criticality hook:** the *gain* (how much you scale tanh's input) moves
     the system along order↔chaos. Low gain = linear/orderly; high gain =
     saturated/chaotic; in between = **edge of chaos** (reservoir-computing sweet
     spot). That one scalar is the criticality knob, and it's already sitting in
     the model.

## Direction we picked: a 3-stage experiment campaign

Goal: "extend the GoL as far as we can." Doing all three, but **sequenced** (one
variable at a time, so results are attributable), not simultaneously. Full spec
is in **`PROMPT_cellular_gaits_v2_campaign.md`** (run from the repo root).

- **Stage 1 — Criticality sweep** (cheap, FIRST): add a `gain` knob, sweep it on
  the v1 best controller, measure a chaos metric (state-change rate + a
  poor-man's Lyapunov) vs gait quality. See where the good gaits sit relative to
  the edge of chaos. CC runs **this stage only, then stops and reports** so we
  interpret before committing compute.
- **Stage 2 — Close the sensory loop** (the big architectural change): feed the
  fly's joint angles / foot contacts *back into the grid* (it's currently
  open-loop), re-evolve, and compare open vs closed loop — including a
  perturbation/recovery test the open loop can't pass.
- **Stage 3 — Gait gallery via MAP-Elites** (heavy, last): swap CMA-ES for
  MAP-Elites over 2 behavior descriptors (e.g. speed + duty factor) to evolve an
  *archive* of diverse gaits; render 4–6 distinct ones for the portfolio.

Each stage emits web-embeddable assets (mp4 + JSON) so the portfolio page can
grow a "v2" section.

## Workflow we agreed on

Iterate on the *interesting* part (concepts, experiment design, interpreting
results) **in chat**; hand the *menial* part (writing code, running CMA-ES ~35
min, rendering, portfolio plumbing) to **Claude Code**. Loop: design here → CC
executes → results back → interpret here → design next. CC never decides what's
interesting.

## Immediate next step

Run `PROMPT_cellular_gaits_v2_campaign.md` from the `cellular-gaits` repo root.
It executes Stage 1 and stops. Bring the Stage 1 report back to chat to decide
Stage 2 setup.

## Bigger arc (for later)

- The fly **connectome** was only fully mapped recently (adult Drosophila,
  ~139,255 neurons, FlyWire, Oct 2024). Connectome = *structure*; behavior =
  *dynamics*; structure under-determines behavior at fly scale.
- Eventual synthesis thread: pull the real **VNC leg circuit from FlyWire** and
  use it (instead of the generic grid) to drive FlyGym — "connectome as a
  reservoir poised at criticality, read out to the legs."
- This whole line connects to the Eon Systems (connectomics / embodied-sim)
  interest.
