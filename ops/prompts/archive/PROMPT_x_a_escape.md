# Claude Code — X-A · Escape: looming detector → fast directed flee

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create (`.worktrees/`, `AGENT_SAFETY.md`,
> `setup-*.sh`, `PROMPT_*.md`, `scratch/`, checkpoints/outputs you didn't make are off-limits).
> No `rm -rf` of anything you didn't create; no broad/glob deletes; scratch in one dir you make
> (`scratch/xa/`). Keep v1 + closed-loop + chemo code intact. **Commit your work on the branch
> before finishing** ("don't merge" ≠ "don't commit"). When unsure, leave it and note it.
> (Full: `AGENT_SAFETY.md`)

> **Run from the `cellular-gaits` repo root.** Escape campaign, wave 1 — the compute job. Runs
> in parallel with X-B (portfolio scaffold). Base off the chemotaxis branch so the 8→16 sensor
> plumbing exists: `git checkout feat/ch-chemotaxis` → `git checkout -b feat/x-escape`. Use a
> todo list.

## Goal

A **looming threat triggers a fast, directed escape** — the fly turns *away* and flees. The
headline is **emergent directed escape**: which way it bolts falls out of a left-vs-right
looming asymmetry, not a hard-coded turn. This is the connectome-aligned behavior: our
hand-built looming front-end stands in for the real **LC4/LPLC2 → DNp01 (Giant Fiber)**
circuit, which is the endgame swap (don't build the real circuit here — just leave the seam
clean).

## 1. Looming sensor + threat (`env.py`, `nca.py`)

- Add an approaching **threat**: a virtual object that, at a random time mid-episode, starts
  approaching the fly from a random azimuth at a set speed. Compute a **bilateral looming
  signal** at the two eyes from the threat's angular size and expansion rate (size ≈ LPLC2,
  expansion-rate ≈ LC4) projected by azimuth → `loom_L`, `loom_R`. Feed **two bilateral loom
  input channels** (the L/R asymmetry is what makes the escape *directed*). Document the exact
  signal in the exported `sensors` spec.
- **Warm-start** from the walking+proprioception controller (`closed_loop_controller.json`);
  the new loom channels zero-init → an 8→16 conv1 (state 4 + proprio 2 + loom 2). Loom-zeroed
  == closed-loop walking dynamics (A/B exact). Document the channel layout.

## 2. Episode + fitness (force emergent, directed escape)

Short episodes (~1–1.5 s): the fly walks, the threat looms, measure the response. Fitness =
**flee away** (net displacement increasing distance from the threat / heading turned away from
the threat azimuth) + **reaction speed** (turn begins soon after loom onset) + **survival**
(not within a hit-radius when the threat arrives). **Average over threat azimuths (left /
right / front)** so a fixed turn can't win — only genuine `loom_L`-vs-`loom_R` steering escapes
all of them. Parallelize the CMA-ES population (ProcessPool, per-worker env). Warm-start, ~50
gens, checkpoint/resumable.

## 3. Validation-first (short run, then STOP and report)

Build everything, then a short run to: confirm warm-started gen 0 still walks; calibrate the
threat speed/size so an **untrained** walker gets hit (so there's something to learn) but
escape is achievable; and check an early signal that it turns *away* on both sides. Report the
calibration + early trend; I green-light the full run.

## 4. Exports → `outputs/web_data_x/`

- `escape_controller.json` — weights + full `sensors` spec (loom geometry, eye projection,
  channel layout, normalization).
- `flee_left.mp4`, `flee_right.mp4` — threat from opposite sides, same controller, showing
  **emergent opposite escape turns**.
- `trajectories.json` — fly + threat paths over a few episodes (top-down viz).
- `escape_metrics.json` — escape success rate, reaction latency, final distance, by azimuth.
- `REPORT_x_a.md` — sensor design, calibration, parallel speedup, results + interpretation,
  ending with the cross-repo copy command (assets → `portfolio/public/cellular-gaits/data-x/`;
  no portfolio access from here).

**Honesty (put in the report + sensor spec):** the looming front-end is **hand-built**, a
stand-in for the real LC4/LPLC2→DNp01 wiring (the connectome endgame, not done here); episodes
are short; note the azimuth generalization scope. Don't compare the fitness scalar across
behaviors.

## Verify
Loom-zeroed == closed-loop walking (A/B); parallel == sequential within noise; the two flee
clips turn opposite ways; metrics finite. Commit on `feat/x-escape`. Then stop. Don't merge.
