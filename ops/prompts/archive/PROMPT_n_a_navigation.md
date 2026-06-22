# Claude Code — N-A · Obstacle navigation (compute): feelers + arena + seek-vs-avoid fitness

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + all prior behaviors intact
> (`nca.py`, `env.py`, `evolve*.py`, existing checkpoints/`outputs/`, `web_data*`). No `rm -rf`
> of anything you didn't create; no broad/glob deletes; scratch in ONE dir you make
> (`scratch/na/`), clean up only that exact path. Append to shared files; don't rewrite.
> `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh` are off-limits. **Commit your work on the
> branch before finishing** (don't merge to main). When unsure, leave it and note it.
> (Full: `AGENT_SAFETY.md`)

> **Run from the `cellular-gaits` repo root.** Navigation campaign, wave 1 — the compute half,
> parallel with N-B (portfolio scaffold). Branch off the escape line so the sensor plumbing
> pattern is fresh: `git checkout feat/x-escape-live 2>/dev/null || git checkout main` →
> `git checkout -b feat/n-navigation`. Use a todo list. **Read first:**
> `../portfolio/docs/cellular-gaits/research-roadmap.md` (the behaviors ledger + bigger arc),
> `PARTNER_BRIEF.md`, and the existing **chemotaxis + escape** code you're extending:
> `src/cellular_gaits/nca.py` (the `chemo`/`loom` modes, `build_chemo_sensor_map`,
> `warm_start_from_closed_loop`), `src/cellular_gaits/env.py` (`OdorField`, `read_chemo`,
> `antenna_positions`, `Threat`/`read_loom`, `_add_pebbles`/`_build_default_world`),
> `src/cellular_gaits/evolve_chemotaxis.py` and `src/cellular_gaits/evolve_escape.py` (config +
> fitness + conditions + parallel CMA-ES), and `scripts/run_evolution_escape.py` (the CLI +
> calibrate/benchmark gates you'll mirror).

## The design (read this before coding — it's the point)

Navigation is **the synthesis behavior**: the fly must **seek a goal** *and* **avoid obstacles**
in its way. We get the seeking **for free** by warm-starting from the **trained chemotaxis
controller** — it already homes on a bilateral odor beacon. We **add two new bilateral "feeler"
channels** (short-range obstacle proximity, left field vs right field) and re-evolve so it learns
to **detour around obstacles while still homing**. The compositional story is the payoff:
*we stacked obstacle avoidance on top of the forager.*

This keeps the project's through-line intact — **every behavior's directional response emerges
from a bilateral asymmetry**: chemotaxis turns *toward* the stronger odor side, escape bolts
*away* from the stronger loom side, and navigation must **arbitrate** a turn-toward-goal (odor
L−R) against a turn-away-from-wall (feeler L−R) when they conflict. Nothing hard-codes the
detour direction; it must fall out of the feeler asymmetry under selection.

### Architecture — a new 10-input `nav` mode

Current NCA: `chemo`/`loom` are 8-input (4 state + 2 proprio + 2 cue). Navigation needs **both**
the odor goal-beacon **and** the feelers, so add a new mode:

- `NCA(nav=True)` → **10 input channels**: 4 state + 2 proprio + **2 odor (goal beacon)** + **2
  feeler (obstacle proximity)**. conv1 = `Conv2d(10→16, 3×3, zero-pad)` → `tanh(gain·)` → conv2
  `Conv2d(16→4, 1×1)` → `clamp[-1,1]`. State stays 4 channels. n_params ≈ 10·16·9+16 + 16·4+4 =
  **1524**.
- Keep `nav` mutually exclusive with `chemo`/`loom` (extend the existing guard). Default `NCA()`
  unchanged; chemo and loom paths **untouched**.
- `build_nav_sensor_map(joint_angles, foot_contacts, odor_left, odor_right, feeler_left,
  feeler_right)` → `(1, 6, 8, 8)` extra-channel tensor: ch0–1 proprio (identical to
  `build_sensor_map`), ch2–3 odor (identical layout to `build_chemo_sensor_map` — left half /
  right half of the motor block), ch4–5 feeler (same bilateral topographic layout: feeler_left
  over the left half, feeler_right over the right half).
- **`warm_start_from_chemo(chemo_vec)`**: load the trained 8-input chemotaxis params into the
  10-input nav model — copy conv1 input channels 0–7 (state+proprio+odor), **zero-init the two
  new feeler input channels (8–9)**, copy conv1.bias and all of conv2. So **feelers-zeroed (or
  feeler-weights-zero) reproduces the chemotaxis dynamics exactly** — the A/B integrity check.
  Source params: `outputs/web_data_ch/chemotaxis_controller.json` (weights) or the chemo
  checkpoint if you find a flat-params one; validate the param count against a fresh
  `NCA(chemo=True).n_params()`.

### The feeler front-end (hand-built, like loom — keep the seam clean)

Add to `env.py`, mirroring the `Threat`/`read_loom` and `OdorField`/`read_chemo` patterns:

- An **`ObstacleField`** (or a list of obstacle disks): each obstacle a vertical cylinder/disk
  of radius `obstacle_radius` at a world xy. `set_obstacles([...])`. They must be **physical**
  in MuJoCo (the body actually collides) AND sensed by the feelers — add them to the world the
  same way `_add_pebbles`/`_build_default_world` add geometry; if adding collidable geoms mid-
  episode is awkward, build them into the world at reset from a layout passed to `FlyEnv`.
- **`read_feelers() -> (feeler_left, feeler_right)`**: bilateral short-range rangefinder. From
  the fly head/thorax, over the **left** and **right** forward visual field, return the
  proximity of the nearest obstacle: `proximity = clip(1 - d_nearest / feeler_range, 0, 1)` (so
  a close wall → near 1, nothing in range → 0). Split by **bearing** like the loom eye-split
  (an obstacle dead-ahead contributes to both; an offset obstacle weights the near side), so the
  **left−right feeler asymmetry carries which way to dodge**. Returns (0, 0) when no obstacles
  are set, so a nav policy run with no obstacles == the pure forager. Put the geometry constants
  (`feeler_range`, `obstacle_radius`, field half-angle) near the loom/odor constants with the
  same documented-constant style.

### Fitness — compose chemo's approach reward with a collision penalty (`evolve_navigation.py`)

New `evolve_navigation.py` mirroring `evolve_escape.py`/`evolve_chemotaxis.py` (a `NavConfig`
dataclass, `conditions(cfg)`, a `nav_fitness(...)`, parallel CMA-ES via the same
ProcessPoolExecutor path). Per condition:

```
F_condition = approach        (= d_start - min_dist to goal, the chemo "closest" shaping)
            + reach_bonus * [reached goal]
            - w_collide * collision_count            (or time-in-contact)
            - time_penalty * (steps_to_reach / N)    (optional, reuse chemo's)
fitness     = mean over conditions
```

**Conditions = obstacle layouts, designed so a fixed swerve can't win** (the escape/chemo
generalization lesson): vary the goal azimuth AND the obstacle placement so the detour direction
must come from the feelers, not a bias — e.g. {goal ahead, obstacle blocking left-of-path},
{goal ahead, obstacle blocking right-of-path}, {goal left, obstacle centered}, etc. **Calibrate
the obstacle so the straight homing path actually collides** — this is the navigation analog of
escape's target-leading: if the forager can reach the goal without touching the obstacle, there's
nothing to learn. Place obstacles ON the chemo-warm-start fly's natural path (you can read that
path from a no-avoidance rollout) so they're genuine impediments.

## VALIDATION-FIRST — build everything, then STOP and report (do NOT run the full evolution)

Add `--calibrate` / `--benchmark` / `--benchmark-only` gates to `scripts/run_evolution_navigation.py`
exactly like `run_evolution_escape.py`. Then run the **short calibration** and **stop for my
go-ahead**. The calibration report must confirm:

1. **Warm-start still walks + homes:** `NCA(nav=True)` warm-started from chemo, with feelers
   zeroed, reproduces the chemotaxis dynamics **bit-exact** (A/B: max|Δ| = 0 vs the 8-input
   chemo forward pass on shared weights), and on a **no-obstacle** layout it still reaches the
   goal.
2. **Obstacles are genuine impediments (the baseline collides):** on the chosen layouts, the
   **untrained** warm-start forager (feelers wired but avoidance not yet learned) **collides /
   fails to reach** — the navigation analog of escape's "untrained hit 3/3." Report collision
   rate + reach rate for the baseline so we can see there's headroom.
3. **Parallel == sequential:** max|Δfit| = 0 across workers; report the speedup.
4. **Both-sides detour is achievable:** a short search shows the left-blocking layout can yield a
   **right** detour and the right-blocking layout a **left** detour (emergent from feeler L−R) —
   if a fixed bias wins, re-weight (raise `w_collide` / the directional term) the way escape
   raised its directional weights above survival, and document it.

Mirror escape's compute envelope: parallel CMA-ES, but n_params is ~1524 (bigger than escape's
1236), so plan a slightly larger population (suggest `--pop 48`) and more generations
(`--gens ~60–80`) for the *full* run — propose the numbers in your report; **don't launch it.**
Write calibration artifacts to `scratch/na/` and a short `REPORT_n_a_calibration.md` at repo
root. End with the exact full-run command you propose and: *"Confirm and I'll launch the full
run."*

## Honesty caveats — document these in the report and in the eventual export meta

Navigation is **the most robot-demo-compelling but least biologically grounded** behavior, and
the page must not overclaim:

1. The **feelers are a hand-built rangefinder abstraction**. Real *Drosophila* avoid obstacles
   via **vision / optic flow / visual looming**, NOT a LIDAR-like distance sensor. Unlike escape
   (which maps onto the real LC4/LPLC2→DNp01 circuit), **navigation has no clean real-circuit
   seam** — it's a robotics-flavored capability demo, not a connectome bridge. Say so plainly.
2. The **"goal" is the chemotaxis odor beacon reused** as a homing target — this is **reactive
   local avoidance + gradient homing**, NOT global path planning or spatial memory. It can get
   **trapped in concave/dead-end obstacle configurations** (a local minimum). Flag this as a real
   limitation, not a bug to hide.
3. The nav **fitness scalar is not comparable** across behaviors (task-specific shaping).
4. Keep `chemo`/`loom`/closed-loop/v1 paths **bit-exact unchanged**; nav is purely additive.

## Definition of done (this session)

- `NCA(nav=True)` + `build_nav_sensor_map` + `warm_start_from_chemo`; `ObstacleField`/obstacles +
  `read_feelers` in `env.py`; `evolve_navigation.py` + `scripts/run_evolution_navigation.py`
  (+ a `test_navigation.py` smoke test mirroring `test_escape.py`).
- All existing tests still pass (`test_nca.py`, `test_chemotaxis.py`, `test_escape.py`, …);
  prior behaviors untouched and bit-exact.
- Calibration run done, the four gate items reported, full-run command proposed — **stopped,
  not launched.**
- **Committed on `feat/n-navigation`** (not merged). Report: files added/changed, the four gate
  results, the proposed full-run numbers, and any re-weighting you had to do.
