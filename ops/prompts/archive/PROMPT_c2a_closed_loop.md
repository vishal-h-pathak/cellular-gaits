# Claude Code — C2-A · Close the sensory loop + perturbation robustness

> **Run from the `cellular-gaits` repo root** (uv project, Python 3.12, FlyGym 2.0.1).
> Campaign 2, wave 1. The compute/critical-path job. Runs in parallel with C2-B (portfolio
> scaffold). Branch `feat/c2-closed-loop`; don't merge. Keep v1 intact. Use a todo list.

## Goal

Give the NCA controller **proprioceptive feedback** (it currently walks blind), then prove
the closed loop matters with a **perturbation-recovery** test the open-loop controller fails.
This is both the shared foundation for all future behaviors and the first behavior itself.

## 1. Close the loop (`nca.py` + `env.py`)

Feed the fly's own state back into the grid each control step. Read the current **joint
angles** (42, normalize to ~[-1,1] by ctrlrange) and **foot-contact booleans** (6 legs) from
the sim, and write them into the NCA as input. **Document the wiring choice**, pick whichever
is cleaner:

- **(a) extra input channels:** widen conv1 from 4→(4+S) input channels; broadcast/tile the
  sensor vector into S channels; or
- **(b) sensor cells:** overwrite a dedicated band of grid cells with the sensor values each
  tick before the rule runs.

Keep it small (the param count should stay in the low thousands at most). Default behavior
with sensors zeroed must reproduce the open-loop dynamics, so you can A/B cleanly.

## 2. Perturbation in the environment (`env.py`)

Add an optional perturbation to a rollout: at a random time in a mid-rollout window, apply a
**lateral impulse** to the thorax (`xfrc_applied`, configurable magnitude/direction). Also
support an optional **uneven-ground** variant (small random height field) behind a flag.
Make the perturbation seedable so open- and closed-loop runs see *identical* shoves.

## 3. Re-evolve the closed-loop controller — and PARALLELIZE

Re-evolve with CMA-ES, **but evaluate the population in parallel** (this is the key change
from v1, which ran sequentially): use `concurrent.futures.ProcessPoolExecutor` (build a fresh
`FlyEnv` inside each worker — MuJoCo handles aren't fork-safe). Target ~all cores; report the
speedup. Fitness = forward distance **+ heading retention after the shove − fall penalty**,
averaged over a few perturbation seeds for robustness. Pop 32, σ₀ 0.3, ~50 gens; warm-start
from the v1 walking weights for the shared dims if convenient. Checkpoint/resumable like
`evolve.py`.

## 4. Open vs closed comparison (the headline)

Run the **v1 open-loop best** and the **new closed-loop best** on the *same* perturbation
seeds. Record, per controller: did it stay upright, did it return to heading, distance after
the shove, recovery time. The expected story: open-loop gets knocked off course and can't
correct; closed-loop catches itself and resumes.

## Exports (for the page — C2-C will consume these)

Write to `outputs/web_data_c2/`:
- `closed_loop_controller.json` — weights **+ a `sensors` spec** (what each added channel is,
  its normalization, and where it enters the grid).
- `perturbation_openloop.mp4`, `perturbation_closedloop.mp4` — same shove, side-by-side-able.
- `robustness_metrics.json` — per controller: upright%, heading error, post-shove distance,
  recovery time, across the seed set.
- a short `REPORT_c2a.md`: the wiring choice, the parallel speedup, and the open-vs-closed
  numbers + 3–5 sentence interpretation.

Cross-repo note: these belong at `portfolio/public/cellular-gaits/data-c2/` — you don't have
portfolio access from here, so leave them in `outputs/web_data_c2/` and end the report with
the exact copy command.

## Verify

Sensors-zeroed closed-loop == open-loop dynamics (A/B integrity); parallel eval matches
sequential fitness within noise; clips play (< ~3 MB); metrics finite. Then stop and report.
**Don't merge.**
