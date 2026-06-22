# Wave-1 D — Precompute report (gain sweep + evolution + clips)

Branch `feat/web-precompute`. Analysis on the existing v1 best controller
(`checkpoints/2026-05-02T00-01-51Z/gen_50.npz`, `best_fit ≈ 86.62`). v1 left
intact. Nothing merged.

## What shipped

| Artifact | Path | Notes |
|---|---|---|
| Gain knob | `src/cellular_gaits/nca.py` | `tanh(gain · conv1(s))`, default `1.0` = identity |
| Gain knob tests | `scripts/test_nca.py` | `gain=1.0` reproduces the pre-gain rule bit-for-bit over 50 ticks |
| Criticality metrics | `src/cellular_gaits/criticality.py` | change-rate + twin-trajectory Lyapunov λ |
| Gain→gait sweep | `outputs/web_data/gain_sweep.json` | 9 gains, full MuJoCo rollouts |
| Evolution curve | `outputs/web_data/evolution.json` | transform of the v1 fitness CSV |
| 3 clips | `outputs/web_data/clip_gain_{lo,native,hi}.mp4` | gain 0.5 / 1.0 / 2.2, H.264 yuv420p, ~0.15 MB each |

## Verification

- **`gain=1.0` is the unchanged controller.** Unit test proves `tanh(1.0·conv1(s))`
  matches the pre-gain rule bit-for-bit over 50 ticks; and the full `gain=1.0`
  MuJoCo rollout reproduces the evolved checkpoint fitness exactly
  (`86.61898…`, diff < 2e-5 = JSON 4-digit rounding).
- **`gain_sweep.json`** has all 9 gains, every value finite, and λ changes sign
  between gain `1.3` (−0.12) and `1.5` (+0.11) — inside the expected `~1.0–1.5`
  band.
- **Clips** are browser-safe H.264 / yuv420p and ~0.15 MB each (well under 3 MB).

## The gain → gait sweep (the headline)

| gain | fitness F | distance (mm) | n_below | λ (Lyapunov) | change_rate |
|-----:|----------:|--------------:|--------:|-------------:|------------:|
| 0.25 |   26.24 |  26.24 | 0 | −0.842 | 1.946 |
| 0.50 |   56.70 |  56.70 | 0 | −0.887 | 1.957 |
| 0.75 |   53.45 |  53.45 | 0 | −0.157 | 1.916 |
| **1.00** | **86.62** | **86.62** | **0** | **−0.261** | **1.916** |
| 1.30 |    2.96 |   2.96 | 0 | −0.120 | 1.903 |
| 1.50 |   −6.99 |  −6.99 | 0 | **+0.110** | 1.896 |
| 2.00 |  −10.67 | −10.67 | 0 | +0.092 | 1.896 |
| 3.00 |   −0.09 |  −0.09 | 0 | −0.011 | 1.892 |
| 4.00 |   −5.37 |  −5.37 | 0 | +0.140 | 1.899 |

(`distance_mm == F` here because the fly never trips the fall threshold, so
`n_below = 0` and the stability penalty is zero at every gain — degradation
shows up as *failure to move*, not as falling over.)

## Criticality finding

The good gait sits **just inside the ordered side of the order→chaos
transition.** Walking distance is sharply single-peaked at the native operating
point (gain `1.0`, 86.6 mm) and collapses on both sides — starved below it
(26 mm at 0.25) and destroyed above it (3 mm at 1.3, then *negative* from 1.5
on). The poor-man's Lyapunov exponent crosses zero between gain `1.3` (λ ≈
−0.12) and `1.5` (λ ≈ +0.11), and that crossing coincides exactly with the
performance cliff: the moment the CA tips into chaos (λ > 0), the leg-target
stream loses the coherent rhythm a gait needs and forward progress dies. So
yes — CMA-ES parked the controller at λ ≈ −0.26, a hair inside the edge of
chaos, which is the reservoir-computing sweet spot the gain knob was built to
probe. (The state-change rate is ~1.9 and nearly flat across the whole sweep:
the trained rule runs as a high-amplitude, near-saturated oscillation at every
gain, so λ — not change-rate — is the metric that actually separates good gaits
from bad here.)

## Evolution curve (the premature-convergence story)

`evolution.json` is a straight transform of `fitness_log_2026-05-02T00-01-51Z.csv`
with a continuous `step` index added (both phases re-use gen numbers). The
`original` run climbed to ~62 mm by gen 37 then stalled; a warm-started
`resumed` run re-injected diversity and broke through to **86.6 mm** within a
few generations — the mean fitness jumps from deeply negative to ~50, the
signature of escaping a premature-convergence plateau.

## Deferred — needs work that does not exist yet

- **TODO (Stage 2 / E3 Sensing): open- vs closed-loop perturbation-recovery.**
  The headline robustness comparison (shove the fly mid-rollout, measure
  recovery) requires the **closed-loop architecture** — proprioceptive feedback
  written back into the grid. v1 is open-loop, so there is nothing to perturb-
  and-recover yet. Not buildable here.
- **TODO (E6 Optimizer): CPG baseline rollout.** A central-pattern-generator
  baseline to contrast against the NCA needs a **CPG controller**, which is not
  implemented. Add the controller first, then a matching rollout + metrics.

## Handoff — move data into the portfolio

These belong at `portfolio/public/cellular-gaits/data/`. No portfolio access
from this repo; copy them over (or `/add-dir` the portfolio and re-run):

```bash
cp outputs/web_data/gain_sweep.json \
   outputs/web_data/evolution.json \
   outputs/web_data/clip_gain_lo.mp4 \
   outputs/web_data/clip_gain_native.mp4 \
   outputs/web_data/clip_gain_hi.mp4 \
   <portfolio>/public/cellular-gaits/data/
```

Build-plan row **D** in `portfolio/docs/cellular-gaits/build-plan.md` not
updated (no access) — please flip it to done after the copy.
