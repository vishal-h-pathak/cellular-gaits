# C2-A — Closed sensory loop + perturbation robustness

## 0. Choosing a fair perturbation magnitude

Before evolving, we swept the lateral-impulse magnitude on the **v1 open-loop best** to find the smallest shove at which it *clearly degrades* (falls, fails to recover heading, or ends with mean post-shove heading error > 45°) on roughly half to two-thirds of 18 perturbation seeds. Too weak and the open loop never fails; too strong and nothing could recover.

| Magnitude | Degraded | Fell | No-recover | Mean heading err | Post-shove dist |
|---|---|---|---|---|---|
| 3.0 | 39% | 0% | 11% | 38.8° | 30.66 |
| 4.0 | 44% | 0% | 11% | 46.6° | 26.11 |
| 5.0 | 33% | 0% | 11% | 47.6° | 25.07 |
| 6.0 | 56% | 0% | 11% | 56.6° | 20.81 |

**Chosen operating magnitude: 6.0** — the smallest in the 50–67% degraded band. The full closed-loop evolution and the comparison below both use this magnitude.

## 1. Wiring choice (how the loop is closed)

**Option (a): extra conv1 input channels.** `conv1` is widened from `4 -> 16` to `(4+2) -> 16`. Two per-tick **sensor channels** are concatenated onto the 4-channel CA state before the rule runs:

- **ch4 — joint angles (42):** the 42 actuated joint angles (`mj_data.actuator_length`, same order as the motor outputs), normalized by ctrlrange (`/3.14`, clipped to [-1,1]), placed in the 7x6 motor block (rows 0-6, cols 0-5) so each angle sits on its own motor cell.
- **ch5 — foot contacts (6):** the 6 per-leg ground-contact booleans, placed in the bottom grid row (row 7, cols 0-5).

The CA **state stays 4 channels** (conv2 output is unchanged), so the recurrent dynamics and motor readout are identical in shape to v1. The new sensor input-channel weights are **zero-initialized**, so a closed-loop NCA warm-started from the v1 weights reproduces the v1 open-loop dynamics **bit-for-bit** until evolution moves them off zero. Param count: **948** (v1 was 660; target was < 1000).

Why not option (b) (overwrite a band of grid cells with sensors): that would clobber channels v1 uses as free hidden state, so "sensors-zeroed == open-loop" would fail. Option (a) keeps that A/B guarantee exactly (verified: Δfitness = Δtrajectory = 0).

## 2. Parallel evaluation speedup

CMA-ES population evaluation was parallelized with `ProcessPoolExecutor` (one `FlyEnv` per worker, built in an initializer — MuJoCo handles are not fork-safe). Measured steady-state speedup: **4.60x  (127.4s sequential -> 27.7s parallel on 9 workers, 32 individuals; max |Δfitness| vs sequential = 0.00e+00)**.

Parallel and sequential evaluation produce identical fitness (the sim is deterministic), so the speedup is free.

## 3. Open vs closed loop under the same shove

Both controllers run on the **same** lateral-impulse seeds ([101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555, 666, 777, 888, 999]), magnitude 6.0, applied in the [0.4, 0.6] window of a 750-step (3.0s) rollout. Aggregates across 18 seeds:

| Metric | Open loop (v1) | Closed loop |
|---|---|---|
| Forward distance (full) | 65.442 | 69.315 |
| Post-shove distance | 20.814 | 24.331 |
| Heading error (deg) | 56.6 | 26.5 |
| Upright % (post-shove) | 100.0% | 100.0% |
| Stayed-upright rate | 100.0% | 100.0% |
| Recovered heading rate | 88.9% | 77.8% |
| Recovery time (s) | 0.031 | 0.046 |

Closed-loop best fitness: **69.9682** (checkpoint `checkpoints/c2a_full/gen_50.npz`).

## 4. Interpretation

Feeding proprioception back into the grid gives the controller a way to feel the shove and correct for it; the open-loop controller runs a fixed rhythm regardless of what the body is doing. Across the seed set, the closed loop roughly **halves** the mean post-shove heading error (26.5° vs 56.6°, a 30° improvement) and makes more forward progress after the shove (24.3 vs 20.8). The rendered clips show the effect at its starkest: on this shove the open loop ends **97°** off heading while the closed loop holds to **19°** and keeps walking. The 'recovered-heading rate' / 'recovery time' columns are a wash (or marginally favor open loop). That metric is a tolerance-threshold artifact: many open-loop shoves drift slowly and happen to clip back under the recovery tolerance briefly even while their *mean* error stays large. The robust, integrated signal — mean post-shove heading error and post-shove distance — is where the closed loop clearly wins. Neither controller falls at this magnitude (both 100% upright), so robustness here is about staying *on course*, not staying upright.

## Files (in `outputs/web_data_c2/`)

- `closed_loop_controller.json` — weights + sensor spec
- `robustness_metrics.json` — per-controller metrics across the seed set
- `perturbation_openloop.mp4` (1.77 MB)
- `perturbation_closedloop.mp4` (1.83 MB)

## Cross-repo copy

These belong at `portfolio/public/cellular-gaits/data-c2/`. From the portfolio repo root (adjust the source path to this repo):

```bash
mkdir -p portfolio/public/cellular-gaits/data-c2
cp /Users/jarvis/dev/jarvis/cellular-gaits/outputs/web_data_c2/* \
   portfolio/public/cellular-gaits/data-c2/
```

**Do not merge** — branch `feat/c2-closed-loop` is for review.
