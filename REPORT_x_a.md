# X-A · Escape: looming detector → fast directed flee

**Branch:** `feat/x-escape` · **Run:** `xa-full` (pop 32, 50 gens) · **best fitness:** 13.23

A looming threat triggers a **fast, directed escape**: the fly turns *away* and
flees, and **which way it bolts is emergent** — it falls out of a left-vs-right
looming asymmetry (`loom_L` vs `loom_R`), not a hard-coded turn. The hand-built
looming front-end stands in for the real **LC4/LPLC2 → DNp01 (Giant Fiber)**
circuit; that connectome swap is the endgame and is *not* built here (the two
bilateral loom channels are the clean seam for it).

---

## 1. Sensor design

The controller is the 8-input loom NCA, `NCA(loom=True)`: **4 state + 2 proprio +
2 loom** input channels into `conv1` (3×3, zero-pad) → `tanh(gain·)` → `conv2`
(1×1, 16→4) → `clamp[-1,1]`. The CA state stays 4 channels; proprioception and
loom enter only at `conv1`'s input. Warm-started from the C2-A closed-loop
walking controller with the loom weights zero-initialized, so **loom-zeroed ==
closed-loop walking, bit-for-bit** (A/B) until evolution moves the loom weights.

**Channel layout** (topographic, over the 7×6 motor block):

| conv1 input ch | name | placement | source |
|---|---|---|---|
| 4 | `joint_angles` (42) | rows 0–6, cols 0–5 | `actuator_length`/3.14, clipped |
| 5 | `foot_contacts` (6) | row 7, cols 0–5 | per-leg ground contact |
| 6 | `loom_left` | left half (cols 0–2) | looming projected on the LEFT eye |
| 7 | `loom_right` | right half (cols 3–5) | looming projected on the RIGHT eye |

**Looming signal.** From the threat's angular size and expansion rate:

```
theta     = 2 * atan2(R, d)                       # angular size (~ LPLC2)
rate      = max(0, dtheta/dt)                      # expansion only (~ LC4)
m         = clip( size_gain*theta/pi
                + exp_gain*min(1, rate/exp_ref), 0, 1 )   # magnitude in [0,1]
```

with `size_gain=1, exp_gain=1, exp_ref=6, R=2`. The magnitude is split between
the two eyes by the threat's body-frame bearing `phi` (CCW positive = the fly's
left):

```
loom_L = m * 0.5 * (1 + sin phi)
loom_R = m * 0.5 * (1 - sin phi)
```

A frontal threat (`phi=0`) excites both eyes equally; a threat directly to one
side excites only that eye. **The L−R difference is the directional cue** — it is
what makes the escape *directed*. The `[0,1]` loom is logged as such, then
multiplied by **`loom_input_gain = 8`** before entering `conv1` (see §2).

---

## 2. Calibration (validation-first)

Short ~1.2 s episodes (300 control steps @ 250 Hz). The threat appears at a
seeded time mid-episode and flies a straight collision course at fixed speed,
aimed not at the fly's onset position but at where a constant-velocity fly would
be when it arrives (**target leading**). Without leading, a forward-walking fly
dodges side threats for free; with leading the threat is a genuine collision
course from every azimuth, and only an actual escape maneuver survives.

Two calibration findings drove the final config:

1. **Threat magnitude.** Swept (speed × lead). At **speed 45 / lead 15** the
   untrained (warm-start) walker is hit on **all three azimuths** (mean closest
   approach 1.6, all inside the 3.0 hit-radius) — so there is genuinely something
   to learn — while the bilateral loom cue stays strong and correctly signed.

2. **`loom_input_gain` (the key finding).** The warm-start gait is *bang-bang*:
   every motor cell is pinned at the ±1 clamp. An unamplified `[0,1]` loom
   therefore cannot move a single motor cell until the loom weights grow to ~2 —
   a **flat fitness plateau** CMA-ES cannot climb from zero. Amplifying the cue
   (`×8`) gives small loom weights immediate leverage, so steering is learnable
   in the generation budget. This is the escape analog of CH-A's deliberately
   strong antenna baseline. **A/B integrity is preserved exactly:** amplifying a
   zero loom, or amplifying through zero loom weights, is still zero.

A third lesson came from the fitness itself: because target leading lets *any*
large maneuver break the intercept, **survival alone does not force
directedness** — an early short run found a single fixed turn that survived all
three azimuths. Weighting the directional terms (turn-away `w_head=5`, react-fast
`w_react=4`) well above survival (`w_survive=3`) makes **opposite** left/right
turns the fitness optimum; a fixed turn forfeits the away-bonus on one side and
loses. Directed escape then emerges in ~13 generations.

**Fitness** (averaged over front / left / right azimuths):

```
F = w_clear*min(closest_threat_dist, 18)     # flee away (graded clearance)
  + w_survive*[not hit]                       # survival
  + w_react*clip(early away-turn / 0.3, 0, 1) # react fast
  + w_head*clip(total away-turn, -pi/2, pi/2) # turn away
  - 0.05 * n_below                            # stay upright (keep walking)
```

---

## 3. Parallel speedup

CMA-ES population evaluated across a `ProcessPoolExecutor` (per-worker env,
`torch` single-threaded to avoid oversubscription). Benchmark (16 individuals,
9 workers): **sequential 29.0 s → parallel 5.8 s = 4.98× speedup**, with
`max|Δfitness| = 0.0` — **parallel is bit-identical to sequential** (onset and
threat geometry are seeded, so every controller sees the same threats). The full
run sustained ~14.6 s/generation (96 rollouts/gen) over 50 generations.

---

## 4. Results

### Escape / survival rate by azimuth (trained vs untrained baseline)

| azimuth | untrained escaped | untrained closest-approach | **trained escaped** | **trained closest-approach** |
|---|---|---|---|---|
| 0° (front) | ✗ | 2.22 | **✓** | **19.31** |
| 90° (left) | ✗ | 0.61 | **✓** | **19.14** |
| 270° (right) | ✗ | 1.93 | **✓** | **15.07** |
| **rate** | **0 / 3 (0%)** | mean 1.59 | **3 / 3 (100%)** | mean 17.84 |

The untrained walker is hit by every threat; the evolved controller escapes
every one, increasing its closest approach to the threat ~11× (1.6 → 17.8).

### Directedness (the headline: emergent opposite turns)

| azimuth | away-turn (Δyaw, away-frame) | **raw turn direction** | reaction latency |
|---|---|---|---|
| 90° (left threat) | **+2.07** | **right / CW** | 28 ms |
| 270° (right threat) | **+1.48** | **left / CCW** | 48 ms |
| 0° (front) | ±0 (no preferred side) | escapes by displacement | 44 ms |

Same controller, opposite threats → **opposite physical turns** (left threat ⇒
bolt right, right threat ⇒ bolt left), driven purely by `loom_L` vs `loom_R`.
The turn begins within **~30–50 ms** of loom onset. The two clips
(`flee_left.mp4`, `flee_right.mp4`) show exactly this mirror-image escape. The
untrained baseline has **no** directional response — its ±0.33 "away-turn" is
just fixed yaw drift (mean 0 across sides).

### Generalization (held-out azimuths)

Evaluated at azimuths the controller never trained on — {45°, 135°, 315°}:
**escape 3 / 3 (100%)**, mean closest approach 15.95. Survival generalizes
cleanly; the directional response is strongest near the trained azimuths and
fuzzier at the off-axis held-out ones (expected — see scope caveat below).

---

## 5. Caveats / honesty

- **Hand-built looming front-end.** `Threat` geometry + `read_loom` are an
  analytic stand-in for the real **LC4/LPLC2 → DNp01 (Giant Fiber)** escape
  circuit (LPLC2 ~ angular-size term, LC4 ~ expansion-rate term, DNp01 ~
  descending escape command). Swapping it for the connectome wiring is the
  endgame and is **not** done here; the two bilateral loom channels are the seam.
- **`loom_input_gain` amplification.** The bilateral cue is amplified ×8 before
  it enters the network so the bang-bang gait is steerable (§2). This is a
  deliberate, documented design choice, not a biological measurement — it makes
  the task learnable, exactly as CH-A used a larger-than-biological antenna
  baseline. A/B integrity is preserved (amplifying zero is zero).
- **180° "behind" omitted.** Azimuths are {front 0°, left 90°, right 270°} in the
  fly's onset-heading frame. From a forward walk, a directly-behind threat needs
  a full U-turn inside the short episode; the headline (directed left/right
  escape) is carried by the symmetric pair, so the behind case is deferred.
- **Short episodes.** ~1.2 s. Long enough for walk → loom → escape, but the
  generalization claim is scoped to the small azimuth panel plus the held-out
  azimuths above.
- **Fitness scalar is not cross-comparable.** The escape reward is task-specific
  (survival + directional shaping); do **not** compare its magnitude against the
  walking, closed-loop, or chemotaxis fitness scalars.

---

## 6. Export bundle → portfolio

Assets written to `outputs/web_data_x/`:
`escape_controller.json` (weights + full sensor spec), `flee_left.mp4`,
`flee_right.mp4`, `trajectories.json`, `escape_metrics.json`, `REPORT_x_a.md`.

Copy into the portfolio repo (no portfolio access from here):

```bash
mkdir -p portfolio/public/cellular-gaits/data-x
cp /Users/jarvis/dev/jarvis/cellular-gaits/outputs/web_data_x/* \
   portfolio/public/cellular-gaits/data-x/
```

**Do not merge** — branch `feat/x-escape` is for review.
