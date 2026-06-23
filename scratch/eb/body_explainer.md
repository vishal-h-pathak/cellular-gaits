# EB-0C — The body-side escape primitive (explainer)

**Branch:** `feat/eb-body` · **Module:** `src/cellular_gaits/embodied/body.py` ·
**Validation:** `scratch/eb/validate_body.py` → `scratch/eb/body_validation.json`

This is the **body half** of the embodied-brain escape loop
(`EMBODIED_BRAIN_PLAN.md`, Phase 1): an on-demand callable that makes the
NeuroMechFly fly **bolt/flee** when told to. It is built so it can be wired,
later (EB-1), to the real FlyWire connectome's **DNp01 (Giant Fiber)** firing
rate. It trains **nothing** — it *reuses* the already-evolved X-A escape
behavior.

---

## 1. What it is, in one paragraph

`apply_escape(env, drive, direction)` takes a **scalar escape drive** (0–1) and a
**direction**, and drives the MuJoCo fly into a fast, directed turn-and-flee for
one control window. Internally it reconstructs the trained X-A escape controller
(an 8-input "loom" neural cellular automaton, run `xa-full`, fitness 13.23),
synthesizes the **bilateral looming signal** that controller already knows how to
react to from the scalar drive + direction, and lets the trained network produce
the leg motion. The scalar `drive` **is** the clean coupling point: in EB-1 it is
where the brain's DNp01 firing rate plugs in.

## 2. Where the motion comes from (reuse, not retrain)

The project already evolved an escape controller in **X-A** (`REPORT_x_a.md`): a
looming threat made the fly turn *away* and flee, and **which way it bolted was
emergent** — it fell out of a left-vs-right looming asymmetry, not a hard-coded
turn. That controller is the loom NCA `NCA(loom=True)`:

```
conv1: Conv2d(4 state + 2 proprio + 2 loom = 8 → 16, 3×3, zero-pad)
       → tanh(gain·) → conv2: Conv2d(16 → 4, 1×1) → clamp[-1,1]
```

Its evolved weights are the **1236-value `flat_params`** vector. We **vendored**
that exact artifact into the module as
`src/cellular_gaits/embodied/escape_controller.json` (the same JSON the portfolio
`data-x` bundle ships) so the primitive is self-contained and reproducible across
machines. `EscapeMotor` loads `flat_params` → `NCA(loom=True).set_params(...)` and
the trained controller is back, bit-for-bit. **No new RL controller was trained**,
and none is required — the task's "would need sentry" follow-up does **not** apply.

## 3. The clean seam: a scalar drive → a synthetic bilateral loom

In X-A the controller reacted to a bilateral looming cue `(loom_L, loom_R)`
produced by a hand-built front-end watching a *simulated* threat. The body-side
primitive **drops the threat** and instead *synthesizes* that cue from the scalar
drive, using the X-A loom math **verbatim** (`loom_from_drive`):

```
m      = clip(drive, 0, 1)                 # escape drive   (≈ DNp01 firing rate)
φ      = bearing(direction)                # threat bearing, CCW-positive = left
loom_L = m · 0.5 · (1 + sin φ)
loom_R = m · 0.5 · (1 − sin φ)
```

Each control step the primitive injects `loom_L`/`loom_R` into the sensor map and
runs the trained NCA step — reusing the exact training-time policy
(`_make_escape_policy`, `loom_input_gain = 8`). A frontal cue (`φ=0`) excites both
"eyes" equally; a side cue excites one. **The L−R asymmetry is the directional
signal** — the same asymmetry X-A learned to turn into a directed bolt.

Why this is the right seam for EB-1: the connectome's escape pathway is bilateral
**LC4/LPLC2 → DNp01**. DNp01's rate is a scalar descending command (→ `drive`);
the left-vs-right LC4/LPLC2 activation is the bilateral cue (→ `direction`). So
the primitive's interface mirrors the biology it will be plugged into.

## 4. What "escape" looks like at the joint/leg level

The controller's output is a 42-D vector of **leg-joint motor targets** (7 joints
× 6 legs), clamped to [−1, 1], applied as position-actuator targets in MuJoCo at
250 Hz. The warm-start walking gait is *bang-bang* (every motor cell pinned at a
clamp); escape works by the loom cue **flipping the sign of specific motor cells**,
breaking the symmetric tripod walk into an asymmetric push — the legs on one side
drive harder than the other, yawing the body away from the looming side, then the
gait carries the fly forward along the new heading. So at the leg level, "escape"
is: *a brief, lateralized perturbation of the walking limit cycle that rotates the
body away and converts the walk into a flee*, not a separate scripted takeoff.

## 5. Validation (short MuJoCo rollout, this Mac)

MuJoCo/flygym runs natively here (no sentry needed). Protocol mirrors the X-A
regime — the fly **walks** (drive=0) for 120 steps to establish its gait, then a
**transient escape pulse** (`escape_pulse_drive`, peak = `CALIBRATED_DRIVE` = 0.2)
fires. The escape run and a no-drive baseline share the env seed and warm-up, so
they diverge *only* at onset; the difference is the loom's causal effect. Turn is
measured over the **reaction-latency window** (~60 ms — X-A's documented 28–48 ms).

| condition | loom-induced early turn | net displacement | min thorax-z (upright) |
|---|---|---|---|
| baseline (drive=0) | — | 12.7 | 0.654 |
| **left** loom | **−34.1° (CW / bolt right)** | 9.8 | 0.697 |
| **right** loom | **+5.2° (CCW / bolt left)** | 17.7 | 0.652 |

**All five headline checks pass:**
- **Directed** — left-loom and right-loom turn opposite ways.
- **X-A polarity** — left-loom → CW/right turn, right-loom → CCW/left turn (the
  emergent mirror-image escape from `REPORT_x_a.md §4`).
- **Stable** — the body stays upright (min thorax-z ≈ baseline).
- **Flees** — the body displaces several body-lengths.
- **Reproducible** — same drive/direction ⇒ bit-identical metric (seeded).

(Full numbers: `scratch/eb/body_validation.json`. Re-run:
`uv run python scratch/eb/validate_body.py`.)

## 6. The EB-1 contract (stable entry point)

```python
from cellular_gaits.embodied import EscapeMotor, apply_escape

# one-shot convenience (caches a default motor):
result = apply_escape(env, drive=0.2, direction="left", n_steps=150)
#   env       : a fresh/seeded FlyEnv with NO threat set (loom is synthesized)
#   drive     : scalar in [0,1]  — EB-1 maps the DNp01 firing rate onto this.
#               May be a callable drive(t)->float for a time-varying rate.
#   direction : "left" | "right" | "front" | "back", or a float bearing in
#               radians (CCW-positive = the fly's left = a threat on the left).
#   n_steps   : control-window length (steps @ 250 Hz).
#   returns   : {"traj", "metrics", "loom", "drive", "direction",
#                "bearing_rad", "run_id"}

# stateful / continuous use (hold CA state across windows):
motor = EscapeMotor()
result = motor.apply_escape(env, drive, direction="right", n_steps=150)
policy = motor.policy(drive, direction)   # policy(t, sensors) for a custom loop
```

`escape_metrics(traj)` returns `{displacement, heading_change, peak_speed, ...}`.
`loom_from_drive(drive, direction)` exposes the scalar→`(loom_L, loom_R)` map.
`escape_pulse_drive(peak, onset, ...)` builds a representative DNp01-rate pulse.

For EB-1, the natural use is: each brain↔body sync window, read DNp01's rate,
pass it (and the LC4/LPLC2 L/R bias as `direction`) as `drive`, call
`apply_escape` for that window, keep the `EscapeMotor` (and its CA state) across
windows.

## 7. Honest caveats (carry these)

- **Escape is a transient.** The directed impulse lives in the first ~40–60 ms
  after onset (X-A's reaction-latency band). A **held, strong, unilateral**
  synthetic loom drifts off the controller's training distribution and the turn
  becomes chaotic — so the body primitive is meant to be pulsed (the loom rises
  and falls as a threat passes), not driven at a sustained maximum. This matches
  how the brain will actually drive it: DNp01 fires a fast escape transient, not a
  DC level.
- **Moderate drive only.** `CALIBRATED_DRIVE ≈ 0.2`. Saturating the drive
  (≳0.5 held) overdrives the gait into a tumble (the body loses balance). EB-1's
  DNp01→drive map should target this moderate range; "more DNp01" is **not**
  "spin harder."
- **Left/right asymmetry.** The trained controller turns more strongly for a
  left-side cue than a right-side one (left −34° vs right +5° here). This is
  inherited from X-A, which documented the same asymmetry (away-turn 2.07 vs 1.48)
  and a fuzzier off-axis response — not a new artifact of the primitive.
- **Synthetic loom, not the real circuit.** The bilateral loom is still the
  hand-built analytic stand-in for **LC4/LPLC2 → DNp01**. Swapping it for the
  connectome circuit is exactly EB-1; this module leaves the scalar `drive` seam
  clean for that.
- **No takeoff/flight.** "Escape" here is a ground turn-and-flee (the X-A
  behavior). NeuroMechFly's wing/takeoff path is not used; if a true ballistic
  takeoff is wanted later, that is a separate body primitive.
