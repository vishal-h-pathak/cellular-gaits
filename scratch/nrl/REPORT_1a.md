# N-RL-1a — the RL env (env_base + nav_env). Report

**Branch:** `feat/n-rl-env` · **Gate 1: PASS** (`scratch/nrl/gate1.md`) · not merged.

## Files added (purely additive — no existing source touched)
- `pyproject.toml` — `+ gymnasium>=0.29,<2.0`; `uv.lock` regenerated (gymnasium 1.3.0).
- `src/cellular_gaits/rl/__init__.py` — package exports.
- `src/cellular_gaits/rl/env_base.py` — `EmbodiedRLEnv(gymnasium.Env)`, policy-agnostic,
  four pluggable hooks (DR sampler / FlyEnv builder / obs builder / reward fn) + action map.
- `src/cellular_gaits/rl/nav_env.py` — `NavRLEnv` + `NavRLConfig`, the nav task.
- `scratch/nrl/gate1.py` + `gate1.md` — the env-sanity gate and its report.

## The contract as implemented (1b/1c code against this)
- **Observation** `Box(float32, (6,8,8))`, per-channel bounds ch0∈[-1,1], ch1–5∈[0,1].
  Built every step via `NCA.build_nav_sensor_map(ja, fc, odorL, odorR, feelerL, feelerR)`:
  ch0 joint angles (7×6 motor block), ch1 foot contacts (row 7), ch2/3 odor goal-beacon
  L/R, ch4/5 feeler L/R. **Feelers are RAW [0,1].**
- **Action** `Box(float32, -1, 1, (42,))` → clipped → straight into `FlyEnv.step` (its
  existing clip+rescale to the ctrlrange is the actuator map, reused verbatim).
- **`reset(seed, options)`** — domain randomization samples goal bearing, obstacle position
  (fraction along the straight start→goal line + signed lateral offset so it clips the path),
  and obstacle radius. `options={"eval": True}` draws from a frozen, fixed-seeded held-out set
  (cycled deterministically); training resets rejection-sample away from it → genuine
  disjointness (gate: 0/5000 leaks). A fresh `FlyEnv` is built per reset (obstacle geoms are
  baked at construction); the Newton≤20/ls10 cap is applied automatically, obstacles-only.
- **`step`** → reward = `Δapproach − w_collide·in_contact − step_cost (+ reach_bonus iff
  reached AND episode collision-free)`. `terminated` on reach/fall; `truncated` on time limit.
  `info`: `collision, dist_to_goal, reached, detour_perp, feeler_L, feeler_R, goal_bearing`.

## Deviations from the spec (flagged loud)
1. **Feeler input gain (×8) is NOT in the observation.** The CMA-ES nav rollout multiplied
   the feeler by `feeler_input_gain=8.0` *before* `build_nav_sensor_map`. We keep the obs
   physical/bounded ([0,1]) and move that gain to the **policy** (1b's `NCAPolicy`, a
   documented knob defaulting to 8.0). Rationale: the gain is a first-layer input weight /
   encoding choice (env = sensory signal, policy = encoding — because the policy is what gets
   swapped for a real connectome later); warm-start bit-exactness is unaffected (feeler
   channels/weights are zero on a chemo warm-start). **1b must apply the ×8 (tunable).**
2. **`detour_perp`** is the signed perpendicular offset from the **straight** start→goal line
   (+ = body-left), not vs the warm-start fly's curved natural path (`_signed_detour` needs a
   per-layout rollout — too heavy under continuous DR). The obstacle is likewise placed on the
   straight line. Consistent and cheap.

## Gate 1 numbers (MAC, this build)
- Shapes/spaces correct; obs in-space every step; all `info` keys present.
- Termination-on-fall works (random fly falls ~step 28); time-limit truncation works.
- Vector envs **bit-deterministic** given a seed (max|Δ| obs+reward = 0 across an AsyncVectorEnv).
- Throughput (6 async envs): ~100 steps/s. **reset ≈ 250 ms vs step ≈ 2.6 ms** — reset (rebuild
  FlyEnv + warmup) costs ~97 steps' wall time. RL burns far more episodes than CMA-ES, so this
  is the throughput ceiling; if it dominates the full run, revisit (move the obstacle as a body
  rather than re-baking geoms each reset).
- DR varies bearing + obstacle every reset; held-out set frozen + repeatable; 0/5000 disjointness leaks.

## ⚠ Carried finding for WIN (important)
On this **`flygym==2.0.1` / MAC** build the obstacle geoms **do not physically block the fly**:
0/69 fly geoms are collidable (all `contype=conaffinity=0`); the obstacle is the only collidable
geom; 0 of 55 explicit contact pairs (foot-ground) involve it; a fly driven forward walks straight
through it. So no fly↔obstacle MuJoCo contact is generated, the pinned-fly heavy-contact regime
never triggers, and the **~40 s→~7 s worst case in `REPORT_n_a_calibration` (measured on WIN) is
not reproducible here** (capped == uncapped, ~1.9 ms/step). The Newton cap is verified *applied*
(obstacles-only) — a harmless safety belt bounding a cost that is ~0 here.

The nav **task** is unaffected — obstacles are sensed geometrically (feelers) and the collision
penalty is geometric time-in-contact — but physical *blocking* is absent on this build. Since
flygym is version-pinned and `uv.lock` is committed, WIN *should* match; **verify on WIN whether
physical blocking / the 40 s pin actually occurs**, since the N-A calibration implies it does. If
WIN also lacks blocking, the "fly bumps into a wall" realism is sensed-only everywhere (fine for
the task; worth knowing for the writeup/visuals).
