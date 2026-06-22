# REPORT — N-RL-PHYS · Make obstacles physically block the fly (real MuJoCo contact pairs)

**Branch:** `feat/n-rl-navigation` · **Date:** 2026-06-22 · **Machine:** MAC · single-threaded
**Status:** physics validated, RL **not** run. Surgical edit to `env.py` (obstacle path only) +
honesty note on `REPORT_n_a_calibration.md`. No-obstacle physics byte-exact; prior behaviors untouched.

## The finding being fixed
FlyGym builds every fly geom with `contype=conaffinity=0` and enables collisions ONLY through
explicit `<pair>` elements (it pairs each contact body segment with the ground in
`world.py::_set_ground_contact`). N-A's `_add_obstacles` added a bare cylinder with **no pair**, so
MuJoCo never generated a fly↔obstacle contact — **the fly walked through**. N-A's `collision_count`
was a *geometric* thorax-to-surface clearance test, and the "pinned fly → Newton cap" timings
described physics that never occurred. Vishal chose to make obstacles **truly physical**.

## The change (env.py, obstacle path only)
1. **Real contact pairs.** `_add_obstacles(world, fly, obstacles)` now runs **after** `world.add_fly`
   (the fly geoms must exist to pair against) and, mirroring `_set_ground_contact`, adds an explicit
   `contact.add("pair", geom1=<fly seg geom>, geom2=<obstacle cylinder>, …)` for **all 55**
   `LEGS_THORAX_ABDOMEN_HEAD` contact-body geoms (legs, thorax, abdomen, head) × each cylinder.
   Verified: one obstacle ⇒ `npair` 55→110 (55 ground + 55 obstacle); obstacle geom `contype/conaff=1/1`,
   fly geoms `0/0`, so the pairs are the *only* source of fly↔obstacle contact.
2. **Newton/CG cap retained, obstacles-only** (see Gate 3 — honest finding).
3. **Real-contact collision metric.** `collision_count` now counts control steps that end with an active
   fly↔obstacle MuJoCo contact (`mj_data.contact[:ncon]`, intersecting the resolved obstacle-geom and
   fly-contact-geom id sets), via `FlyEnv._obstacle_contact_active()`. `obstacle_min_surface` (geometric
   closest approach) is retained as a secondary diagnostic/shaping term. `step()` is left **byte-exact**.
4. **Comments corrected** (no false "default contact like the pebbles"; solver/clearance comments now
   truthful) and an **honesty note** added to `REPORT_n_a_calibration.md`.

### Contact-pair config (final, tuned)
| param | value | vs FlyGym ground default |
|---|---|---|
| `friction` (2 slide, 1 torsion, 2 roll) | `(1.0, 1.0, 0.02, 1e-4, 1e-4)` | identical |
| `solref` (timeconst, dampratio) | `(4e-4, 1.0)` | timeconst 2× faster (stiffer), crit. damped |
| `solimp` (dmin, dmax, mid, sharp) | `(0.99, 0.999, 0.5, 2.0)` | higher impedance floor/ceiling |
| `margin` | `5e-3` | 5× larger (anti-tunnel headroom) |

**Why they differ from ground — and the honest caveat:** the lateral slam of a fly into a vertical
post is a harsher regime than feet-on-floor, so I started from the ground defaults and stiffened. **But
I A/B'd both and the ground defaults *also* block cleanly with no tunneling/instability across the
entire realistic envelope** — stiffening was *precautionary, not required*. It only marginally lowers an
already-small post-contact rebound (~0.05 vs ~0.11 units at the 118 u/s peak gait speed) and adds
anti-tunnel margin, at no measured cost. The stiffer values were kept for the marginally cleaner stop.
(No green gate rests on unstable-but-blocking contact: the stop is clean, see Gate 1/2.)

---

## The four physics gates

### ✅ Gate 1 — blocking is real
- **Head-on drive** (constant forward thorax force into a post at x=8, r=2): the thorax **stops at
  x≈5.48** (≈0.5 short of the near surface = thorax half-width + margin), **390/400 control steps in real
  contact**, never beyond the post. Clean stop, stable.
- **Realistic gait** (warm-start forager driven onto an on-path post at (9,0), r=2): the thorax
  **never enters the cylinder interior** (`min surface-distance = 1.79 > 0`, `thorax_ever_inside_post =
  False`), `collision_count = 3` real contacts, trajectory finite. The forager *deflects and navigates
  around* the finite-width post — the correct no-tunnel criterion is **non-penetration**, not
  "x stayed behind the post" (a finite post can be walked around).
- On the 4 trained block conditions the physical post now **stops the untrained forager from reaching**
  (`reached=False`), where the non-physical N-A posts let it walk through and "reach". Genuine impediment.

### ✅ Gate 2 — no tunneling
Controlled head-on **coast**: impose a forward closing speed `v` until first contact, then **release**
(the contact constraint alone governs the collision — velocity is never overridden during contact).
Swept `v ∈ {10…1600} u/s` × `radius ∈ {1,2,3}`:
- **Zero tunneling, zero penetration, zero NaN at every speed and radius** — including **1600 u/s, ~13×
  the measured peak gait speed (118 u/s)**. `min surface-distance` stays ≈ +0.5 (thorax never inside).
- Worst case tried: v=1600 u/s, r=3 → blocked, not tunneled; rebound grows to a hard elastic bounce
  (~5 units) at that unphysical speed but **never passes through**.
- **Tunneling-onset speed: none up to 1600 u/s.**
- Note: an earlier *force*-drive showed "tunneling" only at absurd forces (≥160) where the **integrator
  itself diverges (NaN QACC)** — a force-explosion artifact, not a contact failure, and confirmed benign
  by the velocity-controlled coast above. The gait cannot produce such forces.

**Gait top speed (sets the margin):** warm-start forager homing freely — **peak 118, p95 75, mean 38 u/s**
(the bang-bang gait is jerky; net homing is ~4.5 u/s). Per physics tick at 118 u/s the body moves
~0.012 units vs cylinder radius ≥1.0 — three orders below the tunneling scale.

### ✅/⚠️ Gate 3 — cost is bounded (with an honest correction)
Cap is **obstacles-only** (confirmed): no-obstacle env keeps MuJoCo default `solver=2 (Newton), iters=100`;
obstacle env uses `solver=1 (CG), iters=30, ls=20`.

Worst-case **pinned** rollouts (sustained press; and an 8-post cage + flailing legs), capped(CG) vs
uncapped(Newton), single-threaded:

| scenario | ncon_peak | capped CG per-step | uncapped Newton per-step | ratio |
|---|---|---|---|---|
| sustained pin (1 post) | 7 | 5.61 ms | 3.05 ms | 0.54× |
| 8-post cage | 23 | 10.74 ms | 8.72 ms | 0.81× |
| 8-post cage + flailing | 16 | 7.14 ms | 4.90 ms | 0.69× |

**Honest finding:** with **real** contacts the fly's contact set stays **small (ncon ≤ ~23)** even in the
worst cases I could construct, and at that size **uncapped Newton is actually ~1.2–1.9× *faster* than the
CG cap** — the cap is **not a measured throughput necessity** and may mildly *cost* throughput. The N-A
"Newton blows ~2 s→~40 s" numbers came from the **non-physical regime** (no fly↔obstacle contacts ever
existed) and were **not reproduced**. The cap is **retained** per the brief (cheap, bounded, and changing
the solver would alter obstacle-env contact resolution), but **wave-2 should re-benchmark on real RL
candidates and may drop it.**

**Throughput (single-threaded, NCA policy, 1000-step / 4 s episodes):**
- **per-reset (reset+warmup): ~53 ms**
- **per-step: ~2.5 ms normal**, up to **~5.6 ms** in a sustained pin
- **per-episode: ~2.5 s normal**, worst-case pinned ~5.6 s.

### ✅ Gate 4 — no-obstacle A/B byte-exact
Post-edit no-obstacle rollout vs a pre-edit baseline (same deterministic policy/seed/steps):
`max|Δ|` = **0.000e+00** on thorax_xyz, joint_targets, yaw, thorax_z, **and scalar fitness**.
All 7 existing `scripts/test_navigation.py` checks pass (NCA-level A/B, env-level A/B, feeler signing,
bilateral steering, finite fitness), as does the obstacle-physical-and-sensed test (now real contacts).
chemo / loom / closed-loop / v1 untouched (every obstacle change is gated on obstacles-present).

---

## collision_count distribution (real contacts) — for wave-2 `w_collide` re-tuning
| regime | real `collision_count` |
|---|---|
| clean pass / no obstacle | **0** |
| clean detour (target) | **≤ 8** (the existing `COLLISION_OK_STEPS` graze tolerance) |
| plow straight in (4 trained block conditions) | **11–28** |
| (for reference) N-A *geometric* counts | 53–133 |

The magnitude **and meaning** changed: real contacts measure *time pressed against* a post (the fly is
stopped at ~0.75–0.93 surface distance and can't get closer), where the geometric metric measured *depth
walked through*. With `w_collide=0.2` the plow-in penalty is now only **~2.2–5.6** (was ~10–27) — it **no
longer dominates `reach_bonus=8`**, so detouring is no longer the clear fitness optimum. **Wave-2 must
raise `w_collide`** (≈0.5–1.0 reproduces the old ~5.5–28 penalty band) and re-confirm Gate-4-of-N-A
(both-sides detour achievable) against these real numbers.

## Reach-bonus / avoidance gate — confirmed keyed off the real-contact signal
No evolver/metrics edits were needed; both consumers already key off `collision_count`, which now carries
**real** contacts (never off the geometric `obstacle_min_surface`):
- `navigation.py`: `avoided = collision_count ≤ COLLISION_OK_STEPS (8)` → `detour_success = reached and
  avoided and detour_correct`. A **brief graze (≤8 real-contact steps) is still tolerated**, so a
  legitimate close pass is not over-punished.
- `evolve_navigation.py`: collision enters as the **soft penalty** `w_collide * collision_count` (the reach
  bonus itself is gated on distance-to-goal, not a hard collision-free gate — an honest description of the
  existing structure, left unchanged).

## Recommendation on episode length / reset cost
- **Keep `NAV_ROLLOUT_STEPS = 1000` (4 s).** It already accounts for the warm-start forager's natural path
  (closest approach near step ~416); the physical post adds detour distance, so 4 s remains adequate and
  shortening it would mis-time the obstacle placement.
- **Throughput ceiling:** ~2.5 s/episode normal, ~5.6 s worst-case pinned, +53 ms/reset, single-threaded.
  At the N-A full-run scale (pop 48 × 70 gens × conditions) budget compute on **WIN with `--workers ~16`**;
  the CG cap does **not** improve this (it slightly hurts) — wave-2 may benchmark dropping it.
- **Re-tune `w_collide`** against the real 11–28 plow-in band before/within the full run (see above).

---

**Confirm and the integrate session can wire RL against the physical task.** Do **not** run RL here.

### Repro (scratch/nrlphys/)
```
uv run python scratch/nrlphys/gate4_ab.py          # Gate 4 byte-exact (needs baseline_noobstacle.npz)
uv run python scratch/nrlphys/gate_controller.py   # gait speed, Gate 1, collision distribution
uv run python scratch/nrlphys/gate2_coast.py       # Gate 2 controlled-speed tunneling sweep
uv run python scratch/nrlphys/gate3_cost.py        # Gate 3 cap obstacles-only + capped/uncapped + cost
uv run python scratch/nrlphys/gate3_worstcase.py   # Gate 3 large-contact-set probe (cage + flail)
uv run python scripts/test_navigation.py           # existing suite (all pass)
```
