# REPORT — N-RL · Obstacle navigation via PPO: integration + calibration gate
Validation-first calibration for the RL navigation stack (env + NCA policy + PPO), run on **sentry (WIN, 5900X + 3080 Ti)** with the obstacles **physically real** (N-RL-PHYS). `w_collide=0.75` and the Newton/CG contact-solver cap dropped back to MuJoCo's default Newton (Gate 1 asserts `ncon` stays bounded). The stack wires cleanly and the harness is correct (Gates 1, 3 ✅); the calibration also surfaced two real findings the full run must act on (Gates 2, 4 ⚠️) — detailed below, not papered over. **The full run has NOT been launched.**

## Verdict at a glance
| gate | verdict | one-line |
|---|---|---|
| 1 env sanity + ncon bounded | ✅ | shapes OK, ncon_peak=21≤25, deterministic, 7.6× parallel |
| 2 N-A baseline in this env | ⚠️ | N-A clean-reaches 0.62 of held-out → **this band is too easy; widen it** |
| 3 A/B bit-exact | ✅ | warm-start == chemo forward, max\|Δ\|=0 |
| 4 short PPO signal | ⚠️ | collisions 120→0 but reach →0.00 (**over-cautious at this budget**) |

## Files added / changed
- **`scripts/run_rl_navigation.py`** (new) — the `--calibrate` / `--full` CLI that wires `NavRLEnv` + warm-started `NCAPolicy` + `rl.ppo.train`, with nav's held-out detour/reach/collision metrics as the PPO `eval_fn` and W&B on.
- **`scripts/test_rl_navigation.py`** (new) — smoke test: env + policy + one real PPO update step run clean.
- **`src/cellular_gaits/rl/nav_env.py`** (additive) — `NavRLConfig.drop_solver_cap` knob (default `False` → prior obstacle-env physics bit-exact); when `True`, `_build_fly_env` restores MuJoCo's default Newton solver (the same values the no-obstacle / chemo / loom path already uses, so those byte-exact paths are unaffected). `w_collide` was already a config field (set to 0.75 here, no code change). No other source touched; `nca.py` / `env.py` / `evolve*.py` unchanged.

## The four gates
### ✅ Gate 1 — env sanity on the now-physical task (ncon bounded)
- obs/action spaces + in-space rollout: `True`; obs in-space every step of a 729-step warm-start rollout: `True`.
- **Newton cap dropped** (per the carried finding): solver=`2` (2 = Newton), iters=`100` (uncapped). With the cap gone, **`ncon_peak = 21` ≤ 25** over the rollout → the contact set stays bounded, so dropping the cap introduces no instability (matches N-RL-PHYS gate 3's ≤23). The assertion replaces the old 'confirm the cap' check.
- vectorized determinism (16 async envs, same seed + action stream): reset obs identical = `True`, max|Δ| (obs+reward) = `0.00e+00` (0 = bit-deterministic).
- throughput (standing action, pure stepping): **1808 steps/s** with 16 async envs vs 238 solo → **7.6×**. Reset (rebuild FlyEnv + warmup) ≈ 462 ms is the per-episode throughput ceiling RL pays (the obstacle geoms are baked per layout).
### ⚠️ Gate 2 — N-A overfit baseline, measured in THIS env (FINDING)
The N-A CMA-ES nav controller (`outputs/web_data_n/navigation_controller.json`) loaded as the policy and evaluated deterministically on this env's frozen held-out set (n=8):

| metric | N-A on held-out |
|---|---|
| clean_reach_rate (reach, **zero** contact) | **0.62** |
| reach_rate | 0.62 |
| detour_success_rate | 0.38 |
| avoid_rate (≤ 8 contacts) | 0.88 |
| detour_correct_rate | 0.50 |
| mean_collisions | 119.8 |
| mean_min_dist | 3.86 |

- **The expected ≈0 did NOT reproduce on this held-out set — and that is itself the finding.** N-A clean-reaches **62%** of these layouts. The per-episode rows show why: N-A succeeds at bearings ≈35–56° and fails only at the low-bearing edge (≈23°, where one episode pinned for ~960 steps). This env's held-out band (`bearing_deg_range=[20.0, 60.0]`) **overlaps the ~40° cone the forager (and hence N-A) is already competent in**, so it is not a hard generalization test. The original export's held-out 0.00 came from a *harder* set (offset/near/radius perturbations) in the *non-physical* env.
- **Consequence for the yardstick (acted on below):** to make 'RL beats N-A' a real claim, the full run's held-out set must reach **outside** the forager's competence — i.e. **widen the bearing band toward omnidirectional**. Within ~40°, CMA-ES already does fine and there is little for RL to prove.
- **Physical-deflection caveat:** obstacles are now physical, so a fly that walks into one is *passively deflected toward the open side* — a purely geometric `detour_correct` is partly produced by physics, not skill. `clean_reach` (reach with NO contact) is the discriminator physics cannot fake, so it is the headline metric here and below.
### ✅ Gate 3 — A/B integrity (warm-start == chemo forward, bit-exact)
- Feeler-zeroed warm-start `mean_action` vs the trained 8-input chemo forward: max|Δ| = `0.00e+00`.
- Feelers ON (raw 1.0/0.6 × the policy's ×8 input gain) vs chemo forward: max|Δ| = `0.00e+00` — amplifying a feeler reading through the zero-initialized feeler weights is still exactly zero, so A/B holds **regardless of the gain knob** until training moves those weights.
- warm start: 1236 chemo params → 1524 nav params (== `NCA(nav=True)`).
- **Existing tests / prior behaviors:** `test_nca` ✅ and `test_rl_policy` ✅ (A/B bit-exact incl. gain-invariance + shuffled-rescore). `test_navigation`: the A/B + bit-exact rollout checks pass; it fails ONLY on a pre-existing physical-feeler assertion (`feeler peak 0.14 < 0.2`) that reproduces byte-identically on a clean checkout with this work stashed — i.e. an env.py/flygym-build property, NOT a regression from this additive change. `test_chemotaxis` / `test_escape` fail only because a gitignored artifact (`outputs/web_data_c2/closed_loop_controller.json`) is absent on this box — environmental, not a code regression. My change is confined to `rl/nav_env.py` (additive, default-off) + two new scripts; `nca.py`/`env.py`/`evolve*.py` are untouched, so all prior behaviors are bit-exact.
### ⚠️ Gate 4 — short PPO run (FINDING: avoidance learns, reach collapses)
Short PPO: `total_steps=48000`, `n_envs=16`, `n_steps=500`, `n_minibatch=20`, `update_epochs=4`, `lr=0.0003`, `ent_coef=0.0` → 6 updates in 620s.

| held-out metric | N-A baseline | pre-PPO (warm start) | post-PPO |
|---|---|---|---|
| clean_reach_rate | 0.62 | 0.00 | **0.00** |
| reach_rate | 0.62 | 0.25 | **0.00** |
| detour_success_rate | 0.38 | 0.12 | **0.00** |
| mean_collisions | 119.8 | 48.2 | **0.0** |
| avoid_rate | 0.88 | 0.50 | **1.00** |

- **PPO is clearly learning — in the wrong direction for now.** Over 6 updates it drove held-out collisions **48→0** and avoid_rate to **1.00**, but reach fell to **0.00**: at `w_collide=0.75` and a tiny 48,000-step budget the fastest way to cut the penalty is to **stop advancing into the obstacle at all** (an over-cautious 'stall' basin), not to learn the harder detour-and-home. So there is a real PPO gradient on the avoid half; the joint avoid+home behavior needs the diagnosis below.
- Net win over N-A on the un-gameable bar (clean_reach/detour up, collisions down with reach held): **False** — collisions collapsed but reach did too, so no net win at this budget. Reported straight, not papered over (per the master prompt).

## Diagnosis (Gate 4) + reward balance against the real contact scale
- **`w_collide` balance (the Gate-4 ask).** Real contacts run ~11–28 in-contact steps/episode (N-RL-PHYS), so `w_collide=0.75` integrates to ~8–21 of penalty on a colliding episode — comparable to the full approach (~source_distance 18) plus the one-off `reach_bonus=8.0`. That balance makes a *clean* detour the true optimum, but it also makes **'stall short of the obstacle' (penalty 0, partial approach) competitive with 'push through and graze' at a short budget** — exactly the basin PPO fell into. N-A's old 0.2 never bit because its collision count was the inflated geometric 100+; at the real ~20, 0.2 integrates to ~4 (negligible) — confirming 0.2 is too low and 0.75 is in range, but needs a *ramp*.
- **What to try in the full run (in priority order):** (1) **anneal `w_collide`** — start low (~0.2–0.3) so homing is learned first, ramp to ~0.75 once reach is stable, so the policy never discovers the stall basin before it can home; (2) **far→near obstacle curriculum** — obstacle off-path / distant early, slid onto the path as reach stabilizes; (3) **much larger budget** (the 48k calibration is ~8 updates; the joint task needs the full run); (4) optionally a small `ent_coef` (e.g. 1e-3) to keep the policy exploring past the stall basin. (5) `w_collide` sweep 0.5–1.0 once homing survives.

## Proposed full run (NOT launched)
```
uv run python scripts/run_rl_navigation.py --full --full-steps 3000000 --n-envs 16 --w-collide 0.75
```
- **Step budget:** **3,000,000 env steps**. The 48,000-step calibration took 620s (~77 steps/s) — but that rate is dominated by held-out eval (5 evals × 8 episodes of up to 1000 steps each on a single env); raw training throughput is the 1808 steps/s of Gate 1, with reset (≈462 ms, the per-episode ceiling) the real limiter. With `eval_every` amortized over a long run, ~3,000,000 steps is a **~2–6 h** job — checkpointed every 25 updates and resumable, so it can span sessions.
- **Bearing-width decision — RECOMMENDATION: bet WIDE, on a curriculum ramp.** Gate 2 shows the moderate band `[20.0, 60.0]` is *too easy* (N-A already clean-reaches 62%), so narrow inherits the ~40° envelope and wastes the reason to use RL. Widen toward omnidirectional over the run (start at the calibrated band, widen as held-out reach stabilizes) — this bets PPO **relearns omnidirectional homing** under the broader reward, the higher-value outcome (it subsumes the separate 'omnidirectional forager' upgrade). Risk: homing breaks outside the band before avoidance is learned — mitigated by the warm start + the ramp + annealed `w_collide`. **The full run must ALSO widen the held-out set to match**, or the yardstick stays uninformative (Gate 2). (These widenings are config/curriculum changes to wire before the full run; flagged here as the go/no-go decision, not silently applied.)
- The `--full` path already implements warm-start + PPO + held-out eval + checkpointing; the curriculum ramp + `w_collide` anneal are the deltas to add before launch.

## What transfers to the connectome endgame
- **Task-agnostic (transfers verbatim):** `rl/ppo.py` (the whole PPO loop — rollout, GAE(λ), clipped surrogate, minibatch epochs, opaque per-env recurrent-state threading, checkpoint/resume, W&B, the `eval_fn` hook) and `rl/env_base.py` (`EmbodiedRLEnv` — the policy-agnostic FlyEnv→Gymnasium wrapper with its four pluggable hooks). A future `ConnectomePolicy` (a FlyWire sub-circuit, itself stateful) drops into the same `train(cfg, make_env, agent, eval_fn)` call unchanged — the explicit state threading was built for exactly that recurrent case. `scripts/run_rl_navigation.py`'s eval/“measure a policy on a frozen held-out set” harness is reusable too.
- **Nav-specific (rebuilt per task):** `rl/nav_env.py` (the DR sampler, obs builder, reward, held-out set) and the warm-start path in `rl/policies.py` (`NCAPolicy` wraps `NCA(nav=True)`; the connectome task supplies its own policy module). The escape behavior is where the real connectome seam (LC4/LPLC2→DNp01) lives; **navigation has no clean real-circuit seam** — the RL *harness* is the bridge, not the nav behavior.

## Honesty caveats (carry into the export meta)
1. **Feelers are a hand-built rangefinder abstraction**; real *Drosophila* avoid via vision / optic flow / looming. Navigation is a robotics-flavored capability demo, not a connectome bridge.
2. **Reactive local avoidance + gradient homing, not path planning / spatial memory** — can trap in concave / dead-end layouts (a local minimum). Not hidden.
3. If RL **widens the homing envelope**, that is the controller *relearning steering under a broader reward*, not new biology. The DR distributions and reward weights (`w_collide`, `reach_bonus`, `step_cost`, the bearing band) are **design choices**, stated here; the policy is tuned to them.
4. Nav return / fitness is **not comparable** across behaviors.
5. The Gate-2 baseline is N-A on a **different** frozen held-out set than the original export, in the **now-physical** env — an honest in-env yardstick, not a re-run of the old numbers; it revealed the band is too easy (a finding, not a failure).
6. The Gate-4 short run is 8 updates on a tiny budget — it shows a *direction* (avoidance learns fast, homing needs the curriculum/anneal), not a converged controller.

---

**Confirm and I'll launch the full run.** (Recommended: first wire the bearing-curriculum + `w_collide` anneal + widened held-out set per the findings above — say the word and I'll add those before launching.)
