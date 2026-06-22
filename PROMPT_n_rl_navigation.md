# Claude Code — N-RL · Obstacle navigation via RL (PPO + domain randomization), built as the reusable connectome harness

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors intact and
> **bit-exact** (`nca.py`, `env.py`, `evolve*.py`, `navigation.py`, existing `checkpoints/`,
> `outputs/`, `web_data*`). No `rm -rf` of anything you didn't create; no broad/glob deletes;
> scratch in ONE dir you make (`scratch/nrl/`), clean up only that exact path. Append to shared
> files; don't rewrite. `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh` are off-limits.
> **Commit your work on the branch before finishing** (don't merge to main). When unsure, leave
> it and note it. (Full: `AGENT_SAFETY.md`)

> **Run from the `cellular-gaits` repo root, on the workstation (`sentry`: 5900X + 3080 Ti).**
> New branch off the nav line: `git checkout feat/n-navigation && git checkout -b feat/n-rl-navigation`.
> Use a todo list. **Read first:** `../portfolio/docs/cellular-gaits/research-roadmap.md` (the
> behaviors ledger + the bigger arc — note navigation has **no connectome seam**; the endgame
> runs through escape's seam), `../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md` (how we work +
> the **visualization principle** below), `REPORT_n_a_calibration.md` and the N-A run result
> (overfit: trained detour_success 0.25, **held-out 0.00**, 4/8 reach), and the code you're
> extending: `src/cellular_gaits/nca.py` (the `nav` mode + `warm_start_from_chemo`),
> `src/cellular_gaits/env.py` (`ObstacleField`/`read_feelers`, `OdorField`/`read_chemo`, the
> Newton-iteration contact cap for obstacle envs), `src/cellular_gaits/evolve_navigation.py`
> (`NavConfig`, `conditions`, `nav_fitness` — the episode reward we'll convert to per-step).

## Why we're here (the point — don't lose it)

N-A trained the nav controller with **CMA-ES over 4 fixed layouts** (`g40_{near,far}_block_{left,right}`).
With 1524 params it **memorized four geometries** instead of learning to read the feelers: held-out
detour 0/8, reach 4/8. The calibration even warned us (Gate 4: 0/48 candidates detoured correctly on
both blocking layouts). Two root causes: **(a)** four fixed layouts can't force a general policy, and
**(b)** the warm-start forager only homes cleanly in a narrow ~40°, left-biased band, which forced the
training cone narrow in the first place.

**RL fixes both.** PPO trains over *thousands* of randomized episodes, so **domain randomization**
(resample the obstacle and the goal bearing every reset) forces a general feeler→action policy, and
the sample budget is enough to let the controller **relearn omnidirectional homing** — potentially
dissolving the ~40° envelope (subsuming the separate "omnidirectional forager" upgrade) in one run.

**The strategic payoff — this is a dry run for the connectome endgame.** The real goal (FlyWire
LC4/LPLC2→DNp01 driving the body) is also an RL-on-the-3080-Ti problem. So **build the RL stack as a
reusable, policy-agnostic harness** here, on a cheap task, then later plug the connectome sub-circuit
into the *same* env + PPO loop. That reuse is the main reason this work is worth doing — navigation
itself is off the critical path.

## The build

### 1. A reusable RL harness — `src/cellular_gaits/rl/` (policy-agnostic by design)

- `env_base.py` — `EmbodiedRLEnv(gym.Env)`: wraps `FlyEnv` as an **episodic Gymnasium env** with
  *pluggable* (i) observation builder, (ii) action→actuator mapping, (iii) reward fn, (iv)
  domain-randomization sampler called on `reset()`. Apply the **Newton≤20 / ls 10 contact cap** in
  every env that has obstacles (from N-A calibration — keeps a pinned-fly rollout ~7 s not ~40 s;
  applied ONLY when obstacles present so chemo/loom/closed-loop/v1 physics stay byte-exact).
- `policies.py` — `NCAPolicy`: wraps the existing `NCA(nav=True)` (1524 params) as a **Gaussian
  policy** — the NCA forward pass is the action mean, plus a learned global `log_std`; value head is
  a small MLP on the pooled grid. **Warm-start from the chemotaxis forager** as today (feelers
  zero-init → starts as the pure forager). Keep it policy-agnostic: a future `ConnectomePolicy`
  (the FlyWire sub-circuit) must drop into the same interface.
- `ppo.py` — a minimal **PPO** loop (cleanrl `ppo_continuous_action` is the template — single-file,
  so the custom NCA policy plugs in directly; SB3 is the faster-to-stand-up but less-flexible
  alternative — pick one, justify briefly). Vectorized envs via `gymnasium.AsyncVectorEnv` /
  Subproc across the 5900X (target ~16 parallel envs); GPU runs the policy + update. Log to W&B
  (see `PROMPT_wandb_integration.md`) — return curves, **held-out** eval, collision rate.

### 2. The nav task — `nav_env.py` (the env spec)

- **Observation:** the same signals the NCA reads — proprio (joint angles + foot contacts), odor
  L/R (goal beacon), feeler L/R (obstacle proximity), as the `(6,8,8)` sensor grid the `nav` mode
  expects. No new senses; this is the N-A sensor set, now read per-step by PPO.
- **Action:** the motor actuator targets the controller already drives (continuous, clamp [-1,1]).
- **Reward (per-step):** `Δapproach` (decrease in distance-to-goal this step) `− w_collide·in_contact`
  `− step_cost` `+ reach_bonus` on arrival. The per-step decomposition of N-A's episode fitness.
  **Require collision-free for the reach bonus** (N-A let it "reach by grazing" — 53 collisions still
  paid; close that hole).
- **Domain randomization on `reset()` (the fix):** sample obstacle xy (continuous, on the homing
  path), obstacle radius, and **goal bearing** from distributions — NOT 4 fixed layouts. Hold out a
  disjoint set of (bearing × obstacle) layouts, fixed-seeded, for eval — this is the yardstick vs N-A.
- **Curriculum (recommended for PPO stability):** stage A — narrow bearing band (the forager's
  reachable cone) + obstacle far; stage B — widen the bearing range toward omnidirectional and move
  the obstacle onto the path as the return stabilizes. **Decision to surface in the report:** how
  wide to push bearing — keeping it narrow inherits the ~40° limit; pushing it wide bets PPO can
  relearn homing (the higher-value outcome). Recommend betting wide with the curriculum as the ramp.

## VALIDATION-FIRST — build everything, run a SHORT calibration, then STOP and report (do NOT launch the full run)

Report (`REPORT_n_rl_calibration.md` at repo root) must confirm:

1. **Env sanity:** random-policy episodes run; obs/action/reward shapes correct; the Newton cap holds
   worst-case rollout to ~7 s; vectorized envs are deterministic given a seed; speedup with N parallel
   envs reported.
2. **Baseline = the N-A overfit, measured in this env:** load the N-A CMA-ES controller as the policy
   and eval it on the **held-out** layout set — it should reproduce ≈0 held-out detour success. This
   proves the held-out yardstick is honest before we claim RL beats it.
3. **A/B integrity (don't break the past):** with obstacles/feelers absent, the warm-started policy
   still walks + homes; `nca.py`/`env.py` changes are purely additive; v1/chemo/loom/closed-loop +
   all existing tests (`test_nca.py`, `test_chemotaxis.py`, `test_escape.py`, `test_navigation.py`)
   pass and stay **bit-exact**.
4. **Short PPO run shows generalization signal:** a small step budget already lifts **held-out**
   detour_success and reach **above the N-A baseline** (even modestly), with collision rate falling.
   If it doesn't move, diagnose (reward scale, DR too hard, curriculum) before proposing the full run.

End with the proposed full-run command + step budget + the bearing-width decision, and:
*"Confirm and I'll launch the full run."* Write calibration artifacts to `scratch/nrl/`.

## Honesty caveats (carry into the report + eventual export meta)

1. **Feelers are a hand-built rangefinder abstraction; navigation has no clean real-circuit seam**
   (real flies avoid via vision/optic flow). This stays a robotics-flavored capability demo, not a
   connectome bridge — the RL *harness* is the bridge to the endgame, not the nav behavior.
2. **Reactive local avoidance + gradient homing, not path planning / spatial memory** — can trap in
   concave/dead-end layouts (a local minimum). Don't hide it.
3. If RL **widens the homing envelope**, report it as the controller *relearning steering under a
   broader reward*, not as new biology. DR distributions and reward shaping are **design choices** —
   state them; the policy is tuned to them.
4. Nav fitness/return is **not comparable** across behaviors.

## Visualization principle (hard rule — applies to every visual this work produces)

Every visualization must build **intuitive understanding** of the design and the experiment by tying
**directly to the fly's anatomy** (and, as the work reaches real circuitry, to **specific brain
regions**). For this campaign: show the feelers and the odor beacon **on the fly's head**, the
seek-vs-avoid arbitration **on the body**, the goal bearing **relative to the animal**; make the RL
story legible (e.g., the training-layout cloud vs the held-out set from the fly's-eye view, so a
viewer *sees* why domain randomization cured the memorization). No decorative/abstract diagrams.
Keep the harness's eventual connectome reuse in view — visuals there will anchor to the actual
LC4/LPLC2→DNp01 neurons. (Full statement in `PARTNER_BRIEF.md` → Conventions & taste.)

## Definition of done (this session)

- `src/cellular_gaits/rl/{env_base,nav_env,policies,ppo}.py` — the **policy-agnostic** harness +
  the nav task; new deps added via `uv` (`gymnasium`, and SB3 *or* a cleanrl-style PPO — torch is
  already present); `pyproject.toml` + `uv.lock` updated.
- `scripts/run_rl_navigation.py` with `--calibrate` / `--full` gates mirroring the
  `run_evolution_*` CLIs; a `test_rl_navigation.py` smoke test.
- All prior behaviors untouched + bit-exact; all existing tests pass.
- Calibration run done, the four gates reported, full-run command + bearing-width decision proposed —
  **stopped, not launched.**
- **Committed on `feat/n-rl-navigation`** (not merged). Report: files added/changed, the four gate
  results, the proposed full-run numbers, and an explicit note on **what of this harness transfers
  to the connectome endgame** (which modules are task-agnostic vs nav-specific).
