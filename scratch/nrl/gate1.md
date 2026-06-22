# Gate 1 — N-RL env sanity (PROMPT_n_rl_1a_env.md)

Env layer only (`rl/env_base.py` + `rl/nav_env.py`); random policy.

## 1. Random-policy episode + shapes/spaces

- observation_space : Box(6, 8, 8) float32, per-channel ch0∈[-1,1], ch1-5∈[0,1]
- action_space      : Box(-1,1,(42,)) float32
- random episode    : 28 steps, terminated=True truncated=False (random fly falls -> termination-on-fall works)
- truncation test   : 1-step limit, zero action -> terminated=False truncated=True  -> PASS (time-limit truncation works)
- reward range       : [-0.4681, 0.2023], sum=-3.0446
- obs in-space every step: PASS
- info keys present : ['collision', 'detour_perp', 'dist_to_goal', 'feeler_L', 'feeler_R', 'goal_bearing', 'reached']  PASS
- obs ch min/max    : ch0[-0.55,0.56], ch1[0.00,1.00], ch2[0.00,0.16], ch3[0.00,0.17], ch4[0.00,0.00], ch5[0.00,0.00]

## 2. Newton contact-cap — applied obstacles-only; worst-case bounded

- cap applied on obstacle env (NavRLEnv): solver=2 iterations=20 ls_iterations=10  -> PASS (target solver=2/iter=20/ls=10)
- obstacle-FREE FlyEnv keeps MuJoCo defaults: iterations=100 ls_iterations=50  -> cap is obstacles-only, v1/chemo/loom byte-exact
- worst-case rollout (1000 steps, fly driven into obstacle ahead):
  - CAPPED   (iter=20, ls=10) :   1.95 s  (1.95 ms/step)
  - UNCAPPED (iter=100, ls=50):   1.97 s  (ratio 1.01x)

- **FINDING (flag for WIN verification):** on this machine's `flygym==2.0.1` build the obstacle geoms do **not** physically block the fly. Evidence: 0/69 fly geoms are collidable (all `contype=conaffinity=0`); the obstacle is the only collidable geom; of 55 explicit contact pairs (foot-ground), 0 involve the obstacle; a fly driven forward walks straight through it. So no fly-obstacle MuJoCo contact is generated, the pinned-fly heavy-contact regime never triggers, and the ~40 s->~7 s worst case in REPORT_n_a_calibration (measured on WIN) is **not reproducible here** (capped==uncapped, ~2.0 ms/step). The cap is a verified-active safety belt bounding a cost that is ~0 on this build. The nav TASK is unaffected — obstacles are sensed geometrically (feelers) and the collision penalty is geometric time-in-contact (thorax within clearance of the surface), neither of which needs a physical contact. But physical *blocking* is absent here; since flygym is version-pinned + uv.lock committed, WIN should match — verify on WIN whether blocking/the 40 s pin actually occurs there.

## 3. AsyncVectorEnv determinism + throughput

- determinism (2 envs, same seed + action stream over 40 steps):
  - reset obs identical : True
  - max |Δ| (obs+reward) across all steps: 0.000e+00  (0 == bit-deterministic)
- throughput (6 async envs):
  - steps/s  :  103.9  (240 env-steps in 2.3 s)
  - resets/s :  14.29  (24 env-resets in 1.7 s)
- single-env cost split (the throughput ceiling driver):
  - reset (rebuild FlyEnv + warmup):   248.3 ms
  - step  (1 control = 40 physics) :     2.6 ms
  - reset == 97.1 steps' worth of wall time. RL burns far more episodes than CMA-ES, so if reset dominates we revisit (move the obstacle as a body vs re-baking geoms). Measured now so the ceiling is known before the full run.

## 4. Domain randomization variation + held-out disjointness

- train resets (12, seeded once then streamed):
  - bearing_deg : min=27.6 max=58.2 std=11.3 (range (20.0, 60.0))
  - obstacle x  : min=2.35 max=7.05
  - obstacle y  : min=3.98 max=8.14
  - radius      : min=1.68 max=2.33
  - distinct obstacle positions: 12/12  (DR visibly varies obstacle + bearing)
- held-out set (8 layouts, seed=20259):
  - repeatable across env instances: True
  - train/held-out disjointness: 0/5000 train draws fell within tolerance of a held-out layout  (DISJOINT)

## Verdict

**Gate 1: PASS** — random episodes run and truncate on the time limit; obs/action shapes + spaces correct; Newton cap verified applied obstacles-only (20/10) and obstacle-free envs untouched (100/50 defaults); worst-case rollout bounded (~2.0 s for 1000 steps); vector envs bit-deterministic (max|Δ|=0e+00); throughput 104 steps/s, reset 248 ms vs step 2.6 ms; DR varies + held-out disjoint (0 leaks).

> **Carried finding for the partner:** physical obstacle blocking is absent on this `flygym==2.0.1`/MAC build (obstacles are sensed-only); the cap defends a regime that doesn't trigger here. Verify on WIN.
