# Claude Code — N-RL-1a · The RL env (Gymnasium wrapper + nav task)

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors **bit-exact**
> (`nca.py`, `env.py`, `evolve*.py`, `navigation.py`, `checkpoints/`, `outputs/`, `web_data*`).
> Additive only — append, don't rewrite. No `rm -rf` of anything you didn't create; scratch in ONE
> dir you make (`scratch/nrl/`). `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh` off-limits.
> **Commit on your branch before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **You are in a git worktree on branch `feat/n-rl-env`.** Run from the repo root. Use a todo list.
> **Read first:** `PROMPT_n_rl_navigation.md` (full design — you're building section "the nav task /
> env_base + nav_env"), `../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md`, and the existing code you
> wrap: `src/cellular_gaits/env.py` (`FlyEnv`, `ObstacleField`/`read_feelers`, `OdorField`/`read_chemo`,
> the **Newton-iteration contact cap** used for obstacle envs), `src/cellular_gaits/nca.py`
> (`build_nav_sensor_map`, the `nav` mode), and `src/cellular_gaits/evolve_navigation.py`
> (`NavConfig`, `conditions`, `nav_fitness` — the episode reward you convert to per-step).

## Your slice

Build the **RL environment layer** — nothing else (policy and PPO are parallel sessions). New deps +
two modules + the env-sanity gate.

1. **Deps:** add `gymnasium` to `pyproject.toml`, `uv sync`, commit `uv.lock`. (torch already present.)
2. **`src/cellular_gaits/rl/env_base.py` — `EmbodiedRLEnv(gymnasium.Env)`:** wraps `FlyEnv` as an
   episodic env with *pluggable* (i) observation builder, (ii) action→actuator map, (iii) reward fn,
   (iv) domain-randomization sampler called on `reset()`. Apply the **Newton≤20 / ls 10 contact cap
   ONLY when obstacles are present** (chemo/loom/closed-loop/v1 physics stay byte-exact).
3. **`src/cellular_gaits/rl/nav_env.py` — `NavRLEnv`:** the nav task on top of `EmbodiedRLEnv`.

### The contract (the other two sessions code against THIS — keep it exact)

- **Observation:** `Box(float32, shape=(6,8,8))` — the nav sensor grid from `build_nav_sensor_map`
  (ch0–1 proprio, ch2–3 odor goal-beacon, ch4–5 feelers), built every step. Document the layout.
- **Action:** `Box(float32, low=-1, high=1, shape=(N_motor,))` — the motor-cell vector the NCA emits;
  clamp to [-1,1]; map to the FlyGym actuators exactly as the current nav rollout does.
- **`reset(seed, options)`:** apply **domain randomization** — sample obstacle xy (continuous, on the
  homing path), obstacle radius, and **goal bearing** from configurable distributions. Support a
  **held-out** layout set (disjoint, fixed-seeded) selectable via `options={"eval": True}`.
- **`step(action)` → `(obs, reward, terminated, truncated, info)`:** reward = `Δapproach`
  `− w_collide·in_contact − step_cost (+ reach_bonus on arrival, gated on collision-free)`. `info`
  carries per-step diagnostics: `collision`, `dist_to_goal`, `reached`, `detour_perp`, `feeler_L`,
  `feeler_R`, `goal_bearing`. `terminated` on reach or fall; `truncated` on time limit.
- Make obs/reward/DR params live in a `NavRLConfig` dataclass (mirror `NavConfig` defaults so the
  task matches N-A's geometry: `dist=18`, `obs_r=2`, `feeler_range=6`, etc.).

## Gate you own — Gate 1 (env sanity), report in `scratch/nrl/gate1.md`

Random-policy episodes run end-to-end; obs/action/reward shapes + spaces correct; the Newton cap holds
worst-case (fly-pinned-on-obstacle) rollout to ~7 s not ~40 s (time it); `gymnasium.AsyncVectorEnv`
with N workers is deterministic given a seed and reports a throughput (steps/s) number. Domain
randomization visibly varies obstacle + bearing across resets; the held-out set is disjoint from train.

## Definition of done

- `rl/env_base.py` + `rl/nav_env.py` + `rl/__init__.py`; `gymnasium` in `pyproject.toml`/`uv.lock`.
- Gate 1 reported in `scratch/nrl/`. Prior behaviors + all existing tests untouched and bit-exact.
- **Committed on `feat/n-rl-env`** (not merged). Report: files added, the contract as implemented
  (any deviation from the spec above — flag loudly, the other two sessions depend on it), Gate 1 numbers.
