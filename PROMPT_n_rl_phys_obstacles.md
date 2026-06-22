# Claude Code — N-RL-PHYS · Make obstacles physically block the fly (real MuJoCo contact pairs)

> ## SAFETY (read first; bypassPermissions)
> This is a **sanctioned, surgical modification** to the obstacle code in `src/cellular_gaits/env.py`
> — specifically `_add_obstacles`, the obstacle constants/comments, and the obstacle collision metric.
> That is the ONLY existing file you may modify, and only those parts. **No-obstacle physics
> (chemo / loom / closed-loop / v1) must stay byte-for-byte unchanged** (the pairs + solver settings
> apply ONLY when obstacles are present). Do not touch `nca.py`, the evolvers, or any prior behavior.
> No `rm -rf`; scratch in ONE dir you make (`scratch/nrlphys/`). Other `PROMPT_*.md`, `setup-*.sh`,
> `.worktrees/` off-limits. **Commit on `feat/n-rl-navigation` before finishing** (don't merge).
> (Full: `AGENT_SAFETY.md`)

> **Run on `feat/n-rl-navigation` (after the three wave-1 branches are merged), from the repo root,
> on the Mac.** Use a todo list. **Read first:** `PROMPT_n_rl_navigation.md`, the agent finding below,
> `src/cellular_gaits/env.py` (`_add_obstacles`, `ObstacleField`, `min_surface_distance`, the
> `OBSTACLE_*` constants + the Newton/CG solver block, and the rollout code that builds
> `traj["collision_count"]`), and **FlyGym's contact-pair setup** in the installed package
> (`flygym/.../world.py` `_set_ground_contact`, `ContactBodiesPreset`) — that is the pattern you mirror.

## Why (the finding you're fixing)

A code investigation proved the obstacles are currently **sensed-only, not physical**: FlyGym builds
every fly geom with `contype=0/conaffinity=0` and enables collisions ONLY via explicit `<pair>`
elements (fly-segment ↔ ground). `_add_obstacles` adds a cylinder with **no pair**, so MuJoCo never
generates a fly↔obstacle contact — the fly walks through. The N-A `collision_count` was a **geometric**
thorax-to-surface distance test, and the "fly pinned → Newton cap → 40s→7s" comments describe physics
that never happened. Vishal chose to make obstacles **truly physical**. Just setting the obstacle's
contype/conaffinity to 1 does nothing (the fly stays 0/0) — you must add explicit contact pairs.

## The change

1. **Add fly-geom ↔ obstacle `<pair>` elements** in/after `_add_obstacles`, mirroring
   `world.py::_set_ground_contact`: for each obstacle cylinder geom, add
   `mjcf_root.contact.add("pair", geom1=<fly contact geom>, geom2=<obstacle geom>, ...)` for the fly
   contact-body set used for ground (`ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD` — legs, thorax,
   abdomen, head). Tune `solref`/`solimp`/`friction`/`margin` for a **stable solid post** at FlyGym's
   physics rate; pick values that block without exploding (document them).
2. **Keep the Newton/CG solver cap** — it is now genuinely needed (a pinned fly is real). Confirm it
   stays applied **obstacles-only**.
3. **Switch the collision metric to real contacts:** count control steps with an active fly↔obstacle
   contact (via `physics.data.ncon` + contact geom ids), not the geometric clearance test. Keep the
   reach bonus **gated on collision-free**. (You may keep `min_surface_distance` only as a secondary
   shaping term if useful, but the penalty/▢ should reflect real contact now.)
4. **Correct the now-accurate comments** in `env.py` so they describe the pair-based blocking truthfully,
   and **add an honest note to `REPORT_n_a_calibration.md`**: N-A's obstacles were NOT physical, so its
   collision counts were geometric and its pin/Newton rationale did not apply to that run. (Radical
   honesty about the prior record — don't quietly overwrite it.)

## VALIDATION-FIRST gates → `scratch/nrlphys/REPORT_phys.md`, then STOP

1. **Blocking is real:** drive the fly straight into an on-path obstacle with a simple forward
   actuation (no trained controller needed) — it physically **stops/deflects and does NOT pass through**;
   `data.ncon` shows fly↔obstacle contacts; final thorax x is short of / around the post, never beyond it.
2. **No tunneling:** sweep approach speed and obstacle radius — the thorax never ends up on the far side
   without a contact (no clip-through at the physics/control rates). Report the worst case tried.
3. **Cost is bounded:** worst-case pinned-candidate rollout time is bounded by the solver cap (report
   capped vs uncapped seconds; confirm obstacles-only). Also report per-reset vs per-step cost so we know
   the RL throughput ceiling on the physical task.
4. **A/B byte-exact:** a no-obstacle rollout is unchanged vs current `main`-line physics (max|Δ| = 0);
   chemo/loom/closed-loop/v1 untouched; existing artifact-free tests pass.

End the report with: the contact-pair config + tuned params, the four gate results, and a recommendation
on episode length / reset cost given the new physical-rollout timings. Then: *"Confirm and the integrate
session can wire RL against the physical task."* **Do not run RL here.**

## Definition of done

- `_add_obstacles` adds real contact pairs; collision metric uses real contacts; Newton cap retained
  (obstacles-only); comments corrected; `REPORT_n_a_calibration.md` honesty note added.
- The four physics gates reported in `scratch/nrlphys/`. No-obstacle physics byte-exact; prior behaviors
  untouched.
- **Committed on `feat/n-rl-navigation`** (not merged).
