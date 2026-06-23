# Claude Code — EB-1 · The escape coupling: close the loop (looming → real brain → body), STOP at the interface decisions

> ## SAFETY (read first; bypassPermissions)
> Additive — the integration module; don't break the brain/body/neurons components or prior behavior.
> No `rm -rf` of anything you didn't create; scratch in `scratch/eb/`. `ops/` off-limits. **Commit on
> your branch before finishing** (don't merge). Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **Runs on the MAC** (brain laptop-runnable + MuJoCo — **no sentry**). On branch `feat/n-rl-navigation`
> (brain + body + neurons + CX-1 are all merged here). Use a todo list. **Read first:**
> `../portfolio/docs/cellular-gaits/EMBODIED_BRAIN_PLAN.md` (esp. the **design-decision log**) and the
> three component notes (now tracked): `docs/embodied/brain_explainer.md`, `docs/embodied/body_explainer.md`,
> `docs/embodied/neurons_report.md`. **Understanding-first + the crux:** this is where the real design
> decisions live. Implement sensible defaults, but **surface the interface choices for Vishal to decide**
> — do NOT silently bake in arbitrary constants.

## Carried findings from EB-0A/0B/0C — APPLY THESE (some are hard requirements, not suggestions)
1. **BUILD THE BRAIN ONCE, THEN `step()` — never rebuild per window.** EB-0B measured network
   construction from the ~100 MB connectivity table at **~7 min/build**; a loop that rebuilds each
   window is dead on arrival. Build once (`BrainModel.load`), advance with `brain.step(0.015)` (the
   network is persistent — voltages carry across calls), and make the looming **input rate
   runtime-settable** (swap the fixed `PoissonInput` for a `PoissonGroup`/`TimedArray` rate source) so
   intensity changes per window with **no rebuild**. **Verify a single step is milliseconds, not
   minutes, before running the full loop** — if it isn't, STOP and report; the loop is infeasible until
   this is solved.
2. **Brain step = EB-0B's helper.** Use `looming_to_giant_fiber(brain, lc4_hz, lplc2_hz) -> dnp01_rate`
   (`brain/neurons.py`); take **direction from the L vs R LC4/LPLC2 activation bias**.
3. **Body = EB-0C's primitive.** Use `apply_escape(env, drive, direction)`; keep one `EscapeMotor`
   across windows (carry its CA state). The motor map should target the **moderate** drive range
   (`CALIBRATED_DRIVE ≈ 0.2`), **not** linear-to-1 — "more DNp01 ≠ spin harder" (a held/saturated drive
   tumbles the body). Escape is a **transient**: pulse the drive as the loom rises/falls, don't hold DC.
   Expect a **left/right asymmetry** (stronger left turn, inherited from X-A) — report it, don't assume symmetry.
4. **HONESTY (carry it into the demo + report).** EB-0B found the **isolated GF saturates** (we drive all
   314 VPNs with no whole-brain inhibition/normalization). So this loop faithfully *wires the
   LC4/LPLC2→DNp01→escape pathway*, but do **NOT** claim a calibrated escape threshold or in-vivo
   selectivity — that lives in the whole-brain context this stack doesn't yet capture. The honest claim
   is "the real connectome routes a looming cue to an embodied escape," with that caveat stated plainly.

## Context
This is **Eon's four-part loop in miniature, on the real connectome** — and one step beyond Eon's public
demo (they wired looming→giant-fiber in the brain but never embodied escape). Wire the closed loop:
looming → LC4/LPLC2 activation → Shiu LIF brain → DNp01 rate → escape motor command → NeuroMechFly bolt
→ movement changes looming → repeat.

## Your slice — own `src/cellular_gaits/embodied/escape_loop.py`
1. **Close the loop**, reusing the components:
   - **Looming sensor:** the existing loom front-end (object angular size + expansion rate from the
     NeuroMechFly scene) → a looming magnitude per eye.
   - **SENSORY MAP** (design decision): looming magnitude → **LC4/LPLC2 activation Hz** (EB-0B helper).
   - **Brain step:** run the brain a sync-window → **DNp01 (giant fiber) firing rate** (EB-0B helper).
   - **MOTOR MAP** (design decision): DNp01 rate → an escape **drive** (+ direction from L/R loom) →
     EB-0C's `body.py` escape primitive.
   - **Sync rate** (design decision): brain↔body window (Eon uses 15 ms; escape is fast — pick + justify).
2. **Demonstrate it:** a looming stimulus, through the **real brain**, makes the body **escape**
   end-to-end. Record a short rollout (looming trace, DNp01 rate trace, body displacement).
3. **The three mappings (sensory Hz, DNp01→drive, sync rate) are the core DESIGN DECISIONS.** Implement
   defaults, **expose them as clearly-named tunable knobs**, and **report them with alternatives + the
   honest caveats** (hand-tuned coupling; sparse single-DN readout; activation-Hz is a modeling choice;
   not a proof of structure→behavior). Do not treat the constants as settled.

## Report → `scratch/eb/escape_loop_report.md`, then STOP
The working loop (with the rollout traces), the **three interface mappings you chose + why + the
alternatives**, the caveats, and the proposed next tuning. End literally: *"Confirm the interface
mappings (or adjust) and we'll tune / move to Phase 2."* **Do not** auto-tune further or start Phase 2.

## Definition of done
- `src/cellular_gaits/embodied/escape_loop.py` — the closed loop: looming → real connectome → body escape.
- The three mappings exposed as knobs + surfaced as decisions in the report.
- `scratch/eb/escape_loop_report.md` written (+ a tracked copy at `docs/embodied/`); component notes referenced.
- **Committed on `feat/n-rl-navigation`.** STOP for the interface-mapping review.
