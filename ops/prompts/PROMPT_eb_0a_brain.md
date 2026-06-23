# Claude Code — EB-0A · Stand up the FlyWire connectome LIF brain (Shiu model) + a clean Python API

> ## SAFETY (read first; bypassPermissions)
> Additive only — this is a **new** module; do not modify or break `nca.py`/`env.py`/`evolve*.py`/`rl/`
> or any prior behavior. No `rm -rf` of anything you didn't create; scratch in `scratch/eb/`. `ops/`
> off-limits. **Commit on your branch before finishing** (don't merge). Heavy data files **gitignored**.
> Unsure → leave it + note it. (`AGENT_SAFETY.md`)

> **Runs on the MAC** (the Shiu LIF brain is laptop-runnable Brian2 — **no sentry**). On branch
> `feat/eb-brain`. Use a todo list. **Read first:** `../portfolio/docs/cellular-gaits/EMBODIED_BRAIN_PLAN.md`
> (the refocus + architecture). **Understanding-first:** your job includes a plain-English explainer
> note — this is report material, not an afterthought.

## Context (the refocus)
We are recreating Eon's embodied fly: a real FlyWire connectome **brain** driving the NeuroMechFly
**body**. This task builds the **brain layer** — wrap the published Shiu model so the rest of the
project can use it. **You do not reimplement the LIF**; you vendor the published model and expose a
clean API.

## Your slice — own `src/cellular_gaits/brain/`
1. **Vendor the Shiu model** ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model),
   MIT). Bring in `model.py` + `utils.py` + the **v783** data (`Completeness_783.csv` ~3.2 MB,
   `Connectivity_783.parquet` ~97 MB) under a clear vendored path (e.g.
   `src/cellular_gaits/brain/_shiu/`). **Gitignore** the 97 MB parquet + any generated weight pickles;
   the 3.2 MB CSV may be committed. Add a tiny `fetch_data.sh`/README so the parquet is reproducibly
   downloadable (it's in the public repo). Keep the upstream MIT LICENSE/attribution.
2. **Get Brian2 running** in the env (`uv add brian2`; for speed follow Brian2's C++ codegen — Mac
   needs Xcode Command Line Tools). Confirm a trivial Brian2 sim runs.
3. **Expose a clean API** in `src/cellular_gaits/brain/` — a thin wrapper over the vendored model, e.g.
   `BrainModel.load(release="783")`, `.activate(flywire_ids, hz)`, `.silence(flywire_ids)`,
   `.run(duration_s) -> {flywire_id: rate_hz}` (+ spike times if cheap). Match the model's existing
   activate/silence semantics (Poisson activation; silencing zeros connections). Don't change the LIF.
4. **Validation — reproduce a KNOWN result.** Following the upstream `example.ipynb`, activate a
   documented sensory set (e.g. **sugar gustatory receptor neurons**) and confirm a **known downstream
   circuit** (the feeding pathway) lights up. Report the activated/observed neurons + rates. This is
   the proof the brain is wired correctly.
5. **Explainer note** → `scratch/eb/brain_explainer.md` (report material): plain-English "how the
   connectome LIF brain works" — each neuron = LIF unit, connectome = wiring (synapse count = weight),
   neurotransmitter prediction = sign, activate→propagate→read-rates loop. Written so a smart
   non-specialist (and Vishal) gets the intuition.

## Definition of done
- `src/cellular_gaits/brain/` API + the vendored model (heavy parquet gitignored + fetch note).
- The known-result validation passes and is reported.
- `scratch/eb/brain_explainer.md` written.
- Prior behaviors untouched; **committed on `feat/eb-brain`** (not merged). STOP and report:
  the API surface, the validation result, env notes (Brian2), and anything that surprised you.
