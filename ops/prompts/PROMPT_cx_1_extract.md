# Claude Code — CX-1 · Extract the real LC4/LPLC2→DNp01 escape sub-circuit from FlyWire (DATA ONLY)

> ## SAFETY (read first; bypassPermissions)
> Never delete or modify files you didn't create. Keep v1 + ALL prior behaviors **bit-exact** —
> this wave is **purely additive data work**; do **not** touch `nca.py`, `evolve.py`, `env.py`, the
> `rl/` harness, or anything under `outputs/`/`checkpoints/` you didn't generate. No `rm -rf` of
> anything you didn't create; do all scratch in ONE dir you make (`scratch/cx/`). `ops/` (other
> prompts/waves) off-limits. **Commit on your branch before finishing** (don't merge). Unsure →
> leave it + note it. (`AGENT_SAFETY.md`)

> **Run on branch `feat/cx-connectome`** (a worktree off `feat/n-rl-navigation`, so the `rl/`
> harness is present for continuity — but you do **not** use or modify it this wave). Run from the
> repo root. Use a todo list. Needs network access (FlyWire/Codex).
> **Read first:** `../portfolio/docs/cellular-gaits/research-roadmap.md` (the **"escape circuit"**
> section — the biology + the literature numbers you will sanity-check against), `ops/reports/REPORT_x_a.md`
> (how our **hand-built** loom front-end is structured — LPLC2-like size channel + LC4-like velocity
> channel, bilateral; the real circuit must map onto that same seam), and
> `../portfolio/docs/cellular-gaits/PARTNER_BRIEF.md` (taste + the visualization rule).

## Why this exists (the endgame, stated plainly)
This is **wave 1 of the connectome endgame** — the Tier-1, Eon-aligned milestone. The plan: replace
escape's hand-built loom front-end with the **real FlyWire `LC4/LPLC2 → DNp01` wiring**, then train
the descending→motor readout with RL on the existing `rl/` harness (CX-2 builds the `ConnectomePolicy`,
CX-3 trains on the 3080 Ti, CX-4 ships it into the escape tab's already-drawn seam). **This wave does
not build or train anything** — it extracts and curates the sub-circuit into a self-contained,
reusable artifact, and stops. Get the data right and honest; everything downstream depends on it.

## The circuit (what to pull)
The cleanest known escape wiring (see the roadmap for refs — Ache et al. 2019; von Reyn et al. 2017):
- **LC4** — lobula columnar visual projection neurons encoding angular **velocity** (≈ linear).
- **LPLC2** — lobula plate/lobula columnar neurons encoding angular **size** (≈ Gaussian / looming).
- **DNp01** — the **Giant Fiber** descending neuron; LC4 + LPLC2 converge onto its lateral dendrite
  (the GF effectively **sums size + velocity**). This is the command output toward the VNC.

## Your slice — extract → curate → sanity-check → report, then STOP

1. **Establish data access (VALIDATION GATE — do this first, then STOP if blocked).** Identify a
   public FlyWire source for the connectome + cell-type annotations (e.g. the Codex public download
   tables — neurons / connections / cell-type classification for a named release, or the
   `caveclient`/`fafbseg` Python path). Pull a **tiny sample** (e.g. the neuron rows for one LC4 and
   one DNp01, and a handful of their synapses) and confirm the schema (column names, ID type, how
   cell types are labelled, whether synapse counts and neurotransmitter/sign are present). **If access
   needs a credential/token you don't have, or the cell-type labels are ambiguous, STOP and report
   what you found + exactly what you need** — do not guess IDs or scrape around restrictions. Record
   the **release version** (e.g. the FlyWire materialization / Codex version) — provenance is
   non-negotiable.

2. **Identify the neurons.** Resolve the population of each type — **LC4**, **LPLC2**, **DNp01** —
   with their root IDs, hemisphere (L/R), and counts. Note the cell-type label/source you matched on
   (these are well-annotated types; DNp01 is the Giant Fiber / "GF").

3. **Extract the connectivity.** The key convergence: synapse counts **LC4→DNp01** and **LPLC2→DNp01**
   (per neuron and aggregate, bilateral), with **sign/neurotransmitter** where available (cholinergic →
   excitatory). Optionally capture LC4/LPLC2 **inputs** (what feeds them) and DNp01 **outputs**
   (downstream toward the VNC) for the eventual visual — but the convergence is the must-have.

4. **Curate a self-contained artifact** (this is the contract CX-2 codes against):
   - `src/cellular_gaits/connectome/lc_dnp01_subcircuit.json` (committed, small) — a clean graph:
     `nodes` (id, type, hemisphere, neurotransmitter) and `edges` (pre_id, post_id, synapse_count,
     sign), plus a `meta` block (source, release version, query date, counts per type, and the
     aggregate LC4→DNp01 / LPLC2→DNp01 synapse totals).
   - `src/cellular_gaits/connectome/README.md` — provenance: exact source + version, how you matched
     the types, the queries/commands to reproduce it, and any judgement calls.
   - Raw dumps (large CSVs, if any) → `outputs/connectome_raw/` (**gitignored**; do not commit).

5. **Literature sanity-check (honesty gate).** Compare your aggregate convergence to the published
   picture (the roadmap cites ≈55 LC4 + ≈108 LPLC2 synapses onto the GF lateral dendrite). State
   whether FlyWire matches, and **if it diverges, say so plainly with the numbers** — do not fudge
   toward the paper. This calibration is the brand.

## The report → `ops/reports/REPORT_cx_1_extract.md`, then STOP
End with: the data source + release version; neuron counts per type per hemisphere; the
LC4→DNp01 / LPLC2→DNp01 convergence (aggregate + the literature cross-check); the artifact paths;
and a **proposed `ConnectomePolicy` contract for CX-2** — i.e. the input dimensionality the real
circuit implies (how many LC4 + LPLC2 units feed in, how they group L/R to map onto our bilateral
loom seam), what the DNp01 layer is, and where the **learned readout** to the 42-actuator motor block
attaches. Flag the open architecture forks for chat (rate vs spiking/LIF units; how much is fixed real
wiring vs learned readout). Then literally: *"Confirm the sub-circuit + the proposed policy contract
and I'll green-light CX-2 (the ConnectomePolicy)."* **Do not build the policy.**

## Definition of done
- `lc_dnp01_subcircuit.json` + `connectome/README.md` (provenance) committed; raw dumps gitignored.
- `ops/reports/REPORT_cx_1_extract.md` with counts, convergence, literature cross-check, and the
  proposed CX-2 policy contract.
- v1 + the `rl/` harness + all prior behaviors **untouched**.
- **Committed on `feat/cx-connectome`** (not merged). Push so the cockpit can pull the artifact + report.
