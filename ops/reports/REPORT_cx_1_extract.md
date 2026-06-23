# REPORT · CX-1 — Extract the real LC4/LPLC2 → DNp01 escape sub-circuit from FlyWire (DATA ONLY)

**Branch:** `feat/cx-connectome` · **Date:** 2026-06-22 · **Scope:** data extraction + curation only —
no policy built, no training, `rl/` harness and all prior behaviors untouched.

This is wave 1 of the connectome endgame: pull the *actual* `LC4 + LPLC2 → DNp01` (Giant Fiber)
escape wiring that our hand-built bilateral loom front-end (REPORT_x_a) stands in for, curate it into
a committed contract, sanity-check it honestly against the literature, and stop. CX-2 builds the
`ConnectomePolicy` against this artifact; CX-3 trains it; CX-4 ships it into the escape tab's seam.

---

## 1. Data source + release version

Two **public, static, credential-free** artifacts, both pinned to FlyWire materialization **`783`**
(snapshot Oct 2023 — the published whole-brain connectome). No login/token used; nothing scraped.
Codex's download app (Google sign-in required) was deliberately avoided in favour of these static,
version-pinned sources — more reproducible, zero credential surface.

| Role | Source | Scale |
|---|---|---|
| Cell types, root IDs, hemisphere, NT | [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations) → `Supplemental_file1_neuron_annotations.tsv` (Schlegel et al. 2024) | 139,244 neurons |
| Connectivity (syn_count, NT avgs, neuropil) | [Zenodo 10676866](https://zenodo.org/records/10676866) → `proofread_connections_783.feather` (Dorkenwald et al. 2024, *Nature*; CC-BY-4.0) | 16,847,997 connections |

Connections file sha256 `24f960ae…d3faf`. Type labels are matched on the `cell_type` column;
**DNp01 cross-checks to `super_class = descending` + `hemibrain_type = "Giant Fiber"`** — GF identity
confirmed in the data. LC4/LPLC2 are `super_class = visual_projection`.

---

## 2. Neuron counts per type, per hemisphere

| type | role | total | left | right |
|---|---|---|---|---|
| **LC4** | angular **velocity** VPN | 104 | 54 | 50 |
| **LPLC2** | angular **size / looming** VPN | 210 | 108 | 102 |
| **DNp01** | Giant Fiber descending command | **2** | 1 | 1 |

DNp01 is exactly the **bilateral Giant Fiber pair** (`left = 720575940622838154`,
`right = 720575940632499757`). All LC4/LPLC2 and both GFs are **cholinergic (`top_nt =
acetylcholine`) → excitatory**, consistent with LC4+LPLC2 exciting the GF.

---

## 3. The convergence (the must-have result)

**Every** LC4/LPLC2 → DNp01 edge is **ipsilateral** (`all_edges_ipsilateral = true`): left VPNs
drive the left GF, right VPNs the right GF, with no contralateral crossing. This is the clean
bilateral seam our loom front-end already assumes.

Per-GF convergence — `(# presynaptic neurons, total synapses)`, summed over neuropils per pair:

| edge | syn ≥ 1 (file default) | syn ≥ 5 (reliable-connection view) |
|---|---|---|
| LC4 → DNp01[left]  | 54 neurons, 374 syn | 40 neurons, 329 syn |
| LC4 → DNp01[right] | 50 neurons, 431 syn | 49 neurons, 428 syn |
| LPLC2 → DNp01[left]  | 94 neurons, 458 syn | 37 neurons, 305 syn |
| LPLC2 → DNp01[right] | 95 neurons, 622 syn | 62 neurons, 539 syn |

Bilateral synapse totals (syn ≥ 1): **LC4 → both GFs = 805**, **LPLC2 → both GFs = 1080**.
Of 210 LPLC2, 189 connect to a GF; all 104 LC4 connect.

**Threshold matters (recorded, not hidden).** `proofread_connections_783.feather` is **not
thresholded** — minimum `syn_count` over all 16.8M rows is **1**. FlyWire/Codex normally *display*
only `syn_count ≥ 5`. The artifact keeps all ≥1-synapse edges (each tagged with its count) and also
stores the ≥5 aggregate. The cut materially changes fan-in, especially **LPLC2 → GF[left]
(94 → 37 neurons)** — a real left/right asymmetry in weak connections, flagged for CX-2 to decide.

---

## 4. Literature sanity-check (honesty gate — cross-dataset, numbers as they fall)

The roadmap cites **≈55 LC4 + ≈108 LPLC2 synapses** onto the GF lateral dendrite (single
hemisphere). Comparing to FlyWire's **per-GF, single-hemisphere** figures (right hemisphere; ≥1 syn):

| quantity | roadmap figure | FlyWire 783 (right GF) | match? |
|---|---|---|---|
| LC4 as **presynaptic-neuron** count | ≈55 | **50** | ✅ close |
| LPLC2 as **presynaptic-neuron** count | ≈108 | **95** | ✅ close |
| LC4 as **synapse** count | ≈55 | **431** | ✗ ~8× higher |
| LPLC2 as **synapse** count | ≈108 | **622** | ✗ ~6× higher |

**Read it straight:** the roadmap's "≈55 / ≈108" line up almost exactly with FlyWire's *presynaptic
neuron counts per hemisphere*, **not** with synapse counts — i.e. the figures are most consistent
with *number of LC4/LPLC2 cells contacting the GF*, despite the word "synapses." Taken literally as
synapse counts, FlyWire is several-fold higher. This is a **cross-dataset** comparison (the roadmap
number traces to the earlier hemibrain/EM escape literature; FlyWire 783 is an independent
whole-brain reconstruction with denser synapse detection), so some divergence is expected. The
neuron-count agreement is strong; the synapse-count gap is real and stated, not fudged toward the
paper.

---

## 5. Artifact paths (the contract CX-2 codes against)

| path | committed? | what |
|---|---|---|
| `src/cellular_gaits/connectome/lc_dnp01_subcircuit.json` | ✅ yes (123 KB) | 316 nodes + 293 per-neuron edges + meta (both threshold views) |
| `src/cellular_gaits/connectome/README.md` | ✅ yes | provenance, schema, reproduce commands, judgement calls |
| `outputs/connectome_raw/proofread_connections_783.feather` | ❌ gitignored | 852 MB raw connectivity |
| `outputs/connectome_raw/Supplemental_file1_neuron_annotations_783.tsv` | ❌ gitignored | 32 MB raw annotations |
| `outputs/connectome_raw/lc_lplc2_to_dnp01_edges_783.csv` | ❌ gitignored | flat edge dump |
| `scratch/cx/extract_subcircuit.py` | ❌ gitignored | the join script (reproducible) |

Edges are **per presynaptic neuron**, each carrying `pre_hemisphere`/`post_hemisphere` and
`synapse_count` — the fan-in structure is preserved, not collapsed, so CX-2 builds directly from it.

---

## 6. Proposed `ConnectomePolicy` contract for CX-2

**Input dimensionality the real fan-in implies** (presynaptic VPNs per GF):

| | LC4 | LPLC2 | total in / GF |
|---|---|---|---|
| GF[left]  (syn ≥ 1) | 54 | 94 | **148** |
| GF[right] (syn ≥ 1) | 50 | 95 | **145** |
| GF[left]  (syn ≥ 5) | 40 | 37 | 77 |
| GF[right] (syn ≥ 5) | 49 | 62 | 111 |

Bilateral input layer = **293 VPN units** (104 LC4 + 189 connected LPLC2) at the ≥1 cut, or 188 at
≥5. (Which cut is fork #3 below.)

**Proposed architecture (for confirmation, not yet built):**

```
loom stimulus (θ, dθ/dt, bearing φ)              ← same front-end geometry as REPORT_x_a
        │   [input encoder → VPN activations]     ← FORK #4: fixed tuning map vs learned encoder
        ▼
  293 VPN units  ── grouped 4 ways ──┐            LC4_L, LPLC2_L → GF_left
   (LC4/LPLC2 × L/R)                 │            LC4_R, LPLC2_R → GF_right   (all ipsilateral)
        │  VPN→GF weights from synapse_count       ← FORK #2: frozen real wiring vs learned
        ▼
  DNp01 layer = 2 GF units (left, right), each = Σ ipsilateral VPN · weight
        │   [learned descending → motor readout]   ← THE LEARNED PART
        ▼
  42-actuator motor block  (the existing 7×6 motor grid; readout attaches exactly where the two
                            bilateral loom channels feed conv1 today — the seam is already drawn)
```

- **L/R grouping onto the bilateral loom seam:** the four VPN groups (LC4/LPLC2 × left/right) map
  cleanly onto the existing `loom_L`/`loom_R` channels — LC4 carries the velocity term, LPLC2 the
  size term, exactly the two functional axes the hand-built front-end split by side. The real
  circuit replaces *2 hand-built channels* with *293 real units feeding 2 GFs*, same seam.
- **DNp01 layer:** 2 units (the GF pair), each summing its ipsilateral LC4+LPLC2 input weighted by
  `synapse_count`. The GF "sums size + velocity," matching the biology.
- **Where the learned readout attaches:** descending `DNp01(2) → 42 actuators`. The fixed real
  wiring is VPN→GF (synapse counts); the learned block is GF→motor (and optionally the input
  encoder). This keeps A/B integrity with X-A's seam.

### Open architecture forks — flagged for you, NOT decided here

1. **Rate vs spiking (LIF).** Rate units (tanh/ReLU) drop straight into the existing conv/CMA/PPO
   stack. LIF spiking units capture the *single-GF-spike-timing → short vs long takeoff* that the
   escape biology (von Reyn 2017) centers on — but need a spiking sim + surrogate-gradient/other
   training. Bigger lift, more biologically faithful.
2. **Fixed wiring vs learned readout — how much is real.** (a) freeze VPN→GF at synapse-count
   weights, learn only GF→motor; (b) initialize VPN→GF from synapse counts, let RL fine-tune;
   (c) use the connectome for topology only (who connects whom), learn all weights. Trades
   "real connectome drives the body" purity against learnability.
3. **Connection threshold (≥1 vs ≥5 synapses).** Sets input width (148/145 vs 77/111 per GF) and
   changes the L/R LPLC2 balance noticeably. ≥5 is the FlyWire reliability convention; ≥1 is the
   raw file. Affects fan-in and the left/right asymmetry the policy inherits.
4. **Input encoding.** How the loom stimulus drives the 293 VPNs — a fixed retinotopic/tuning map
   (LC4↦velocity, LPLC2↦size, retinotopy by bearing) vs a small learned encoder. Determines how
   much of the "front-end" stays hand-built vs becomes circuit.

---

## Definition of done — status

- ✅ `lc_dnp01_subcircuit.json` + `connectome/README.md` (provenance) committed; raw dumps gitignored.
- ✅ This report: counts, convergence, honest literature cross-check, proposed CX-2 contract.
- ✅ v1 + `rl/` harness + all prior behaviors untouched (no edits outside `connectome/`, `ops/reports/`,
  gitignored `outputs/`+`scratch/`).
- ⏳ Committed on `feat/cx-connectome` (not merged) + pushed — on completion of this report.

---

**Confirm the sub-circuit + the proposed policy contract and I'll green-light CX-2 (the
ConnectomePolicy).**
