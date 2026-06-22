# `connectome/` — real FlyWire sub-circuits for embodied control

This directory holds **curated, self-contained connectome sub-circuits** pulled from FlyWire,
small enough to commit and reuse. They are the *data contract* the connectome-policy waves (CX-2+)
code against — no harness code lives here, just the wiring.

## `lc_dnp01_subcircuit.json` — the escape convergence (CX-1)

The real `LC4 + LPLC2 → DNp01` (Giant Fiber) escape circuit: the wiring that our hand-built
bilateral "loom front-end" (REPORT_x_a) stands in for. LC4 ≈ angular **velocity**, LPLC2 ≈ angular
**size/looming**; both are cholinergic (excitatory) and converge on the Giant Fiber descending
neuron, which sums size + velocity into the escape command toward the VNC.

### Schema

```
meta : {
  sources, flywire_release, query_date,
  edge_definition, edge_threshold_syn_count,        # how "connected" is defined (see below)
  counts_per_type,                                  # LC4/LPLC2/DNp01 totals + L/R
  all_edges_ipsilateral,                            # true: every VPN→GF edge is same-side
  convergence_per_hemisphere_syn_ge_1,              # (n presyn neurons, total synapses) per GF
  convergence_per_hemisphere_syn_ge_5,              # same, restricted to reliable (>=5 syn) edges
  convergence_bilateral_syn_ge_1
}
nodes : [ { id, type, hemisphere, neurotransmitter }, ... ]   # 316: 104 LC4 + 210 LPLC2 + 2 DNp01
edges : [ { pre_id, post_id, pre_type, pre_hemisphere, post_hemisphere,
            synapse_count, sign, neurotransmitter }, ... ]    # 293 per-neuron LC4/LPLC2 → GF edges
```

The `edges` list is **per presynaptic neuron** (not collapsed to an aggregate) and carries
`pre_hemisphere`/`post_hemisphere`, so CX-2 can build the policy fan-in directly from it and group
L/R onto the bilateral loom seam. `synapse_count` is summed over neuropils per `(pre, post)` pair;
threshold as needed (the `syn_ge_5` meta view is provided for the common reliable-connection cut).

## Provenance (exact, reproducible, credential-free)

Two **public, static, version-pinned** sources, both keyed to FlyWire materialization **`783`**
(snapshot Oct 2023 — the published whole-brain connectome). No login, token, or API credential was
used; nothing was scraped. Codex's download app (which requires Google sign-in) was deliberately
**not** used in favour of these static artifacts.

| Role | Source | Notes |
|---|---|---|
| Cell types, root IDs, hemisphere, NT | [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations) → `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | Schlegel et al. 2024; 139,244 neuron rows; root_ids pinned to 783 |
| Connectivity (syn counts, NT avgs, neuropil) | [Zenodo 10676866](https://zenodo.org/records/10676866) → `proofread_connections_783.feather` | Dorkenwald et al. 2024 (Nature); CC-BY-4.0; 16,847,997 connection rows; 852 MB |

`proofread_connections_783.feather` sha256: `24f960ae3e7d4f8cd30db3b62e99fb5179cc3d1e76d8c155bfb441e9737d3faf`

### How the cell types were matched

In `Supplemental_file1`, the `cell_type` column carries the labels `LC4`, `LPLC2`, `DNp01`
verbatim. Cross-checks held: LC4/LPLC2 have `super_class = visual_projection`; **DNp01 has
`super_class = descending` and `hemibrain_type = "Giant Fiber"`** — the GF identity confirmed
directly in the data, not assumed. `side` (left/right) gives hemisphere; `top_nt` gives the
predicted transmitter. All 104 LC4 and all 210 LPLC2 are `top_nt = acetylcholine` →
**cholinergic → excitatory**, which sets the `sign` on every edge.

### How "connected" is defined (read this before trusting counts)

`proofread_connections_783.feather` is **not thresholded**: the minimum `syn_count` over all
16.8M rows is **1**. So an edge in this file means *≥1 synapse*. FlyWire/Codex typically *display*
only connections with `syn_count ≥ 5` (single-synapse pairs are detection-noise-prone). The
artifact keeps **all** ≥1-synapse edges (each tagged with its `synapse_count`) and additionally
records the `≥5` aggregate in `meta`, so downstream code can pick the cut. The roadmap's literature
figures and our numbers are compared apples-to-apples in `ops/reports/REPORT_cx_1_extract.md`.

## Reproduce

```bash
# 1. cell-type annotations (32 MB, no auth)
curl -L -o Supplemental_file1_neuron_annotations_783.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

# 2. proofread connectivity (852 MB, no auth, CC-BY-4.0)
curl -L -o proofread_connections_783.feather \
  "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"

# 3. join + curate  (script + raw dumps are gitignored under scratch/cx + outputs/connectome_raw)
uv run --with pyarrow --with pandas python scratch/cx/extract_subcircuit.py
```

The join: select `cell_type ∈ {LC4, LPLC2, DNp01}` from the annotations → get root_ids + side +
NT; filter `proofread_connections` to rows where `post_pt_root_id ∈ DNp01` and
`pre_pt_root_id ∈ LC4 ∪ LPLC2`; sum `syn_count` per `(pre, post)`; tag sign from presynaptic NT.

### Judgement calls

- **Kept all ≥1-synapse edges** (rather than pre-thresholding at ≥5) so CX-2 owns the cut; the ≥5
  view is recorded alongside. Single-synapse LPLC2→GF[left] edges are a real chunk of the left-side
  count (94 → 37 neurons under ≥5), so the threshold materially changes fan-in — flagged, not hidden.
- **Hemisphere from `side`** (soma side). Every LC4/LPLC2→DNp01 edge turned out ipsilateral
  (`all_edges_ipsilateral = true`), so left-VPN→left-GF and right-VPN→right-GF are cleanly separable
  — which is exactly the bilateral seam our loom front-end already uses.
- **Sign = excitatory** is inferred from presynaptic transmitter (cholinergic), not from a per-edge
  sign field (FlyWire has none); the per-edge NT-average columns in the feather are consistent.

Raw dumps (the 852 MB feather, the 32 MB TSV, the flat edge CSV) live in **gitignored**
`outputs/connectome_raw/` and are **not** committed; only this curated JSON + README cross git.
