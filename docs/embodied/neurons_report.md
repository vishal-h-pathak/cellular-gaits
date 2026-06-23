# EB-0B — LC4/LPLC2 → DNp01 in the brain, and a brain-only looming→giant-fiber check

*Report material. Written for a smart non-specialist (and Vishal). EB-0B bridges the
**escape circuit** to the running connectome brain: CX-1 extracted the real
`LC4 + LPLC2 → DNp01` (Giant Fiber) convergence from FlyWire; here we confirm those
identified cells live in the brain's v783 neuron list and that the **real wiring
reproduces the convergence** — the brain half of the escape loop, validated in isolation,
before any body.*

Artifacts: `src/cellular_gaits/brain/neurons.py` (the module), `scratch/eb/check_looming_gf.py`
(the demo), `scratch/eb/neurons_check.json` (the numbers), `scratch/eb/resolve_ids.py` (the
ID cross-check). Builds on EB-0A's `BrainModel` and CX-1's `connectome/lc_dnp01_subcircuit.json`.

---

## 1. Where this sits

The embodied fly's four-part loop is **sense → brain → descending neurons → body**. The
*escape* version of that loop is the cleanest one biology hands us:

1. **Sense** — a predator looms; the image of it expands on the retina.
2. **Brain** — two populations of visual projection neurons report the loom: **LC4** and
   **LPLC2**. They converge onto one descending command neuron, **DNp01**, the
   **Giant Fiber (GF)**.
3. **Descending** — the GF carries the "escape now" command down to the VNC.
4. **Body** — the VNC fires the jump/takeoff (this is the part the body half, EB-0/X-A,
   already stands in for).

EB-0B is **step 2 for escape**: confirm LC4, LPLC2 and DNp01 are real, addressable cells in
the brain we built, and show that driving the looming inputs makes the Giant Fiber fire —
using *only* the real connectome, no body, no hand-built front-end.

---

## 2. The neurons — who they are, why these three

| cell type | what it encodes | where | transmitter | role in the loop |
|---|---|---|---|---|
| **LC4** | angular **velocity** of the looming object (how fast it grows) | lobula (visual) projection neuron | acetylcholine → **excitatory** | "it's coming fast" |
| **LPLC2** | angular **size / looming** (loom geometry, radial motion) | lobula/lobula-plate projection neuron | acetylcholine → **excitatory** | "it's getting big" |
| **DNp01** | the **escape command** (sums size + velocity) | descending neuron, a.k.a. **Giant Fiber** | — (read-out) | "escape now" → VNC |

This is textbook escape wiring: **von Reyn et al. 2014** showed the Giant Fiber sums
angular size and angular velocity to time a takeoff; **Ache et al. 2019** dissected the
LC4 (velocity) vs LPLC2 (size) division of labour converging on the GF. CX-1 found that exact
convergence in the FlyWire v783 connectome — all LC4/LPLC2 cholinergic, every VPN→GF edge
**ipsilateral**, DNp01 carrying `super_class = descending`, `hemibrain_type = "Giant Fiber"`
in the data itself. EB-0B is the test that the convergence *functions* when you actually run
the wiring.

---

## 3. Part 1 — the cell-type → FlyWire-ID mapping (and it all resolves)

CX-1's curated sub-circuit lists **316 identified neurons**. EB-0B's job was to confirm those
FlyWire IDs are present in the brain's own neuron list (`Completeness_783.csv`, 138,639
neurons) — i.e. that the escape cells are *addressable in the brain we run*, with no
release/version drift.

**Result: all 316 resolve (100%).** Both the CX-1 artifact and the brain's neuron list are
keyed to FlyWire materialization **v783**, so no reconciliation was needed — unlike EB-0A's
sugar example, whose IDs came from the older 630 release (20/21 resolved). Counts by type and
hemisphere:

| cell type | left | right | total |
|---|---:|---:|---:|
| **LC4** | 54 | 50 | **104** |
| **LPLC2** | 108 | 102 | **210** |
| **DNp01** (Giant Fiber) | 1 | 1 | **2** |
| **all** | 163 | 153 | **316** |

The module `brain/neurons.py` exposes these as `lc4_ids()`, `lplc2_ids()`, `dnp01_ids()`
(each takes `"left"`/`"right"`/`"both"`), plus `provenance()` (sources + counts) and
`resolve_in_brain(brain)` (the auditable resolution check above). Every convergence number is
carried straight from CX-1's curation of two public, version-pinned sources (annotations:
Schlegel et al. 2024; connectivity: Dorkenwald et al. 2024, Zenodo 10676866).

---

## 4. Part 2 — the brain-only looming → Giant Fiber check

**No body.** We drive the looming pathway (LC4 + LPLC2) with a Poisson input at a chosen rate
— the same activate/propagate/read interface EB-0A used for sugar→feeding — and read DNp01's
firing rate over a 1 s window. Each measurement is an independent single-shot trial
(`looming_to_giant_fiber` clears and rebuilds per call).

### 4a. Baseline → silent, as it must be

With no drive, **DNp01 fires 0.0 Hz** (both GFs). The brain is silent at rest; anything we see
below is caused by the looming input flowing through the real wiring.

### 4b. The response curve (drive LC4 = LPLC2 = N Hz, bilateral)

| input drive (Hz) | DNp01 rate (Hz) |
|---:|---:|
| 0 | 0.0 |
| 10 | 36.5 |
| 25 | 86.5 |
| 50 | 126.5 |
| 75 | 146.5 |
| 100 | 162.5 |
| 125 | 175.5 |
| 150 | 188.0 |
| 175 | 194.5 |
| 200 | 201.5 |
| 250 | 216.5 |
| 300 | 227.5 |

```
DNp01 (Hz)
 228 |                                              . . . . . . . . . . . o (300)
 200 |                              . . . . . o  o
 175 |                  . . . o  o  o
 150 |          . o  o
 125 |      o
 100 |    o
  86 |   o
  50 |
  36 | o
   0 o______________________________________________________________________
     0   25   50   75  100  125  150  175  200  250  300   input drive (Hz)
```

**The shape, reported honestly.** The brief flagged that DNp01, as a high-threshold *command*
neuron, might give a **sharp/thresholded** curve. In this **isolated** check it does the
opposite: the curve is a **high-gain saturating (compressive) curve** — it leaps off zero
(36.5 Hz output at only 10 Hz input), passes its own input rate, and then rolls over toward a
soft ceiling (~190 Hz by 150 Hz drive, only crawling to ~228 by 300 Hz). There is a hard
threshold at exactly one place — **0 input → 0 output** — but above that the GF is *easily*
driven, not reluctant.

That is itself the finding, and §6 takes it seriously: in isolation we inject Poisson spikes
directly into *all 314* converging VPNs at once with **no opposing inhibition and no
normalization**, so the ~1,885 converging excitatory synapses (CX-1: 805 LC4 + 1,080 LPLC2,
≥1-syn bilateral) pour straight onto the GF. The in-vivo "only fire for a genuine loom"
selectivity must therefore live in the **whole-brain context this check deliberately strips
away** (feedforward/feedback inhibition, the requirement for a real spatiotemporal looming
*match*, competition among descending neurons) — not in the LC4/LPLC2→GF convergence by
itself. The convergence's job is to *sum size + velocity and amplify*; that is exactly what we
see.

### 4c. Channel decomposition (which input matters), at 150 Hz drive

| driven | DNp01 rate (Hz) |
|---|---:|
| LC4 only | 111.5 |
| LPLC2 only | 156.0 |
| **both** | **187.5** |

Both channels alone already drive the GF strongly, and **LPLC2 > LC4** — consistent with the
connectome: LPLC2 brings ~210 neurons / ~1,080 synapses onto the GF vs LC4's ~104 / ~805, so
the size channel has the heavier hand. The combination is **strongly sub-additive**
(111.5 + 156 = 267.5 ≫ 187.5 observed): the GF is near saturation, summing size and velocity
with diminishing returns — the compressive "sum two cues, don't double-count" behaviour you'd
want from a command neuron.

### 4d. Specificity control (is it really LC4/LPLC2?) → yes, cleanly

Drive at 150 Hz **but silence LC4 + LPLC2** (the upstream "zero all their outgoing synapses"
knockout — they can spike but reach no one): **DNp01 → 0.0 Hz** on both GFs. So the giant-fiber
response is **entirely** routed through LC4/LPLC2; none of it is the Poisson drive leaking to
the GF by another path. The convergence is the cause.

### 4e. Per-hemisphere (the loom is bilateral), at 150 Hz drive

| Giant Fiber | rate (Hz) |
|---|---:|
| left GF | 172.0 |
| right GF | 191.0 |

Both GFs engage (a head-on loom recruits both sides), with **right slightly stronger** —
again matching the wiring: CX-1's right-hemisphere convergence carries more synapses
(LC4→GF 431 vs 374 left; LPLC2→GF 622 vs 458 left). The asymmetry is the real connectome's,
not an artifact.

---

## 5. How it squares with the biology

- **The convergence reproduces.** Activating the real LC4 + LPLC2 populations drives the real
  DNp01 from silent to high rate, through the real v783 wiring, with **no hand-built circuit**
  — the connectome alone carries looming → giant fiber. This is the escape analogue of EB-0A's
  sugar → MN9 proof.
- **Size + velocity summation (von Reyn 2014; Ache 2019).** Both LC4 (velocity) and LPLC2
  (size) channels independently drive the GF and combine sub-additively — the GF *sums* the two
  looming cues rather than needing both, exactly the published division of labour.
- **LPLC2-dominant, right-dominant — and both predicted by the wiring.** The functional
  asymmetries (size > velocity; right GF > left GF) fall straight out of CX-1's synapse counts.
  Structure predicts function here, which is the whole bet.
- **Bilateral seam intact.** Every VPN→GF edge is ipsilateral, so left and right escape
  channels stay cleanly separable — the same bilateral seam the shipped loom front-end uses.

---

## 6. Honest limitations (carry these)

- **Isolated ≠ whole-brain.** We drove the converging VPNs directly and read the GF; we did
  *not* reproduce the brain's selectivity. With no inhibition or normalization, the GF responds
  to almost any VPN drive — so this check validates the **convergence and its summation**, not
  the in-vivo escape *threshold*. The real "fire only for a genuine loom" decision lives in
  context we removed. **Do not read the curve as "the fly escapes at 10 Hz of looming."**
- **Activation-Hz is a modeling choice.** Poisson drive at N Hz stands in for "how hard the
  looming stimulus pushes LC4/LPLC2." The magnitude→Hz mapping is hand-chosen, not measured
  (EB-0A's standing caveat); the curve's *shape* is meaningful, its *x-axis units* are a knob.
- **LIF is a cartoon neuron** and the **wiring is a frozen, predicted-NT snapshot** — EB-0A's
  limitations carry over unchanged (no plasticity, synapse-count weights, predicted signs).
- **Single trial per point.** Each curve point is one 1 s window of one live network (the
  upstream paper averages 30 trials). The trends are large and monotone, but treat the
  second-decimal precision as indicative, not statistical.
- **Cost note.** Each measurement rebuilds the network from the 100 MB connectivity table
  (~7 min/build; the full demo ran ~2.4 h for 18 builds). For EB-1's closed loop this must be
  built once and *stepped* (EB-0A's persistent-network path), not rebuilt per window.

---

## 7. What EB-1 gets (the contract)

`brain/neurons.py` exposes, for EB-1's brain step:

- `lc4_ids(h)`, `lplc2_ids(h)`, `dnp01_ids(h)` — the resolved FlyWire ID sets by hemisphere
  (`"left"`/`"right"`/`"both"`), plus eager `LC4_IDS`/`LPLC2_IDS`/`DNP01_IDS` dicts and
  `provenance()`.
- `looming_to_giant_fiber(brain, lc4_hz, lplc2_hz, *, hemisphere="both", duration_s=1.0)
  → dnp01_rate` — drive the looming pathway, read the Giant Fiber's mean rate. The brain half
  of the escape loop as one call. (It does a clean single-shot measurement; EB-1's persistent
  closed loop should drive LC4/LPLC2 and read `dnp01_ids()` rates from `brain.step()` directly,
  per the rebuild-vs-step caveat above.)

**Bottom line:** the escape circuit's brain half is real and addressable — all 316
LC4/LPLC2/DNp01 cells resolve in the v783 brain, and the real connectome turns looming input
into a Giant Fiber command (silent → ~190 Hz), driven specifically through LC4/LPLC2, summing
size and velocity sub-additively, with left/right and channel asymmetries that match the
wiring. The in-vivo escape *threshold* is explicitly out of scope here and belongs to the
whole-brain context EB-1 will start to close.
