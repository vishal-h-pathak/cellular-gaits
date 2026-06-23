"""Named escape-circuit neurons in the brain's v783 list — LC4, LPLC2, DNp01.

EB-0B. This module bridges the **escape circuit** to the running connectome
brain. CX-1 (``connectome/lc_dnp01_subcircuit.json``) extracted the real FlyWire
``LC4 + LPLC2 -> DNp01`` (Giant Fiber) convergence; here we (1) confirm those
identified cells resolve in the brain's v783 neuron list and expose them as
ID sets per type and hemisphere, and (2) provide the contract helper EB-1 uses
as its brain step:

    from cellular_gaits.brain import BrainModel, looming_to_giant_fiber
    brain = BrainModel.load("783")
    rate = looming_to_giant_fiber(brain, lc4_hz=150, lplc2_hz=150)   # -> DNp01 Hz

Biology this stands on (see ``scratch/eb/neurons_report.md``)
------------------------------------------------------------
LC4 and LPLC2 are visual projection neurons of the lobula: **LC4 ~ angular
velocity**, **LPLC2 ~ angular size / looming**. Both are cholinergic
(excitatory) and converge ipsilaterally onto **DNp01**, the **Giant Fiber**
descending neuron, which sums size + velocity into the escape/takeoff command
toward the VNC (von Reyn et al. 2014; Ache et al. 2019; Dorkenwald/Schlegel
2024 connectome). DNp01 is a high-threshold *command* neuron: it is built to
fire for a genuine looming match, so the input-rate -> DNp01-rate relationship
is expected to be sharp/thresholded, not a gentle line.

What is faithful vs. what this module adds
------------------------------------------
- The neuron identities, hemisphere, and convergence are CX-1's curation of the
  public FlyWire v783 connectome (annotations: Schlegel et al. 2024; connectivity:
  Dorkenwald et al. 2024, Zenodo 10676866). Unchanged here.
- This module only: loads that curated artifact, resolves the IDs against the
  brain's ``Completeness_783.csv`` neuron list, and wraps ``BrainModel``'s
  activate/run into a single-shot looming->GF measurement. No LIF dynamics here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .brain_model import BrainModel

_HERE = Path(__file__).resolve().parent
# The CX-1 curated sub-circuit (committed; ~316 nodes).
_ARTIFACT = _HERE.parent / "connectome" / "lc_dnp01_subcircuit.json"

_HEMIS = ("left", "right")
_TYPES = ("LC4", "LPLC2", "DNp01")


# --------------------------------------------------------------------- loading
@lru_cache(maxsize=1)
def _load_artifact() -> dict:
    if not _ARTIFACT.is_file():
        raise FileNotFoundError(
            f"missing CX-1 sub-circuit artifact {_ARTIFACT}. It is committed on "
            f"the connectome branch; see connectome/README.md to reproduce.")
    with open(_ARTIFACT) as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _ids_by_type_hemi() -> dict[str, dict[str, tuple[int, ...]]]:
    """``{type: {"left": (...), "right": (...), "both": (...)}}`` from the artifact."""
    art = _load_artifact()
    out: dict[str, dict[str, list[int]]] = {
        t: {"left": [], "right": []} for t in _TYPES}
    for node in art["nodes"]:
        t, h = node["type"], node["hemisphere"]
        if t in out and h in out[t]:
            out[t][h].append(int(node["id"]))
    # freeze + add a "both" convenience key (left then right, deduped, ordered)
    frozen: dict[str, dict[str, tuple[int, ...]]] = {}
    for t, hemi in out.items():
        both: list[int] = []
        seen: set[int] = set()
        for h in _HEMIS:
            for i in hemi[h]:
                if i not in seen:
                    seen.add(i)
                    both.append(i)
        frozen[t] = {h: tuple(hemi[h]) for h in _HEMIS}
        frozen[t]["both"] = tuple(both)
    return frozen


def _ids(cell_type: str, hemisphere: str = "both") -> tuple[int, ...]:
    table = _ids_by_type_hemi()
    if cell_type not in table:
        raise KeyError(f"unknown cell type {cell_type!r}; known: {_TYPES}")
    hemi = table[cell_type]
    if hemisphere not in hemi:
        raise KeyError(
            f"unknown hemisphere {hemisphere!r}; known: {tuple(hemi)}")
    return hemi[hemisphere]


# --------------------------------------------------------- public ID accessors
def lc4_ids(hemisphere: str = "both") -> tuple[int, ...]:
    """FlyWire IDs of LC4 (lobula columnar, ~angular velocity). ``"left"``/``"right"``/``"both"``."""
    return _ids("LC4", hemisphere)


def lplc2_ids(hemisphere: str = "both") -> tuple[int, ...]:
    """FlyWire IDs of LPLC2 (lobula plate/lobula columnar, ~looming/size). ``"left"``/``"right"``/``"both"``."""
    return _ids("LPLC2", hemisphere)


def dnp01_ids(hemisphere: str = "both") -> tuple[int, ...]:
    """FlyWire IDs of DNp01 (the Giant Fiber escape command neuron). ``"left"``/``"right"``/``"both"``."""
    return _ids("DNp01", hemisphere)


# Eager snapshots, for callers (e.g. EB-1) that want plain dicts at import time.
LC4_IDS: dict[str, tuple[int, ...]] = {h: lc4_ids(h) for h in (*_HEMIS, "both")}
LPLC2_IDS: dict[str, tuple[int, ...]] = {h: lplc2_ids(h) for h in (*_HEMIS, "both")}
DNP01_IDS: dict[str, tuple[int, ...]] = {h: dnp01_ids(h) for h in (*_HEMIS, "both")}


def provenance() -> dict:
    """Provenance + per-type/hemisphere counts for the escape neurons.

    Returns the artifact's ``meta`` (sources, FlyWire release, query date, the
    convergence summaries) plus a ``counts`` block derived from the resolved
    ID sets, so a report or a downstream contract can cite exactly where these
    identified cells come from.
    """
    art = _load_artifact()
    table = _ids_by_type_hemi()
    counts = {
        t: {h: len(table[t][h]) for h in (*_HEMIS, "both")} for t in _TYPES}
    return {
        "artifact": str(_ARTIFACT.relative_to(_HERE.parents[2])),
        "flywire_release": art["meta"].get("flywire_release"),
        "query_date": art["meta"].get("query_date"),
        "sources": art["meta"].get("sources"),
        "counts": counts,
        "convergence_per_hemisphere_syn_ge_1":
            art["meta"].get("convergence_per_hemisphere_syn_ge_1"),
        "all_edges_ipsilateral": art["meta"].get("all_edges_ipsilateral"),
    }


def resolve_in_brain(brain: "BrainModel") -> dict:
    """Check every escape-neuron ID against ``brain``'s v783 neuron list.

    Returns ``{type: {hemisphere: {"total", "resolved", "missing"}}}``. With the
    v783-keyed CX-1 artifact and the v783 ``Completeness`` list, all 316 resolve;
    this helper makes that auditable (and would surface any release/version drift).
    """
    table = _ids_by_type_hemi()
    out: dict[str, dict[str, dict]] = {}
    for t in _TYPES:
        out[t] = {}
        for h in _HEMIS:
            ids = table[t][h]
            missing = [i for i in ids if not brain.has(i)]
            out[t][h] = {
                "total": len(ids),
                "resolved": len(ids) - len(missing),
                "missing": missing,
            }
    return out


# ------------------------------------------------------------ the brain step
def looming_to_giant_fiber(
    brain: "BrainModel",
    lc4_hz: float,
    lplc2_hz: float,
    *,
    hemisphere: str = "both",
    duration_s: float = 1.0,
) -> float:
    """Drive the looming pathway (LC4 + LPLC2) and read the Giant Fiber's rate.

    The brain half of the escape loop, as a single function EB-1 can call as its
    brain step: activate LC4 at ``lc4_hz`` and LPLC2 at ``lplc2_hz`` (Poisson
    drive standing in for the looming visual input), advance the brain
    ``duration_s``, and return the **mean DNp01 firing rate in Hz**.

    This is a *clean, single-shot* measurement: it ``clear()``s any pending
    activation/silencing and rebuilds, so repeated calls are independent (each
    point on a response sweep is its own trial). EB-1 driving a persistent
    closed loop should manage activation itself and read ``dnp01_ids()`` rates
    from ``brain.step(...)`` directly; this helper is the convenience/contract
    path used for the brain-only validation and as a drop-in step.

    Parameters
    ----------
    brain       : a loaded :class:`BrainModel`
    lc4_hz      : Poisson activation rate for LC4 (angular-velocity channel)
    lplc2_hz    : Poisson activation rate for LPLC2 (looming/size channel)
    hemisphere  : ``"both"`` (default), ``"left"`` or ``"right"`` — which side's
                  LC4/LPLC2 to drive and which DNp01 to read
    duration_s  : measurement window (default 1.0 s)

    Returns
    -------
    float : mean DNp01 firing rate (Hz) over the chosen hemisphere's GF neuron(s).
    """
    lc4 = list(lc4_ids(hemisphere))
    lplc2 = list(lplc2_ids(hemisphere))
    gf = list(dnp01_ids(hemisphere))

    brain.clear()
    # BrainModel supports up to two distinct activation rates. Activating LC4 and
    # LPLC2 at (possibly) different Hz is exactly those two groups; if the rates
    # are equal they collapse to one group, which is also fine.
    if lc4_hz > 0 and lc4:
        brain.activate(lc4, hz=lc4_hz)
    if lplc2_hz > 0 and lplc2:
        brain.activate(lplc2, hz=lplc2_hz)

    rates = brain.run(duration_s, query=gf, rebuild=True)
    return sum(rates[i] for i in gf) / len(gf)
