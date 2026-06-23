"""Connectome LIF brain layer for the embodied fly.

A thin, addressable wrapper over the vendored Shiu *Drosophila* brain model
(FlyWire v783). See `brain_model.BrainModel` and `_shiu/README.md`.
"""

from .brain_model import BrainModel
from .neurons import (
    DNP01_IDS,
    LC4_IDS,
    LPLC2_IDS,
    dnp01_ids,
    lc4_ids,
    looming_to_giant_fiber,
    lplc2_ids,
    provenance,
    resolve_in_brain,
)

__all__ = [
    "BrainModel",
    "lc4_ids",
    "lplc2_ids",
    "dnp01_ids",
    "LC4_IDS",
    "LPLC2_IDS",
    "DNP01_IDS",
    "provenance",
    "resolve_in_brain",
    "looming_to_giant_fiber",
]
