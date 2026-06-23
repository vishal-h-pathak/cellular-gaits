"""Connectome LIF brain layer for the embodied fly.

A thin, addressable wrapper over the vendored Shiu *Drosophila* brain model
(FlyWire v783). See `brain_model.BrainModel` and `_shiu/README.md`.
"""

from .brain_model import BrainModel

__all__ = ["BrainModel"]
