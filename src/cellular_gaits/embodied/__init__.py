"""Embodied-brain body side: on-demand motor primitives for NeuroMechFly.

EB-0C ships the escape/flee primitive (``body.py``): a callable that drives the
MuJoCo fly into a fast, directed bolt on command, by *reusing* the trained X-A
escape controller (no new RL). The scalar ``drive`` it takes is the clean seam
EB-1 will wire to the connectome brain's DNp01 (Giant Fiber) firing rate.
"""

from .body import (
    CALIBRATED_DRIVE,
    EscapeMotor,
    apply_escape,
    escape_metrics,
    escape_pulse_drive,
    loom_from_drive,
)

__all__ = [
    "CALIBRATED_DRIVE",
    "EscapeMotor",
    "apply_escape",
    "escape_metrics",
    "escape_pulse_drive",
    "loom_from_drive",
]
