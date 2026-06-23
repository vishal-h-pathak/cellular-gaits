"""EB-0C — a body-side escape/flee motor primitive on NeuroMechFly.

The embodied loop (EMBODIED_BRAIN_PLAN.md, Phase 1) needs the *body half* of
escape: a callable that makes the MuJoCo fly bolt/flee on command, so it can be
built and validated in parallel with the brain. Later (EB-1) the same callable is
driven by the real FlyWire connectome's **DNp01 (Giant Fiber)** firing rate.

**This module trains nothing.** It REUSES the already-evolved X-A escape
controller — the 8-input loom NCA (`NCA(loom=True)`), run `xa-full`, best fitness
13.23 — whose weights are vendored here as ``escape_controller.json``
(``flat_params``, the same artifact the portfolio `data-x` bundle ships). See
``ops/reports/REPORT_x_a.md`` for how that controller was evolved.

The clean seam
--------------
In X-A the controller reacts to a **bilateral looming signal** ``(loom_L, loom_R)``
produced by a hand-built front-end watching a simulated threat. The body-side
primitive drops the threat entirely and instead *synthesizes* that bilateral loom
from a scalar **escape drive** plus a **direction**, using the controller's own
documented loom math, verbatim::

    m      = clip(drive, 0, 1)                 # escape drive  (~ DNp01 rate)
    loom_L = m * 0.5 * (1 + sin phi)           # phi = threat bearing, CCW+ = left
    loom_R = m * 0.5 * (1 - sin phi)

The trained controller already knows how to turn that L-vs-R asymmetry into a
fast directed bolt (left-loom ⇒ turn away to the right, and vice-versa — the
emergent mirror-image escape documented in REPORT_x_a.md §4). So ``drive`` is
exactly the scalar EB-1 maps the DNp01 firing rate onto, and ``direction`` is the
bilateral cue LC4/LPLC2 supply. No retraining is required to wrap it.

EB-1 contract
-------------
``apply_escape(env, drive, direction=..., n_steps=...)`` (and the
``EscapeMotor`` class behind it) is the stable entry point. ``drive`` is a scalar
in [0, 1]; ``direction`` is the threat bearing (``"left"``/``"right"``/``"front"``
or a float in radians, CCW-positive = the fly's left). It applies the escape
motion for one control window and returns the trajectory plus a displacement /
heading-change metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..env import CONTROL_DT_S, FlyEnv
from ..evolve_escape import CA_INIT_SEED, _make_escape_policy
from ..nca import NCA

# Vendored trained controller: the X-A escape weights (run `xa-full`).
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "escape_controller.json"

# The X-A controller was evolved with the bilateral loom amplified ×8 before it
# enters conv1 (REPORT_x_a.md §2 — the bang-bang warm-start gait needs the cue
# amplified to be steerable). Reusing the controller faithfully means reusing the
# same gain; it is part of the trained controller's operating point.
DEFAULT_LOOM_INPUT_GAIN = 8.0

# One escape "control window" for validation: ~0.6 s (150 steps @ 250 Hz). Long
# enough to see the bolt; EB-1 can call with whatever window its brain↔body sync
# uses (Eon 15 ms; escape is faster) and keep the CA state across windows.
DEFAULT_WINDOW_STEPS = 150

# Calibrated operating point (EB-0C validation, scratch/eb/validate_body.py). The
# reused controller gives a CLEAN, STABLE, directed bolt at a *moderate* drive: a
# brief pulse peaking near here turns the body away within the reaction-latency
# window (left-loom -> CW/right turn, right-loom -> CCW/left turn — the emergent
# X-A polarity) and keeps it upright. Two honest caveats this number encodes:
#   * The directed impulse lives in the first ~40-60 ms after onset (X-A's 28-48 ms
#     reaction latency); held longer, a strong unilateral synthetic loom drifts
#     off-distribution and the turn becomes chaotic. Escape is a *transient*.
#   * Saturating the drive (>~0.5 held) overdrives the gait into a tumble (the body
#     loses balance). EB-1's DNp01->drive map should target this moderate range.
CALIBRATED_DRIVE = 0.2

# Named threat bearings in the fly's body frame (radians, CCW-positive = left),
# matching the X-A azimuth convention (front 0°, left +90°, right −90°/270°).
_NAMED_DIRECTIONS = {
    "front": 0.0,
    "left": np.pi / 2,
    "right": -np.pi / 2,
    "back": np.pi,  # symmetric loom (excites both eyes); no preferred side
}


def _bearing_radians(direction: float | str) -> float:
    """Resolve a direction (name or radians) to a body-frame bearing phi."""
    if isinstance(direction, str):
        key = direction.strip().lower()
        if key not in _NAMED_DIRECTIONS:
            raise ValueError(
                f"unknown direction {direction!r}; use one of "
                f"{sorted(_NAMED_DIRECTIONS)} or a float bearing in radians"
            )
        return _NAMED_DIRECTIONS[key]
    return float(direction)


def loom_from_drive(drive: float, direction: float | str) -> tuple[float, float]:
    """Synthesize the bilateral loom ``(loom_L, loom_R)`` from a scalar drive.

    ``drive`` is clipped to [0, 1] (it is a normalized escape command, the seam
    EB-1 maps the DNp01 firing rate onto). ``direction`` is the threat bearing
    (CCW-positive = left). Uses the X-A loom split verbatim::

        loom_L = m * 0.5 * (1 + sin phi),  loom_R = m * 0.5 * (1 - sin phi)
    """
    m = float(np.clip(drive, 0.0, 1.0))
    phi = _bearing_radians(direction)
    sin_phi = float(np.sin(phi))
    loom_l = m * 0.5 * (1.0 + sin_phi)
    loom_r = m * 0.5 * (1.0 - sin_phi)
    return loom_l, loom_r


def escape_pulse_drive(
    peak: float = CALIBRATED_DRIVE,
    onset: int = 0,
    rise: int = 15,
    hold: int = 20,
    fall: int = 25,
):
    """A transient escape-drive schedule ``drive(t) -> float`` (rise/hold/fall).

    A representative stand-in for the **DNp01 firing-rate profile** EB-1 will feed:
    a brief pulse that rises to ``peak`` at ``onset``, holds, then decays back to
    zero — mirroring a looming threat that approaches and passes. ``peak`` defaults
    to the calibrated moderate operating point (:data:`CALIBRATED_DRIVE`); see its
    note on why a sustained max drive is *not* the right command.
    """

    def drive(t: int) -> float:
        u = t - onset
        if u < 0:
            return 0.0
        if u < rise:
            return peak * u / rise
        if u < rise + hold:
            return peak
        if u < rise + hold + fall:
            return peak * max(0.0, 1.0 - (u - rise - hold) / fall)
        return 0.0

    return drive


def escape_metrics(traj: dict) -> dict:
    """Displacement + heading-change summary of one escape window.

    - ``displacement``: net horizontal distance the thorax travelled (xy).
    - ``heading_change``: signed yaw change over the window, wrapped to (−π, π].
    - ``peak_speed``: max per-step horizontal speed (body units / s).
    """
    xy = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    yaw = np.asarray(traj["yaw"], dtype=np.float64)
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    dyaw = float(yaw[-1] - yaw[0])
    heading_change = float((dyaw + np.pi) % (2 * np.pi) - np.pi)  # wrap to (−π, π]
    if xy.shape[0] > 1:
        step_d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        peak_speed = float(step_d.max() / CONTROL_DT_S)
    else:
        peak_speed = 0.0
    return {
        "displacement": displacement,
        "heading_change": heading_change,
        "peak_speed": peak_speed,
        "n_steps": int(xy.shape[0] - 1),
        "window_seconds": round((xy.shape[0] - 1) * CONTROL_DT_S, 4),
    }


class EscapeMotor:
    """The body-side escape primitive: a trained controller driven by a scalar.

    Wraps the reused X-A escape NCA. Holds the controller and (between calls) the
    cellular-automaton state, so EB-1 can drive escape continuously across control
    windows. Stateless single-shot use goes through :meth:`apply_escape` (which
    resets the CA state by default); continuous use builds a per-window policy via
    :meth:`policy` and threads the state itself.
    """

    def __init__(
        self,
        params_path: str | Path = DEFAULT_PARAMS_PATH,
        loom_input_gain: float = DEFAULT_LOOM_INPUT_GAIN,
        ca_seed: int = CA_INIT_SEED,
    ) -> None:
        payload = json.loads(Path(params_path).read_text())
        self.params = np.asarray(payload["flat_params"], dtype=np.float64)
        self.run_id = payload.get("meta", {}).get("run_id", "unknown")
        self.loom_input_gain = float(loom_input_gain)
        self.ca_seed = int(ca_seed)
        self.nca = NCA(loom=True)
        if self.nca.n_params != self.params.size:
            raise ValueError(
                f"param count mismatch: controller expects {self.nca.n_params}, "
                f"vendored flat_params has {self.params.size}"
            )
        self.nca.set_params(self.params)

    def policy(self, drive: float, direction: float | str = "front"):
        """Build a ``policy(t, sensors)`` for one escape window.

        The returned closure injects the synthesized bilateral loom into the
        sensors dict each control step, then defers to the X-A escape policy (the
        same ``_make_escape_policy`` used in training). Its CA state is fresh and
        private to this call; for continuous control hold onto one policy and call
        it across steps. ``drive`` may be a scalar (constant over the window) or a
        callable ``drive(t) -> float`` for a time-varying DNp01 ramp.
        """
        inner = _make_escape_policy(
            self.nca, ca_seed=self.ca_seed, loom_input_gain=self.loom_input_gain
        )

        def policy(t: int, sensors: dict) -> np.ndarray:
            d = drive(t) if callable(drive) else drive
            loom_l, loom_r = loom_from_drive(d, direction)
            sensors = dict(sensors)
            sensors["loom_left"] = loom_l
            sensors["loom_right"] = loom_r
            return inner(t, sensors)

        return policy

    def apply_escape(
        self,
        env: FlyEnv,
        drive: float,
        direction: float | str = "front",
        n_steps: int = DEFAULT_WINDOW_STEPS,
    ) -> dict:
        """Apply the escape motion for one control window and report a metric.

        This is the EB-1 contract entry point. ``env`` is a fresh/seeded
        :class:`FlyEnv` (no threat set — the loom is synthesized, not simulated).
        Returns ``{"traj": ..., "metrics": ..., "loom": (loom_L, loom_R),
        "drive": ..., "direction": ...}``.
        """
        if env._threat is not None:  # noqa: SLF001 — guard the synthetic-loom seam
            raise ValueError(
                "apply_escape drives a synthetic loom; env must have no threat set "
                "(env.set_threat(None)) so the simulated loom does not override it"
            )
        policy = self.policy(drive, direction)
        _, traj = env.rollout(policy, n_steps=n_steps, pass_sensors=True)
        d = drive(n_steps - 1) if callable(drive) else drive
        return {
            "traj": traj,
            "metrics": escape_metrics(traj),
            "loom": loom_from_drive(d, direction),
            "drive": float(np.clip(d, 0.0, 1.0)),
            "direction": direction,
            "bearing_rad": _bearing_radians(direction),
            "run_id": self.run_id,
        }


# Module-level convenience: construct a motor (cached) and drive one window.
_DEFAULT_MOTOR: EscapeMotor | None = None


def apply_escape(
    env: FlyEnv,
    drive: float,
    direction: float | str = "front",
    n_steps: int = DEFAULT_WINDOW_STEPS,
    motor: EscapeMotor | None = None,
) -> dict:
    """Drive the escape primitive for one control window (see EB-1 contract).

    Convenience wrapper around :meth:`EscapeMotor.apply_escape`. Reuses a cached
    default :class:`EscapeMotor` unless one is passed in.
    """
    global _DEFAULT_MOTOR
    if motor is None:
        if _DEFAULT_MOTOR is None:
            _DEFAULT_MOTOR = EscapeMotor()
        motor = _DEFAULT_MOTOR
    return motor.apply_escape(env, drive, direction=direction, n_steps=n_steps)
