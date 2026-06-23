"""EB-0C validation: the reused escape primitive drives a directed bolt in MuJoCo.

Protocol mirrors the X-A trained regime — walk (drive=0) to establish the gait,
then a transient escape pulse (a stand-in for EB-1's DNp01 firing-rate profile:
rises and falls as a threat passes). The loom is *synthetic*, split from the
scalar drive — no threat object. Escape and a no-drive baseline share the env
seed and warm-up, so they diverge only at onset: the difference is the loom's
causal effect. We report, over the early reaction window (~X-A's 30-50 ms
latency), the loom-induced turn and displacement, and check the bolt is

  * directed   — left-loom and right-loom turn opposite ways,
  * X-A-polar  — left-loom -> CW/right turn, right-loom -> CCW/left turn,
  * stable     — the body stays upright (thorax height ~ baseline), and
  * reproducible — same drive/direction => bit-identical metric.

Run:  uv run python scratch/eb/validate_body.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cellular_gaits.embodied.body import (
    CALIBRATED_DRIVE,
    EscapeMotor,
    escape_pulse_drive,
)
from cellular_gaits.env import CONTROL_DT_S, FlyEnv

N_WARM = 120     # plain-walk steps to establish the gait (~0.48 s)
N_REACT = 120    # reaction window after onset (~0.48 s)
EARLY = 15       # reaction-latency sub-window (~0.06 s) — X-A's 28-48 ms regime,
                 # where the directed impulse lives (see body.py CALIBRATED_DRIVE)
PEAK = CALIBRATED_DRIVE  # calibrated moderate operating point (0.2)


def _unwrap_deg(yaw: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.asarray(yaw, dtype=np.float64)))


def rollout(motor: EscapeMotor, peak: float, direction):
    env = FlyEnv()  # fresh, seeded, no threat
    drive = escape_pulse_drive(peak=peak, onset=N_WARM)
    out = motor.apply_escape(env, drive, direction=direction, n_steps=N_WARM + N_REACT)
    return out["traj"]


def summarize(traj: dict) -> dict:
    yaw = _unwrap_deg(traj["yaw"])
    xy = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, :2]
    z = np.asarray(traj["thorax_xyz"], dtype=np.float64)[:, 2]
    return {
        "turn_early_deg": float(yaw[N_WARM + EARLY] - yaw[N_WARM]),
        "turn_react_deg": float(yaw[N_WARM + N_REACT] - yaw[N_WARM]),
        "disp_react": float(np.linalg.norm(xy[N_WARM + N_REACT] - xy[N_WARM])),
        "min_thorax_z": float(np.min(z[N_WARM:])),
        "final_thorax_z": float(z[-1]),
    }


def main() -> None:
    motor = EscapeMotor()
    print(f"reused controller: run_id={motor.run_id}  n_params={motor.nca.n_params}  "
          f"loom_input_gain={motor.loom_input_gain}")
    print(f"protocol: {N_WARM} warm-up steps -> escape pulse (peak={PEAK}) -> "
          f"{N_REACT}-step window; early window = {EARLY} steps "
          f"(~{EARLY * CONTROL_DT_S:.2f} s)\n")

    base = summarize(rollout(motor, 0.0, "front"))
    left = summarize(rollout(motor, PEAK, "left"))
    right = summarize(rollout(motor, PEAK, "right"))
    front = summarize(rollout(motor, PEAK, "front"))

    bl_e = base["turn_early_deg"]
    rows = {"baseline": base, "left": left, "right": right, "front": front}
    print(f"{'condition':10s} {'early-turn':>11s} {'loom-induced':>13s} "
          f"{'disp':>7s} {'min_z':>7s} {'final_z':>8s}")
    for name, r in rows.items():
        induced = "" if name == "baseline" else f"{r['turn_early_deg'] - bl_e:+10.1f}°"
        print(f"{name:10s} {r['turn_early_deg']:+9.1f}°  {induced:>13s} "
              f"{r['disp_react']:7.2f} {r['min_thorax_z']:7.3f} {r['final_thorax_z']:8.3f}")

    rel_left = left["turn_early_deg"] - bl_e
    rel_right = right["turn_early_deg"] - bl_e
    print("\n--- headline checks ---")
    directed = rel_left * rel_right < 0
    print(f"[{'PASS' if directed else 'FAIL'}] directed: left/right turn opposite "
          f"({rel_left:+.1f}° vs {rel_right:+.1f}°)")
    polar = rel_left < 0 < rel_right
    print(f"[{'PASS' if polar else 'FAIL'}] X-A polarity: left-loom->CW/right turn, "
          f"right-loom->CCW/left turn")
    stable = min(left["min_thorax_z"], right["min_thorax_z"]) > 0.5 * base["min_thorax_z"]
    print(f"[{'PASS' if stable else 'FAIL'}] stable: body stays upright "
          f"(min_z left {left['min_thorax_z']:.3f}, right {right['min_thorax_z']:.3f}, "
          f"baseline {base['min_thorax_z']:.3f})")
    fled = max(left["disp_react"], right["disp_react"]) > 1.0
    print(f"[{'PASS' if fled else 'CHECK'}] body flees: displacement "
          f"left {left['disp_react']:.2f}, right {right['disp_react']:.2f}")

    a = summarize(rollout(motor, PEAK, "left"))
    b = summarize(rollout(motor, PEAK, "left"))
    rep = a == b
    print(f"[{'PASS' if rep else 'FAIL'}] reproducible (left twice, bit-identical)")

    report = {
        "controller": {"run_id": motor.run_id, "n_params": int(motor.nca.n_params),
                       "loom_input_gain": motor.loom_input_gain},
        "protocol": {"n_warm": N_WARM, "n_react": N_REACT, "early_steps": EARLY,
                     "peak_drive": PEAK, "control_dt_s": CONTROL_DT_S},
        "conditions": rows,
        "checks": {"directed": bool(directed), "xa_polarity": bool(polar),
                   "stable": bool(stable), "flees": bool(fled), "reproducible": bool(rep)},
        "loom_induced_early_turn_deg": {"left": rel_left, "right": rel_right},
    }
    out = Path(__file__).resolve().parent / "body_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
