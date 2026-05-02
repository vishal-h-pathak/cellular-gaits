"""CA state log writer — stable schema for the React widget.

Schema (frozen — the React side reads this):

    {
      "meta": {
        "run_id":       "<UTC ISO-ish>",
        "n_ticks":      <int>,
        "grid":         [H, W],
        "channels":     C,
        "control_dt_s": 0.004,
        "motor_cells":  [[r, c], ...]   (length = N_MOTORS, row-major)
      },
      "frames": [                        # length = n_ticks
        [                                 # length = H*W = 64
          [c0, c1, c2, c3],               # one cell, channels
          ...
        ],
        ...
      ]
    }

Cell ordering inside a frame is row-major over the 8x8 grid:
``cell_index = r * W + c``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..nca import CHANNELS, GRID_H, GRID_W, MOTOR_COLS, MOTOR_ROWS

CONTROL_DT_S = 0.004


def state_to_frame(state: torch.Tensor, ndigits: int = 4) -> list[list[float]]:
    arr = state[0].detach().cpu().numpy()  # (C, H, W)
    flat = arr.transpose(1, 2, 0).reshape(GRID_H * GRID_W, CHANNELS)
    return np.round(flat, ndigits).tolist()


def write_state_log(
    out_path: Path,
    frames: list[list[list[float]]],
    run_id: str,
) -> None:
    motor_cells = [[r, c] for r in range(MOTOR_ROWS) for c in range(MOTOR_COLS)]
    payload = {
        "meta": {
            "run_id": run_id,
            "n_ticks": len(frames),
            "grid": [GRID_H, GRID_W],
            "channels": CHANNELS,
            "control_dt_s": CONTROL_DT_S,
            "motor_cells": motor_cells,
        },
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
