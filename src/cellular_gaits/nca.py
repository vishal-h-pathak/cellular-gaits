"""Neural cellular automaton motor controller.

State tensor: ``(1, CHANNELS=4, H=8, W=8)``.

Update rule: a single shared 2-layer MLP that consumes a cell and its 8
Moore neighbors (9 * C = 36 inputs) and emits the new C-d cell state. We
implement it as a 3x3 zero-padded conv (mathematically identical to
applying the same MLP at each grid position; boundary cells receive zero
for missing neighbors). Hidden width 16, tanh between layers, output
clamped to [-1, 1]. One CA tick per control step.

Motor cells: 42 cells laid out as a contiguous 7x6 sub-grid (rows 0-6,
cols 0-5). Their channel-0 values are read in row-major order and become
joint targets for Flygym's 42 LEGS_ACTIVE_ONLY position actuators. This
deviates from the spec's "18 motors (3 DoF x 6 legs)" because Flygym
2.0.1's LEGS_ACTIVE_ONLY preset actuates 7 DoFs/leg = 42 (see README).

CMA-ES operates on a flat float64 parameter vector; ``flatten_params()``
and ``set_params()`` provide a round-trip-stable boundary between the
torch model and CMA's optimizer state.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

GRID_H = 8
GRID_W = 8
CHANNELS = 4
HIDDEN_DEFAULT = 16

MOTOR_ROWS = 7
MOTOR_COLS = 6
N_MOTORS = MOTOR_ROWS * MOTOR_COLS  # 42

STATE_SHAPE = (1, CHANNELS, GRID_H, GRID_W)


class NCA(nn.Module):
    def __init__(self, hidden: int = HIDDEN_DEFAULT) -> None:
        super().__init__()
        self.hidden = hidden
        self.conv1 = nn.Conv2d(
            CHANNELS, hidden, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.conv2 = nn.Conv2d(hidden, CHANNELS, kernel_size=1)
        for p in self.parameters():
            p.requires_grad_(False)
        self._n_params = sum(p.numel() for p in self.parameters())

    @property
    def n_params(self) -> int:
        return self._n_params

    def step(self, state: torch.Tensor) -> torch.Tensor:
        if tuple(state.shape) != STATE_SHAPE:
            raise ValueError(
                f"NCA.step expected state of shape {STATE_SHAPE}, got {tuple(state.shape)}"
            )
        h = torch.tanh(self.conv1(state))
        out = self.conv2(h)
        return torch.clamp(out, -1.0, 1.0)

    def motor_targets(self, state: torch.Tensor) -> np.ndarray:
        sub = state[0, 0, :MOTOR_ROWS, :MOTOR_COLS]
        return sub.detach().reshape(-1).cpu().numpy().astype(np.float64)

    @staticmethod
    def init_state(seed: int | None = None) -> torch.Tensor:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        s = torch.empty(*STATE_SHAPE)
        s.uniform_(-0.1, 0.1, generator=g)
        return s

    def flatten_params(self) -> np.ndarray:
        parts = [p.detach().reshape(-1) for p in self.parameters()]
        return torch.cat(parts).cpu().numpy().astype(np.float64)

    def set_params(self, vec: np.ndarray) -> None:
        v = torch.as_tensor(np.asarray(vec).reshape(-1), dtype=torch.float32)
        if v.numel() != self._n_params:
            raise ValueError(
                f"set_params: vector has {v.numel()} elements, model has {self._n_params}"
            )
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(v[offset : offset + n].view_as(p))
                offset += n


def print_param_summary() -> None:
    nca = NCA()
    print(f"NCA params: {nca.n_params} (target: <=1000)")


if __name__ == "__main__":
    print_param_summary()
