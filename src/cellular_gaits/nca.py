"""Neural cellular automaton motor controller.

State tensor: ``(1, CHANNELS=4, H=8, W=8)``.

Update rule: a single shared 2-layer MLP that consumes a cell and its 8
Moore neighbors and emits the new C-d cell state. We implement it as a 3x3
zero-padded conv (mathematically identical to applying the same MLP at each
grid position; boundary cells receive zero for missing neighbors). Hidden
width 16, tanh between layers, output clamped to [-1, 1]. One CA tick per
control step.

Closed-loop sensors (C2-A)
--------------------------
The v1 controller walked blind. We close the loop by feeding the fly's own
proprioception back into the grid every tick. Wiring choice **(a): extra
input channels** — ``conv1`` is widened from ``CHANNELS`` to
``CHANNELS + SENSOR_CHANNELS`` input channels, and a per-tick sensor map is
concatenated onto the 4-channel state before the rule runs. The CA *state*
stays 4 channels (``conv2`` output is unchanged), so the recurrent dynamics
and the motor readout are untouched.

Two sensor channels, placed topographically so each sensor sits near where
it acts:
  - channel 0 (SENSOR): 42 joint angles in the 7x6 motor block (rows 0-6,
    cols 0-5), normalized to ~[-1, 1] by ctrlrange; zeros elsewhere.
  - channel 1 (SENSOR): 6 foot-contact booleans in row 7, cols 0-5; zeros
    elsewhere.

A/B integrity: the new input-channel weights are **zero-initialized**, and
feeding a zero sensor map (or ``sensors_enabled=False``) contributes exactly
nothing through ``conv1``. So a closed-loop NCA warm-started from v1 weights
reproduces the v1 open-loop dynamics bit-for-bit until evolution moves the
sensor weights off zero. ``warm_start_from_v1`` performs that load.

Chemotaxis sensors (CH-A)
-------------------------
``NCA(chemo=True)`` adds **two more** input channels on top of the two proprio
channels, so ``conv1`` widens to ``CHANNELS + SENSOR_CHANNELS + CHEMO_CHANNELS``
= ``4 + 2 + 2 = 8`` input channels. The two chemo channels carry a *bilateral*
odor reading — left antenna ``cL`` and right antenna ``cR`` — placed
topographically: the left value fills the left half of the motor block, the
right value the right half (channels 6 and 7 respectively). This left-vs-right
asymmetry is the whole point: turning toward the source must fall out of
``cL`` vs ``cR``, not a hard-coded bias.

The same zero-init / A/B argument applies one level up:
``warm_start_from_closed_loop`` loads the trained 6-input closed-loop weights
into the first six input channels and leaves the two chemo channels at zero, so
a chemo NCA reproduces the closed-loop walking dynamics exactly until evolution
moves the chemo weights off zero (``chemo``-zeroed == closed-loop behaviour).
The default ``NCA()`` (``chemo=False``) is unchanged and stays bit-identical to
the closed-loop controller.

Motor cells: 42 cells laid out as a contiguous 7x6 sub-grid (rows 0-6,
cols 0-5). Their channel-0 values are read in row-major order and become
joint targets for Flygym's 42 LEGS_ACTIVE_ONLY position actuators.

CMA-ES operates on a flat float64 parameter vector; ``flatten_params()``
and ``set_params()`` provide a round-trip-stable boundary between the torch
model and CMA's optimizer state.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

GRID_H = 8
GRID_W = 8
CHANNELS = 4
SENSOR_CHANNELS = 2  # proprio: ch0 joint angles, ch1 foot contacts
CHEMO_CHANNELS = 2  # bilateral odor: ch0 left antenna (cL), ch1 right (cR)
HIDDEN_DEFAULT = 16

MOTOR_ROWS = 7
MOTOR_COLS = 6
N_MOTORS = MOTOR_ROWS * MOTOR_COLS  # 42

N_LEGS = 6
CONTACT_ROW = 7  # foot contacts written into the bottom grid row

STATE_SHAPE = (1, CHANNELS, GRID_H, GRID_W)
SENSOR_SHAPE = (1, SENSOR_CHANNELS, GRID_H, GRID_W)
# Chemo channels split the 7x6 motor block into a left and a right half. The
# left antenna value fills the left columns, the right antenna the right ones.
CHEMO_COL_SPLIT = MOTOR_COLS // 2  # cols [0, SPLIT) = left, [SPLIT, MOTOR_COLS) = right

# v1 (open-loop) parameter count, for warm-start validation. conv1 4->16 (3x3)
# = 4*16*9 + 16 = 592; conv2 16->4 (1x1) = 16*4 + 4 = 68; total 660.
V1_N_PARAMS = (CHANNELS * HIDDEN_DEFAULT * 9 + HIDDEN_DEFAULT) + (
    HIDDEN_DEFAULT * CHANNELS + CHANNELS
)

# Closed-loop (C2-A) parameter count, for chemo warm-start validation. conv1
# (4+2)->16 (3x3) = 6*16*9 + 16 = 880; conv2 unchanged = 68; total 948.
CL_N_PARAMS = (
    (CHANNELS + SENSOR_CHANNELS) * HIDDEN_DEFAULT * 9 + HIDDEN_DEFAULT
) + (HIDDEN_DEFAULT * CHANNELS + CHANNELS)


class NCA(nn.Module):
    def __init__(
        self, hidden: int = HIDDEN_DEFAULT, gain: float = 1.0, chemo: bool = False
    ) -> None:
        super().__init__()
        self.hidden = hidden
        # chemo=False: 6-input closed-loop controller (4 state + 2 proprio).
        # chemo=True: 8-input chemotaxis controller (+ 2 bilateral odor channels).
        self.chemo = bool(chemo)
        self.sensor_channels = SENSOR_CHANNELS + (CHEMO_CHANNELS if chemo else 0)
        self.sensor_shape = (1, self.sensor_channels, GRID_H, GRID_W)
        # Criticality knob: scales the pre-activation fed to tanh. gain=1.0 is
        # the trained operating point (identity); <1 drives the CA toward the
        # ordered/linear regime, >1 toward saturation/chaos. NOT a learned
        # parameter — it is held fixed during a rollout and swept externally.
        self.gain = float(gain)
        self.conv1 = nn.Conv2d(
            CHANNELS + self.sensor_channels,
            hidden,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(hidden, CHANNELS, kernel_size=1)
        for p in self.parameters():
            p.requires_grad_(False)
        # Zero the sensor input-channel weights so a default-constructed model
        # ignores sensors until they are trained on (A/B integrity).
        with torch.no_grad():
            self.conv1.weight[:, CHANNELS:, :, :].zero_()
        self._n_params = sum(p.numel() for p in self.parameters())

    @property
    def n_params(self) -> int:
        return self._n_params

    def step(
        self, state: torch.Tensor, sensors: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tuple(state.shape) != STATE_SHAPE:
            raise ValueError(
                f"NCA.step expected state of shape {STATE_SHAPE}, got {tuple(state.shape)}"
            )
        if sensors is None:
            sensors = torch.zeros(*self.sensor_shape, dtype=state.dtype)
        elif tuple(sensors.shape) != self.sensor_shape:
            raise ValueError(
                f"NCA.step expected sensors of shape {self.sensor_shape}, "
                f"got {tuple(sensors.shape)}"
            )
        x = torch.cat([state, sensors], dim=1)
        h = torch.tanh(self.gain * self.conv1(x))
        out = self.conv2(h)
        return torch.clamp(out, -1.0, 1.0)

    def motor_targets(self, state: torch.Tensor) -> np.ndarray:
        sub = state[0, 0, :MOTOR_ROWS, :MOTOR_COLS]
        return sub.detach().reshape(-1).cpu().numpy().astype(np.float64)

    @staticmethod
    def build_sensor_map(
        joint_angles_unit: np.ndarray, foot_contacts: np.ndarray
    ) -> torch.Tensor:
        """Lay raw sensor readings out spatially into a (1, 2, 8, 8) tensor.

        ``joint_angles_unit``: 42 angles already normalized to ~[-1, 1].
        ``foot_contacts``: 6 booleans/floats in {0, 1}.
        """
        ja = np.asarray(joint_angles_unit, dtype=np.float32).reshape(-1)
        fc = np.asarray(foot_contacts, dtype=np.float32).reshape(-1)
        if ja.size != N_MOTORS:
            raise ValueError(f"expected {N_MOTORS} joint angles, got {ja.size}")
        if fc.size != N_LEGS:
            raise ValueError(f"expected {N_LEGS} foot contacts, got {fc.size}")
        sensors = torch.zeros(*SENSOR_SHAPE, dtype=torch.float32)
        sensors[0, 0, :MOTOR_ROWS, :MOTOR_COLS] = torch.from_numpy(
            ja.reshape(MOTOR_ROWS, MOTOR_COLS)
        )
        sensors[0, 1, CONTACT_ROW, :N_LEGS] = torch.from_numpy(fc)
        return sensors

    @staticmethod
    def build_chemo_sensor_map(
        joint_angles_unit: np.ndarray,
        foot_contacts: np.ndarray,
        c_left: float,
        c_right: float,
    ) -> torch.Tensor:
        """Lay proprio + bilateral odor out into a (1, 4, 8, 8) tensor.

        Channels 0-1 are the proprio map (identical to ``build_sensor_map``).
        Channels 2-3 are the bilateral odor reading, placed topographically:
        ``c_left`` fills the left half of the 7x6 motor block (cols
        [0, CHEMO_COL_SPLIT)) and ``c_right`` the right half, so a left-vs-right
        concentration difference is presented as a left-vs-right spatial bias
        over the very cells that drive the legs.
        """
        ja = np.asarray(joint_angles_unit, dtype=np.float32).reshape(-1)
        fc = np.asarray(foot_contacts, dtype=np.float32).reshape(-1)
        if ja.size != N_MOTORS:
            raise ValueError(f"expected {N_MOTORS} joint angles, got {ja.size}")
        if fc.size != N_LEGS:
            raise ValueError(f"expected {N_LEGS} foot contacts, got {fc.size}")
        sensors = torch.zeros(1, SENSOR_CHANNELS + CHEMO_CHANNELS, GRID_H, GRID_W)
        sensors[0, 0, :MOTOR_ROWS, :MOTOR_COLS] = torch.from_numpy(
            ja.reshape(MOTOR_ROWS, MOTOR_COLS)
        )
        sensors[0, 1, CONTACT_ROW, :N_LEGS] = torch.from_numpy(fc)
        # ch2 = left antenna, ch3 = right antenna.
        sensors[0, 2, :MOTOR_ROWS, :CHEMO_COL_SPLIT] = float(c_left)
        sensors[0, 3, :MOTOR_ROWS, CHEMO_COL_SPLIT:MOTOR_COLS] = float(c_right)
        return sensors

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

    def warm_start_from_v1(self, v1_vec: np.ndarray) -> None:
        """Load a v1 (open-loop, 4-input-channel) parameter vector.

        The v1 layout is conv1.weight (16, 4, 3, 3), conv1.bias (16,),
        conv2.weight (4, 16, 1, 1), conv2.bias (4,). We copy those into the
        first 4 input channels of conv1 and into conv2 verbatim, leaving the
        2 sensor input channels at zero. The result reproduces v1 dynamics
        exactly when sensors are zeroed.
        """
        v = np.asarray(v1_vec, dtype=np.float32).reshape(-1)
        if v.size != V1_N_PARAMS:
            raise ValueError(
                f"warm_start_from_v1: expected {V1_N_PARAMS} v1 params, got {v.size}"
            )
        c1w_n = self.hidden * CHANNELS * 9
        c1b_n = self.hidden
        c2w_n = CHANNELS * self.hidden
        c2b_n = CHANNELS
        off = 0
        c1w = v[off : off + c1w_n].reshape(self.hidden, CHANNELS, 3, 3)
        off += c1w_n
        c1b = v[off : off + c1b_n]
        off += c1b_n
        c2w = v[off : off + c2w_n].reshape(CHANNELS, self.hidden, 1, 1)
        off += c2w_n
        c2b = v[off : off + c2b_n]
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.weight[:, :CHANNELS, :, :].copy_(torch.from_numpy(c1w))
            self.conv1.bias.copy_(torch.from_numpy(c1b))
            self.conv2.weight.copy_(torch.from_numpy(c2w))
            self.conv2.bias.copy_(torch.from_numpy(c2b))

    def warm_start_from_closed_loop(self, cl_vec: np.ndarray) -> None:
        """Load a closed-loop (6-input-channel) parameter vector into a chemo NCA.

        The closed-loop layout is conv1.weight (16, 6, 3, 3), conv1.bias (16,),
        conv2.weight (4, 16, 1, 1), conv2.bias (4,). We copy conv1's weights
        into the first 6 input channels of this 8-input conv1, leave the 2 chemo
        channels at zero, and copy conv1.bias / conv2 verbatim. The result
        reproduces the closed-loop dynamics exactly when the chemo channels are
        zeroed (A/B integrity).
        """
        if not self.chemo:
            raise ValueError(
                "warm_start_from_closed_loop requires NCA(chemo=True); "
                "use warm_start_from_v1 for the 6-input closed-loop model"
            )
        v = np.asarray(cl_vec, dtype=np.float32).reshape(-1)
        if v.size != CL_N_PARAMS:
            raise ValueError(
                f"warm_start_from_closed_loop: expected {CL_N_PARAMS} closed-loop "
                f"params, got {v.size}"
            )
        in_cl = CHANNELS + SENSOR_CHANNELS  # 6
        c1w_n = self.hidden * in_cl * 9
        c1b_n = self.hidden
        c2w_n = CHANNELS * self.hidden
        c2b_n = CHANNELS
        off = 0
        c1w = v[off : off + c1w_n].reshape(self.hidden, in_cl, 3, 3)
        off += c1w_n
        c1b = v[off : off + c1b_n]
        off += c1b_n
        c2w = v[off : off + c2w_n].reshape(CHANNELS, self.hidden, 1, 1)
        off += c2w_n
        c2b = v[off : off + c2b_n]
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.weight[:, :in_cl, :, :].copy_(torch.from_numpy(c1w))
            self.conv1.bias.copy_(torch.from_numpy(c1b))
            self.conv2.weight.copy_(torch.from_numpy(c2w))
            self.conv2.bias.copy_(torch.from_numpy(c2b))


def print_param_summary() -> None:
    nca = NCA()
    print(f"NCA params: {nca.n_params} (target: <=1000; v1 was {V1_N_PARAMS})")


if __name__ == "__main__":
    print_param_summary()
