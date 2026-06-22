"""NCA wrapped as a PPO-ready Gaussian policy (the N-RL policy slice).

This is the *policy* half of the reusable RL harness: it makes the existing
``NCA(nav=True)`` controller (1524 params, 10-input nav mode) usable by a
cleanrl-style PPO loop without touching ``nca.py``. The env and the PPO loop are
built by parallel sessions; this module only owns ``NCAPolicy``.

What it is
----------
A **Gaussian policy** over the 42 motor-cell targets:

  * the **action mean** is one NCA forward pass (``cat([ca_state, obs]) ->
    tanh(gain*conv1) -> conv2 -> clamp[-1,1]``), read off the 7x6 motor block;
  * a learned **global** ``log_std`` (one per motor, broadcast over the batch);
  * a small **value head** — an MLP on the spatially-pooled post-step CA grid.

Explicit CA-state threading (NOT a hidden buffer)
-------------------------------------------------
The NCA is recurrent: each control step advances a ``(B,4,8,8)`` cellular grid.
A hidden recurrent buffer would silently corrupt the PPO update — PPO re-scores
*stored* observations in **shuffled minibatches** during its update epochs, so a
buffer holding "whatever state is current" would be applied to the wrong
transition and ``newlogprob``/``newvalue`` would not match the rollout (the ratio
becomes meaningless and the gradient is wrong).

So the CA grid is threaded **explicitly** and stored by PPO per step:

    initial_state(batch_size, device) -> ca_state                  # (B,4,8,8) zeros
    get_action_and_value(obs, ca_state, action=None)
        -> (action, logprob, entropy, value, next_ca_state)
    mean_action(obs, ca_state) -> (action, next_ca_state)          # eval / A/B
    get_value(obs, ca_state) -> value                              # drop-in helper

``ca_state`` is an **opaque, detached, per-env input** that PPO stores alongside
``(obs, action)`` and resets per-env on ``done`` (to ``initial_state``). Because
the NCA transition is deterministic and the *realized* input state is stored for
every step, the update can shuffle minibatches freely — feeding the stored
``(obs, ca_state, action)`` reproduces the rollout distribution exactly, with no
truncated-BPTT / sequence chunking. This is a **myopic stored-state policy
gradient**: gradients flow through the conv weights for the *current* step only,
not back through the CA across steps (``next_ca_state`` is returned detached).
Full BPTT through the CA is a later upgrade. (A state-threading PPO is also the
more reusable shape for the connectome endgame, where the real sub-circuit is
itself stateful.)

Feeler input gain (applied here, not in the obs)
------------------------------------------------
Per the env-session contract, ``obs`` arrives with the feeler channels (4-5) in
**raw [0,1]**. The warm-start gait is bang-bang (motor cells pinned at the +/-1
clamp), so an unamplified [0,1] feeler cannot move a motor cell until the feeler
weights grow large; the policy multiplies obs channels 4-5 by
``feeler_input_gain`` (default 8.0) before forming the conv1 input, giving small
feeler weights leverage. This matches N-A's ``feeler_input_gain`` knob, which was
applied in ``_make_nav_policy``; here it lives in the policy and stays a tunable
knob we ablate in calibration. The **odor** channels (2-3) are NOT amplified.

A/B integrity
-------------
``warm_start_from_chemo`` loads the trained 8-input chemotaxis controller into the
nav NCA and **zero-inits the two feeler input channels** (handled by
``NCA.warm_start_from_chemo``). With the feeler weights at zero, the feeler obs
contributes exactly nothing through conv1 *regardless of the gain*, so a
feeler-zeroed ``mean_action`` reproduces the chemotaxis forward pass bit-for-bit
until training moves the feeler weights off zero. The value head and ``log_std``
are fresh trainable params and are left untouched by the warm start.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from ..nca import (
    CHANNELS,
    CHEMO_N_PARAMS,
    GRID_H,
    GRID_W,
    MOTOR_COLS,
    MOTOR_ROWS,
    NCA,
)

# The trained chemotaxis controller (CH-A web export) is the seeking prior. It is
# a heavy artifact kept local/gitignored (see CLAUDE.md cross-machine workflow),
# so it may be absent on a cockpit machine; warm_start_from_chemo takes an
# explicit path so a caller can point at a synthesized vector for the A/B gate.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHEMO_JSON = REPO_ROOT / "outputs" / "web_data_ch" / "chemotaxis_controller.json"

# obs layout (the env-session contract): (B, 6, 8, 8) float32.
#   channels 0-1 = proprio, 2-3 = odor (goal beacon), 4-5 = feeler (raw [0,1]).
OBS_CHANNELS = 6
FEELER_OBS_SLICE = slice(4, 6)
DEFAULT_FEELER_INPUT_GAIN = 8.0


class NCAPolicy(nn.Module):
    """``NCA(nav=True)`` as a state-threaded Gaussian PPO policy.

    Parameters
    ----------
    nca:
        An ``NCA(nav=True)`` to wrap. If ``None``, a fresh one is constructed.
    value_hidden:
        Hidden width of the value-head MLP (on the pooled 4-channel grid).
    feeler_input_gain:
        Multiplier applied to obs channels 4-5 before conv1 (tunable knob).
    log_std_init:
        Initial value of the learned global log-std (cleanrl uses 0 -> std 1).
    """

    def __init__(
        self,
        nca: NCA | None = None,
        value_hidden: int = 64,
        feeler_input_gain: float = DEFAULT_FEELER_INPUT_GAIN,
        log_std_init: float = 0.0,
    ) -> None:
        super().__init__()
        if nca is None:
            nca = NCA(nav=True)
        if not nca.nav:
            raise ValueError("NCAPolicy requires an NCA(nav=True) controller")
        self.nca = nca
        # The NCA freezes its params at construction (CMA-ES path); PPO trains
        # them, so re-enable grad. Additive — nca.py itself is untouched.
        for p in self.nca.parameters():
            p.requires_grad_(True)
        self.gain = nca.gain
        self.feeler_input_gain = float(feeler_input_gain)

        # N_motor read from the model's motor readout, not a literal.
        with torch.no_grad():
            probe = torch.zeros(1, CHANNELS, GRID_H, GRID_W)
            self.n_motor = int(self.nca.motor_targets(probe).size)

        # Learned global log-std: one per motor, broadcast over the batch.
        self.actor_logstd = nn.Parameter(
            torch.full((1, self.n_motor), float(log_std_init))
        )
        # Value head: small MLP on the spatially-pooled (B,4) post-step grid.
        self.value_head = nn.Sequential(
            nn.Linear(CHANNELS, value_hidden),
            nn.Tanh(),
            nn.Linear(value_hidden, 1),
        )

    # ----------------------------------------------------------------- state
    def initial_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Episode-start CA grid: ``(B, 4, 8, 8)`` zeros (reset per-env on done)."""
        return torch.zeros(
            batch_size, CHANNELS, GRID_H, GRID_W, dtype=torch.float32, device=device
        )

    # --------------------------------------------------------------- forward
    def _step(
        self, obs: torch.Tensor, ca_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One NCA tick. Returns (action_mean, value, next_ca_state).

        Replicates ``NCA.step``'s math (cat -> tanh(gain*conv1) -> conv2 ->
        clamp) directly on conv1/conv2 so it is batched (NCA.step is pinned to
        B=1) and bit-identical to ``NCA.step`` for B=1.
        """
        w_dtype = self.nca.conv1.weight.dtype
        obs = obs.to(dtype=w_dtype)
        ca_state = ca_state.to(dtype=w_dtype)
        if obs.dim() != 4 or obs.shape[1] != OBS_CHANNELS:
            raise ValueError(
                f"NCAPolicy expected obs of shape (B, {OBS_CHANNELS}, "
                f"{GRID_H}, {GRID_W}), got {tuple(obs.shape)}"
            )
        # Feeler input gain applied here (obs arrives raw [0,1] in ch 4-5). Odor
        # (ch 2-3) is left at raw scale. Out-of-place so the caller's obs is
        # unchanged; A/B is preserved exactly because the feeler weights are zero
        # at warm start (gain * 0-weight contribution == 0).
        if self.feeler_input_gain != 1.0:
            gained = obs.clone()
            gained[:, FEELER_OBS_SLICE] = gained[:, FEELER_OBS_SLICE] * self.feeler_input_gain
            obs = gained

        x = torch.cat([ca_state, obs], dim=1)  # (B, 4 + 6 = 10, 8, 8)
        h = torch.tanh(self.gain * self.nca.conv1(x))
        out = self.nca.conv2(h)
        next_state = torch.clamp(out, -1.0, 1.0)

        # Action mean = channel-0 7x6 motor block, row-major (== motor_targets).
        action_mean = next_state[:, 0, :MOTOR_ROWS, :MOTOR_COLS].reshape(
            next_state.shape[0], -1
        )
        # Value = MLP on the spatially-pooled post-step grid (B, 4).
        pooled = next_state.mean(dim=(2, 3))
        value = self.value_head(pooled)
        return action_mean, value, next_state

    # --------------------------------------------------------- cleanrl API
    def get_value(self, obs: torch.Tensor, ca_state: torch.Tensor) -> torch.Tensor:
        _, value, _ = self._step(obs, ca_state)
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        ca_state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample (or score) an action and return the value + next CA state.

        Returns ``(action, logprob, entropy, value, next_ca_state)``.
        ``next_ca_state`` is detached (myopic stored-state PG: no BPTT across
        steps). When ``action`` is given (update epochs) the action is scored,
        not resampled, so stored ``(obs, ca_state, action)`` reproduce the
        rollout distribution exactly under minibatch shuffling.
        """
        action_mean, value, next_state = self._step(obs, ca_state)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, value, next_state.detach()

    def mean_action(
        self, obs: torch.Tensor, ca_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic action (the NCA forward, clamped to [-1,1]) for eval/A/B.

        Returns ``(action_mean, next_ca_state)``; ``next_ca_state`` is detached.
        """
        action_mean, _, next_state = self._step(obs, ca_state)
        return action_mean, next_state.detach()

    # ------------------------------------------------------------ warm start
    def warm_start_from_chemo(
        self, path: str | Path = DEFAULT_CHEMO_JSON
    ) -> dict[str, int]:
        """Load the trained 8-input chemotaxis controller into the nav NCA.

        Reads ``"flat_params"`` from the controller JSON (same key
        ``evolve_navigation.load_chemo_params`` uses), validates it against the
        chemo param count, and delegates to ``NCA.warm_start_from_chemo`` (which
        copies conv1[0:8] + bias + conv2 and **zero-inits the two feeler channels
        8-9**). The value head and ``log_std`` are untouched. Validates the
        resulting nav model against ``NCA(nav=True)``'s param count.

        Returns a small dict ``{"chemo_params", "nav_params"}`` for reporting.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        vec = np.asarray(data["flat_params"], dtype=np.float64)
        if vec.size != CHEMO_N_PARAMS:
            raise ValueError(
                f"warm_start_from_chemo: expected {CHEMO_N_PARAMS} chemo "
                f"flat_params in {path}, got {vec.size}"
            )
        self.nca.warm_start_from_chemo(vec)
        expected_nav = NCA(nav=True).n_params
        if self.nca.n_params != expected_nav:
            raise ValueError(
                f"warm_start_from_chemo: nav model has {self.nca.n_params} params, "
                f"expected {expected_nav} (NCA(nav=True))"
            )
        return {"chemo_params": int(vec.size), "nav_params": int(self.nca.n_params)}
