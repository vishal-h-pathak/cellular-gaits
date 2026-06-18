"""CA-side criticality metrics for the gain sweep.

The NCA grid is an *autonomous* map: ``s_{t+1} = nca.step(s_t)`` depends only
on the grid, never on the body (v1 is open-loop). So order-vs-chaos is a
property of the rule + gain alone and can be measured by ticking the grid
without running MuJoCo. Two complementary metrics, both with the early
transient discarded:

  - **change_rate** — mean per-tick state-change magnitude (mean absolute
    delta per cell-channel). Near-zero when the grid has settled to a fixed
    point / static pattern; larger when it churns.

  - **lambda** — a poor-man's (largest) Lyapunov exponent via the standard
    twin-trajectory + renormalization method: run a reference grid and a twin
    started ``eps`` away, and each tick measure how far they have separated,
    accumulate ``log(d_t / eps)``, then rescale the twin back to distance
    ``eps`` along the separation direction. The per-tick average is lambda.
    lambda < 0 → perturbations shrink (ordered); lambda > 0 → they grow
    (chaotic); lambda ~ 0 → the edge of chaos.

Both share one warmup so the same transient is dropped from each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .nca import NCA

DEFAULT_N_STEPS = 750  # matches env.DEFAULT_N_STEPS (3 s rollout @ 250 Hz)
DEFAULT_WARMUP = 100
DEFAULT_EPS = 1e-4
_FLOOR = 1e-30  # keeps log finite if a contracting map collapses the twin


@dataclass
class CriticalityResult:
    change_rate: float
    lyapunov: float
    n_counted: int  # ticks averaged over (after warmup)


def measure_criticality(
    nca: NCA,
    n_steps: int = DEFAULT_N_STEPS,
    warmup: int = DEFAULT_WARMUP,
    eps: float = DEFAULT_EPS,
    seed: int = 0,
    perturb_seed: int = 1234,
) -> CriticalityResult:
    """Tick the autonomous CA and return change_rate + Lyapunov lambda.

    ``nca.gain`` is read as-is, so configure it before calling. ``seed``
    matches the rollout's CA init seed so the metric reflects the same
    trajectory that drives the legs.
    """
    if warmup >= n_steps:
        raise ValueError(f"warmup ({warmup}) must be < n_steps ({n_steps})")

    g = torch.Generator().manual_seed(perturb_seed)

    ref = NCA.init_state(seed=seed)
    delta0 = torch.empty_like(ref)
    delta0.uniform_(-1.0, 1.0, generator=g)
    delta0 = delta0 / delta0.norm() * eps
    twin = ref + delta0

    change_sum = 0.0
    change_n = 0
    log_sum = 0.0
    log_n = 0

    with torch.no_grad():
        for t in range(n_steps):
            nxt = nca.step(ref)
            twin = nca.step(twin)

            if t >= warmup:
                change_sum += (nxt - ref).abs().mean().item()
                change_n += 1

            ref = nxt
            d = (twin - ref).norm().item()
            d = max(d, _FLOOR)
            if t >= warmup:
                log_sum += math.log(d / eps)
                log_n += 1
            # renormalize the twin back to distance eps along the separation
            twin = ref + (twin - ref) * (eps / d)

    change_rate = change_sum / change_n if change_n else 0.0
    lyapunov = log_sum / log_n if log_n else 0.0
    return CriticalityResult(
        change_rate=float(change_rate),
        lyapunov=float(lyapunov),
        n_counted=log_n,
    )
