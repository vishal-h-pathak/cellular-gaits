"""Clean Python API over the vendored Shiu connectome LIF brain.

This is a *thin* wrapper. It does not reimplement the leaky-integrate-and-fire
model: it loads the FlyWire v783 connectome, calls the published
`_shiu.model` functions to build the Brian2 network, and exposes an
addressable, by-FlyWire-ID interface:

    brain = BrainModel.load(release="783")
    brain.activate([id1, id2, ...], hz=150)     # Poisson drive at a rate
    brain.silence([id3, ...])                    # zero a neuron's synapses
    rates = brain.run(1.0)                        # -> {flywire_id: rate_hz}

The four-part embodied loop (sense -> brain -> descending neurons -> body)
addresses real neurons by FlyWire ID; this class is the *brain* layer.

Persistent state (for the EB-1 closed loop)
--------------------------------------------
A Brian2 `Network` keeps the membrane state (`v`, `g`) of every neuron between
successive `net.run()` calls. `BrainModel` keeps that Network alive on the
instance, so calling `.run()` repeatedly *advances* the same brain in short
windows rather than restarting it — exactly what a ~15 ms brain<->body sync
needs. `.run(window_s)` returns the firing rate measured *within that window*
(via per-neuron spike-count deltas), so you can drive the body from one window
and feed the next window's sensory activation back in. See `.step()` and the
`brain_explainer.md` note for how the looming->escape loop will use this.

What is faithful vs. what this wrapper adds
-------------------------------------------
- The LIF dynamics, synapse model, weights and the activate/silence mechanics
  come verbatim from `_shiu.model` (`create_model`, `poi`, `silence`). Unchanged.
- This wrapper only: resolves FlyWire IDs <-> Brian indices, lets you choose the
  activation rate per call, keeps the Network persistent, and reads windowed
  rates. It is a single-process, single-trial driver (the upstream `run_exp`
  averages 30 parallel trials to disk; here we expose one live network you can
  step). Average across calls yourself if you want trial statistics.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from brian2 import Network, Hz, second

from ._shiu import model as _shiu

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "_shiu" / "data"

_RELEASE_FILES = {
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
    # The vendored repo also ships the 630 release; only 783 data is bundled.
    "630": ("2023_03_23_completeness_630_final.csv",
            "2023_03_23_connectivity_630_final.parquet"),
}


def _as_id_list(ids) -> list[int]:
    """Normalize a single id or an iterable of ids to a list[int]."""
    if isinstance(ids, (int, np.integer)):
        return [int(ids)]
    return [int(i) for i in ids]


class BrainModel:
    """An addressable, steppable connectome LIF brain (FlyWire v783)."""

    def __init__(self, df_comp: pd.DataFrame, path_comp: Path, path_con: Path,
                 params: dict, release: str):
        self._df_comp = df_comp
        self._path_comp = str(path_comp)
        self._path_con = str(path_con)
        self.params = params
        self.release = release

        # FlyWire ID <-> Brian index maps (Brian index == row order in df_comp)
        self._flyid2i: dict[int, int] = {f: i for i, f in enumerate(df_comp.index)}
        self._i2flyid: dict[int, int] = {i: f for f, i in self._flyid2i.items()}

        # pending configuration (applied at next build)
        self._activations: dict[float, set[int]] = {}  # hz -> set of indices
        self._silenced: set[int] = set()

        # live Brian2 objects (lazily built)
        self._net: Network | None = None
        self._neu = None
        self._syn = None
        self._spk_mon = None
        self._dirty = True  # config changed since last build

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, release: str = "783", *, params: dict | None = None,
             data_dir: str | Path | None = None) -> "BrainModel":
        """Load the connectome for a release and return a brain.

        Parameters
        ----------
        release : "783" (default) or "630"
        params  : optional override of the LIF parameter dict
                  (defaults to a copy of the published `default_params`)
        data_dir: optional directory holding the comp CSV + connectivity parquet
                  (defaults to the vendored `_shiu/data`)
        """
        if release not in _RELEASE_FILES:
            raise ValueError(f"unknown release {release!r}; "
                             f"known: {sorted(_RELEASE_FILES)}")
        data_dir = Path(data_dir) if data_dir is not None else _DATA
        comp_name, con_name = _RELEASE_FILES[release]
        path_comp = data_dir / comp_name
        path_con = data_dir / con_name

        if not path_comp.is_file():
            raise FileNotFoundError(f"missing neuron list {path_comp}")
        if not path_con.is_file():
            raise FileNotFoundError(
                f"missing connectivity {path_con}\n"
                f"It is gitignored (~97 MB). Fetch it with:\n"
                f"    bash {_HERE / '_shiu' / 'fetch_data.sh'}")

        df_comp = pd.read_csv(path_comp, index_col=0)
        p = copy(_shiu.default_params) if params is None else params
        return cls(df_comp, path_comp, path_con, p, release)

    # ------------------------------------------------------------- addressing
    @property
    def n_neurons(self) -> int:
        return len(self._df_comp)

    def has(self, flywire_id: int) -> bool:
        """Does this FlyWire ID resolve in the connectome?"""
        return int(flywire_id) in self._flyid2i

    def resolve(self, ids) -> list[int]:
        """Map FlyWire IDs to Brian indices, raising on any unknown id."""
        ids = _as_id_list(ids)
        missing = [i for i in ids if i not in self._flyid2i]
        if missing:
            raise KeyError(f"{len(missing)} FlyWire ID(s) not in release "
                           f"{self.release}: {missing[:10]}"
                           f"{' ...' if len(missing) > 10 else ''}")
        return [self._flyid2i[i] for i in ids]

    # -------------------------------------------------------- configuration
    def activate(self, flywire_ids, hz: float | None = None) -> "BrainModel":
        """Drive a set of neurons with Poisson input at `hz`.

        Mirrors the upstream Poisson activation (optogenetic-like). `hz`
        defaults to the model's `r_poi`. Multiple `activate` calls at the same
        `hz` accumulate; calls at different `hz` form distinct rate groups
        (the upstream model supports up to two simultaneous rates).
        """
        idx = self.resolve(flywire_ids)
        rate = float(self.params["r_poi"] / Hz) if hz is None else float(hz)
        self._activations.setdefault(rate, set()).update(idx)
        self._dirty = True
        return self

    def silence(self, flywire_ids) -> "BrainModel":
        """Silence neurons by zeroing all synapses from them (upstream semantics)."""
        self._silenced.update(self.resolve(flywire_ids))
        self._dirty = True
        return self

    def clear(self) -> "BrainModel":
        """Forget all pending activations/silencings and drop the live network."""
        self._activations.clear()
        self._silenced.clear()
        self._teardown()
        self._dirty = True
        return self

    # ---------------------------------------------------------------- build
    def _teardown(self) -> None:
        self._net = self._neu = self._syn = self._spk_mon = None

    def _build(self) -> None:
        """Construct the live Brian2 network from the current configuration.

        Uses the vendored `create_model`/`poi`/`silence` so the LIF dynamics are
        untouched; only the per-group activation rate is chosen here.
        """
        groups = sorted(self._activations.items())  # [(hz, {idx})...]
        if len(groups) > 2:
            raise NotImplementedError(
                "the upstream model supports at most two distinct activation "
                f"rates; got {len(groups)}. Activate at one or two rates.")

        params = copy(self.params)
        exc: list[int] = []
        exc2: list[int] = []
        if len(groups) >= 1:
            params["r_poi"] = groups[0][0] * Hz
            exc = sorted(groups[0][1])
        if len(groups) == 2:
            params["r_poi2"] = groups[1][0] * Hz
            exc2 = sorted(groups[1][1])

        neu, syn, spk_mon = _shiu.create_model(self._path_comp, self._path_con, params)
        poi_inp, neu = _shiu.poi(neu, exc, exc2, params)
        syn = _shiu.silence(sorted(self._silenced), syn)

        self._neu, self._syn, self._spk_mon = neu, syn, spk_mon
        self._net = Network(neu, syn, spk_mon, *poi_inp)
        self._dirty = False

    # ------------------------------------------------------------------ run
    def run(self, duration_s: float, *, query: Iterable[int] | None = None,
            with_spikes: bool = False, rebuild: bool = False):
        """Advance the brain by `duration_s` and return per-neuron firing rates.

        Returns ``{flywire_id: rate_hz}`` for every neuron that fired in this
        window. The network is built on first use and then *persists*: calling
        `.run()` again advances the same brain (membrane state carries over),
        so successive calls measure successive windows. Pass ``rebuild=True``
        (or call `.activate()`/`.silence()`/`.clear()`) to start fresh.

        Parameters
        ----------
        duration_s : window length in seconds
        query      : if given, also guarantee these FlyWire IDs appear in the
                     result (with rate 0.0 if they did not fire)
        with_spikes: if True, return ``(rates, spikes)`` where `spikes` is
                     ``{flywire_id: [spike_times_s]}`` within this window
        rebuild    : force a fresh network before running
        """
        if rebuild or self._net is None or self._dirty:
            self._teardown()
            self._build()

        spk = self._spk_mon
        pre_count = np.asarray(spk.count[:], dtype=np.int64).copy()
        t_start = float(self._net.t / second)
        self._net.run(duration_s * second)
        post_count = np.asarray(spk.count[:], dtype=np.int64)

        delta = post_count - pre_count
        fired = np.nonzero(delta)[0]
        rates = {self._i2flyid[int(i)]: float(delta[i]) / duration_s for i in fired}

        if query is not None:
            for fid in _as_id_list(query):
                rates.setdefault(fid, 0.0)

        if not with_spikes:
            return rates

        spikes: dict[int, list[float]] = {}
        trains = spk.spike_trains()
        for i in fired:
            t = np.asarray(trains[int(i)] / second, dtype=float)
            t = t[t >= t_start - 1e-12]
            if len(t):
                spikes[self._i2flyid[int(i)]] = t.tolist()
        return rates, spikes

    def step(self, window_s: float = 0.015, **kw):
        """Alias for `run` framed as a brain<->body sync step (default 15 ms).

        Intended for the EB-1 closed loop: advance the persistent brain one
        sync window and read the rates that drive descending-neuron outputs.
        """
        return self.run(window_s, **kw)

    @property
    def t_s(self) -> float:
        """Current simulation time of the live network, in seconds (0 if unbuilt)."""
        return 0.0 if self._net is None else float(self._net.t / second)

    def __repr__(self) -> str:
        return (f"BrainModel(release={self.release!r}, n_neurons={self.n_neurons}, "
                f"activated={sum(len(s) for s in self._activations.values())}, "
                f"silenced={len(self._silenced)}, t={self.t_s:.3f}s)")
