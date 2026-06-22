# N-RL-1b gate — NCAPolicy A/B integrity + cleanrl interface

**Slice:** `src/cellular_gaits/rl/policies.py` (the NCA wrapped as a PPO-ready
Gaussian policy). Built on branch `feat/n-rl-policy`. No env, no training loop.

**Reproduce:** `uv run python scripts/test_rl_policy.py`
(machine-readable results: `scratch/nrl/gate_ab_results.json`)

**Result: PASS — A/B max|Δ| = 0 (bit-exact).**

---

## 1. `get_action_and_value` signature (as implemented)

State is threaded **explicitly** (not hidden in a buffer), so PPO can store the
realized CA grid per step and shuffle minibatches freely without corrupting the
update. The full Agent interface (1c / integrate code against exactly this):

```python
initial_state(batch_size, device=None) -> ca_state          # (B,4,8,8) float32 zeros
get_action_and_value(obs, ca_state, action=None)
    -> (action, logprob, entropy, value, next_ca_state)
mean_action(obs, ca_state) -> (action, next_ca_state)        # eval / A/B
get_value(obs, ca_state) -> value                            # drop-in helper
```

- `obs`: `(B, 6, 8, 8)` float32 — ch 0-1 proprio, 2-3 odor, **4-5 feeler (raw
  [0,1])**. The policy applies `feeler_input_gain` (default **8.0**) to ch 4-5
  internally before conv1; odor is left at raw scale.
- `action`: `(B, 42)` float32 (`N_motor` read from the model's motor readout, not
  a literal). `action_mean` is the NCA forward clamped to [-1,1]; samples are
  drawn unclamped (cleanrl convention — the env clips).
- `value`: `(B, 1)`; `logprob`, `entropy`: `(B,)`.
- `next_ca_state`: `(B,4,8,8)`, **detached** → myopic stored-state policy
  gradient (gradients flow through conv weights for the current step only, no
  BPTT through the CA across steps). Documented as the right first cut for a
  warm-started gait; full BPTT is a later upgrade.

Observed shapes (B=4): action `[4, 42]`, logprob `[4]`, entropy `[4]`, value
`[4, 1]`, next_ca `[4, 4, 8, 8]`, `next_ca.requires_grad == False`. ✔

**Stored-state reproducibility (the reason for explicit threading):** re-scoring
the stored `(obs, ca_state, action)` under a shuffled permutation reproduces the
rollout distribution exactly — Δlogprob = **0.0**, Δvalue = **0.0**. This is what
a hidden recurrent buffer would have broken.

## 2. Warm-start param-count check

`NCAPolicy.warm_start_from_chemo(path)` reads `"flat_params"` from the chemo
controller JSON (same key as `evolve_navigation.load_chemo_params`), delegates to
`NCA.warm_start_from_chemo` (copies conv1[0:8] + bias + conv2, **zero-inits the
two feeler input channels 8-9**), and validates the result:

| check | value |
|---|---|
| chemo `flat_params` size | **1236** (== `CHEMO_N_PARAMS`) |
| nav model `n_params` after warm start | **1524** (== `NCA(nav=True).n_params`) |

The value head and global `log_std` are **untouched** by the warm start and are
present + finite (`log_std` shape `(1, 42)`, all finite; a sample value is
finite). ✔

## 3. A/B integrity — max|Δ|

With the feelers' weights zero (warm start), `mean_action` reproduces the 8-input
chemotaxis forward pass **bit-for-bit** on shared weights, from the same
(zero) CA state. Same A/B check the N-A calibration used
(`test_navigation.py::test_ab_nca_level`, which asserts exact `d == 0.0`):

| comparison | max\|Δ\| |
|---|---|
| feeler channels = 0, nav `mean_action` vs chemo forward | **0.0** |
| feeler channels ON (raw 1.0/0.6 × gain 8.0) vs chemo forward | **0.0** |
| feeler ON, `feeler_input_gain` 1.0 vs 64.0 (gain invariance) | **0.0** |

The feeler-on rows are the stronger claim: amplifying a feeler reading through
**zero feeler weights** is still exactly zero, so A/B holds *regardless of the
gain knob* — warm-start changes nothing until training moves the feeler weights
off zero.

**Gradient flow (PPO-trainability):** a backward pass produces finite, nonzero
gradients on `conv1.weight` and the value head, and a present gradient on
`log_std`. ✔

---

## Honesty notes

- **Chemo source on this machine.** `outputs/web_data_ch/chemotaxis_controller.json`
  (the trained CH-A forager) is a heavy, gitignored artifact and is **not on this
  Mac cockpit**. Bit-exact A/B is **weight-agnostic** — it holds for *any* chemo
  weight vector because the feeler input channels are zeroed at warm start — so
  this gate synthesized a deterministic chemo param vector
  (`scratch/nrl/synthetic_chemo_controller.json`, seed 0) and ran the identical
  code path. When run on WIN with the real JSON present, `warm_start_from_chemo`
  picks it up automatically and the same gate applies to the trained weights.
- **Sim / env A/B not run here.** The real-sim A/B checks (warm-start homing
  reproduces the chemo rollout; obstacle collision) live in
  `scripts/test_navigation.py` and require flygym + the trained JSON — they must
  run on WIN. Noted, not fake-passed.
- **`nca.py` untouched.** The policy reuses `NCA(nav=True)`'s `conv1`/`conv2`/
  `gain` and replicates `NCA.step`'s math for a batched forward (NCA.step is
  pinned to B=1); for B=1 it is bit-identical. `requires_grad` is re-enabled on
  the NCA params *in the policy* (PPO trains them) — additive, the NCA module is
  not modified.
