"""N-RL-1b gate: NCAPolicy A/B integrity + cleanrl interface contract.

Pure NCA/torch-level checks (no flygym sim, no CMA-ES), so this runs on the
cockpit Mac. The env-level / real-sim A/B checks (warm-start homing, obstacle
collision) live in scripts/test_navigation.py and must run on WIN where flygym
and the trained outputs/web_data_ch/chemotaxis_controller.json are present.

The trained chemotaxis controller is a heavy, gitignored artifact and is NOT on
this Mac. Bit-exact A/B is **weight-agnostic** (it holds for any chemo weight
vector because the feeler input channels are zeroed at warm start), so this gate
synthesizes a deterministic chemo param vector when the real JSON is absent, and
uses the real one if present. Both paths exercise the identical code.

Run from repo root:

    uv run python scripts/test_rl_policy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cellular_gaits.nca import CHEMO_N_PARAMS, NCA
from cellular_gaits.rl.policies import DEFAULT_CHEMO_JSON, NCAPolicy

SCRATCH = ROOT / "scratch" / "nrl"
GATE_SEED = 0


def _chemo_json() -> tuple[Path, bool, np.ndarray]:
    """Return (path, is_real, flat_params). Synthesize if the real one is absent."""
    if DEFAULT_CHEMO_JSON.exists():
        vec = np.asarray(
            json.loads(DEFAULT_CHEMO_JSON.read_text())["flat_params"], dtype=np.float64
        )
        return DEFAULT_CHEMO_JSON, True, vec
    rng = np.random.RandomState(GATE_SEED)
    vec = rng.uniform(-0.5, 0.5, CHEMO_N_PARAMS).astype(np.float64)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    synth = SCRATCH / "synthetic_chemo_controller.json"
    synth.write_text(json.dumps({"flat_params": vec.tolist()}))
    return synth, False, vec


def _sample_sensors(seed: int = GATE_SEED):
    rng = np.random.RandomState(seed)
    ja = rng.uniform(-1, 1, 42)
    fc = (rng.uniform(0, 1, 6) > 0.5).astype(float)
    return ja, fc


def run_gate() -> dict:
    results: dict = {}
    chemo_path, is_real, chemo_vec = _chemo_json()
    results["chemo_source"] = {"path": str(chemo_path), "is_real": is_real}

    # --- warm-start + param-count check -----------------------------------
    policy = NCAPolicy(feeler_input_gain=8.0)
    counts = policy.warm_start_from_chemo(chemo_path)
    assert counts["chemo_params"] == CHEMO_N_PARAMS == 1236, counts
    assert counts["nav_params"] == NCA(nav=True).n_params == 1524, counts
    results["param_check"] = counts

    # --- A/B: feeler-zeroed mean_action == chemo forward, bit-exact -------
    cm = NCA(chemo=True)
    cm.set_params(chemo_vec)
    ja, fc = _sample_sensors()
    cL, cR = 0.3, 0.7

    state0 = policy.initial_state(1)  # (1,4,8,8) zeros
    # chemo reference forward from the SAME zero state.
    sc = NCA.build_chemo_sensor_map(ja, fc, cL, cR)
    with torch.no_grad():
        chemo_next = cm.step(state0.clone(), sc)
    chemo_action = chemo_next[0, 0, :7, :6].reshape(-1).double()

    # nav obs: raw feeler values (policy applies the gain internally).
    obs_zero_feel = NCA.build_nav_sensor_map(ja, fc, cL, cR, 0.0, 0.0)
    obs_on_feel = NCA.build_nav_sensor_map(ja, fc, cL, cR, 1.0, 0.6)
    with torch.no_grad():
        a_zero, _ = policy.mean_action(obs_zero_feel, state0.clone())
        a_on, _ = policy.mean_action(obs_on_feel, state0.clone())
    a_zero = a_zero[0].double()
    a_on = a_on[0].double()

    d_zero = float((a_zero - chemo_action).abs().max())
    d_on = float((a_on - chemo_action).abs().max())
    assert d_zero == 0.0, f"A/B (feeler-zeroed) mismatch: {d_zero}"
    # Feelers ON (raw 1.0/0.6, x gain 8.0) must STILL match chemo at warm start
    # because the feeler weights are zero.
    assert d_on == 0.0, f"A/B (feeler-on@warmstart) mismatch: {d_on}"
    results["ab_max_abs_delta"] = {"feeler_zeroed": d_zero, "feeler_on_warmstart": d_on}

    # --- feeler-input-gain invariance at warm start -----------------------
    p_g1 = NCAPolicy(feeler_input_gain=1.0)
    p_g1.warm_start_from_chemo(chemo_path)
    p_g64 = NCAPolicy(feeler_input_gain=64.0)
    p_g64.warm_start_from_chemo(chemo_path)
    with torch.no_grad():
        a1, _ = p_g1.mean_action(obs_on_feel, state0.clone())
        a64, _ = p_g64.mean_action(obs_on_feel, state0.clone())
    d_gain = float((a1[0].double() - a64[0].double()).abs().max())
    assert d_gain == 0.0, f"gain invariance broken at warm start: {d_gain}"
    results["gain_invariance_max_abs_delta"] = d_gain

    # --- value head + log_std present and finite --------------------------
    with torch.no_grad():
        v = policy.get_value(obs_on_feel, state0.clone())
    logstd = policy.actor_logstd.detach()
    assert torch.isfinite(v).all(), v
    assert torch.isfinite(logstd).all(), logstd
    results["value_logstd"] = {
        "value": float(v.reshape(-1)[0]),
        "value_shape": tuple(v.shape),
        "log_std_shape": tuple(logstd.shape),
        "log_std_finite": bool(torch.isfinite(logstd).all()),
    }

    # --- cleanrl interface contract (shapes + stored-state reproducibility)
    B = 4
    ca = policy.initial_state(B)
    assert tuple(ca.shape) == (B, 4, 8, 8), ca.shape
    obs_b = obs_on_feel.repeat(B, 1, 1, 1)
    with torch.no_grad():
        action, logprob, entropy, value, next_ca = policy.get_action_and_value(obs_b, ca)
    assert tuple(action.shape) == (B, policy.n_motor) == (B, 42), action.shape
    assert tuple(logprob.shape) == (B,), logprob.shape
    assert tuple(entropy.shape) == (B,), entropy.shape
    assert tuple(value.shape) == (B, 1), value.shape
    assert tuple(next_ca.shape) == (B, 4, 8, 8), next_ca.shape
    assert not next_ca.requires_grad, "next_ca_state must be detached (myopic PG)"
    # Re-scoring the stored (obs, ca_state, action) reproduces the rollout
    # distribution exactly under shuffling (the whole point of stored-state PG).
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        _, lp2, _, v2, _ = policy.get_action_and_value(
            obs_b[perm], ca[perm], action[perm]
        )
    d_lp = float((lp2 - logprob[perm]).abs().max())
    d_v = float((v2 - value[perm]).abs().max())
    assert d_lp == 0.0, f"stored-state logprob not reproducible under shuffle: {d_lp}"
    assert d_v == 0.0, f"stored-state value not reproducible under shuffle: {d_v}"
    mean_a, _ = policy.mean_action(obs_b, ca)
    assert tuple(mean_a.shape) == (B, 42), mean_a.shape
    results["interface"] = {
        "action_shape": list(action.shape),
        "logprob_shape": list(logprob.shape),
        "entropy_shape": list(entropy.shape),
        "value_shape": list(value.shape),
        "next_ca_shape": list(next_ca.shape),
        "next_ca_detached": True,
        "shuffled_rescore_logprob_delta": d_lp,
        "shuffled_rescore_value_delta": d_v,
    }

    # --- grad flows to the conv weights (PPO can train them) --------------
    pol = NCAPolicy(feeler_input_gain=8.0)
    pol.warm_start_from_chemo(chemo_path)
    ca1 = pol.initial_state(1)
    act, lp, ent, val, _ = pol.get_action_and_value(obs_on_feel, ca1)
    loss = lp.sum() + val.sum()
    loss.backward()
    g_conv1 = pol.nca.conv1.weight.grad
    g_logstd = pol.actor_logstd.grad
    g_value = pol.value_head[0].weight.grad
    assert g_conv1 is not None and torch.isfinite(g_conv1).all() and g_conv1.abs().sum() > 0
    assert g_logstd is not None and torch.isfinite(g_logstd).all()
    assert g_value is not None and torch.isfinite(g_value).all() and g_value.abs().sum() > 0
    results["grad_flow"] = {
        "conv1_grad_nonzero": bool(g_conv1.abs().sum() > 0),
        "value_head_grad_nonzero": bool(g_value.abs().sum() > 0),
        "log_std_grad_present": True,
    }

    return results


def main() -> None:
    r = run_gate()
    print("N-RL-1b policy gate — PASS")
    print(f"  chemo source: {r['chemo_source']['path']} (real={r['chemo_source']['is_real']})")
    print(f"  param check:  chemo={r['param_check']['chemo_params']} -> nav={r['param_check']['nav_params']}")
    print(f"  A/B max|delta|: feeler-zeroed={r['ab_max_abs_delta']['feeler_zeroed']:.2e}  "
          f"feeler-on@warmstart={r['ab_max_abs_delta']['feeler_on_warmstart']:.2e}")
    print(f"  gain invariance max|delta|: {r['gain_invariance_max_abs_delta']:.2e}")
    print(f"  shuffled re-score delta: logprob={r['interface']['shuffled_rescore_logprob_delta']:.2e} "
          f"value={r['interface']['shuffled_rescore_value_delta']:.2e}")
    print(f"  value finite={r['value_logstd']['log_std_finite']} log_std_shape={r['value_logstd']['log_std_shape']}")


if __name__ == "__main__":
    main()
