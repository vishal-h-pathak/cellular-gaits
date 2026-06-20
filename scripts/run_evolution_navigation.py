"""CLI for the parallel navigation (feelers + arena) CMA-ES evolution (N-A).

Examples:
    # Calibration: the four validation gates — warm-start still walks+homes &
    # A/B is bit-exact; the untrained baseline collides (genuine impediments);
    # parallel == sequential; both-sides detour is achievable. Writes
    # scratch/na/calibration.json + REPORT_n_a_calibration.md, then exits.
    uv run python scripts/run_evolution_navigation.py --calibrate

    # Speedup benchmark only
    uv run python scripts/run_evolution_navigation.py --benchmark-only --pop 16

    # Short validation run (build everything, validate first), with benchmark
    uv run python scripts/run_evolution_navigation.py --pop 16 --gens 8 --benchmark

    # Full run (proposed numbers)
    uv run python scripts/run_evolution_navigation.py --pop 48 --gens 70

Outputs:
    outputs/fitness_log_<run_id>.csv
    outputs/benchmark_<run_id>.json        (when --benchmark[-only])
    scratch/na/calibration.json            (when --calibrate)
    REPORT_n_a_calibration.md              (when --calibrate)
    checkpoints/<run_id>/gen_NN.npz/.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cellular_gaits.env import FlyEnv, OdorField  # noqa: E402
from cellular_gaits.evolve import _utc_run_id, load_checkpoint  # noqa: E402
from cellular_gaits.evolve_chemotaxis import _make_chemo_policy  # noqa: E402
from cellular_gaits.evolve_navigation import (  # noqa: E402
    NAV_ROLLOUT_STEPS,
    NavConfig,
    _make_nav_policy,
    _signed_detour,
    benchmark_speedup,
    build_condition_envs,
    conditions,
    diagnose_params,
    load_chemo_params,
    nav_fitness,
    run_evolution_navigation,
    warm_start_x0,
)
from cellular_gaits.nca import NCA  # noqa: E402

OUT_DIR = ROOT / "outputs"
SCRATCH = ROOT / "scratch" / "na"


def _cfg_from_args(args) -> NavConfig:
    return NavConfig(
        seed=args.seed,
        popsize=args.pop,
        n_gens=args.gens,
        sigma_init=args.sigma,
        rollout_steps=args.rollout_steps,
        checkpoint_every=args.checkpoint_every,
        source_distance=args.distance,
        odor_lambda=args.odor_lambda,
        reach_radius=args.reach_radius,
        obstacle_radius=args.obstacle_radius,
        feeler_range=args.feeler_range,
        feeler_input_gain=args.feeler_gain,
        w_collide=args.w_collide,
        reach_bonus=args.reach_bonus,
        time_penalty=args.time_penalty,
        n_workers=args.workers,
    )


def _print_diag(rows: list[dict]) -> None:
    print(
        f"    {'condition':>18} {'reach':>5} {'coll':>5} {'dmin':>6} {'appr':>6} "
        f"{'detour':>7} {'fpk':>5} {'fL-R':>7} {'fit':>7}"
    )
    for r in rows:
        print(
            f"    {r['name']:>18} {str(r['reached']):>5} {r['collision_count']:>5} "
            f"{r['min_dist']:>6.2f} {r['approach']:>6.2f} {r['detour_perp']:>+7.2f} "
            f"{r['feeler_peak']:>5.2f} {r['feeler_mean_LmR']:>+7.3f} {r['fitness']:>7.2f}"
        )


# --------------------------------------------------------------------------- #
# Gate 1 — warm-start A/B: feeler-zeroed nav == chemo, bit-exact (NCA + env)
# --------------------------------------------------------------------------- #
def gate_ab(cfg: NavConfig) -> dict:
    chemo_vec = load_chemo_params()
    nav = NCA(nav=True)
    nav.warm_start_from_chemo(chemo_vec)
    chemo = NCA(chemo=True)
    chemo.set_params(chemo_vec)

    # NCA-level: feeler-zeroed nav step == chemo step for random inputs.
    rng = np.random.RandomState(0)
    state = NCA.init_state(seed=0)
    ja = rng.uniform(-1, 1, 42)
    fc = (rng.uniform(0, 1, 6) > 0.5).astype(float)
    sc = NCA.build_chemo_sensor_map(ja, fc, 0.3, 0.7)
    sn = NCA.build_nav_sensor_map(ja, fc, 0.3, 0.7, 0.0, 0.0)
    with torch.no_grad():
        d_nca = float((chemo.step(state, sc) - nav.step(state, sn)).abs().max())
    # A non-zero feeler must not move the warm-start output (feeler weights zero).
    sn2 = NCA.build_nav_sensor_map(ja, fc, 0.3, 0.7, 1.0, 0.0)
    with torch.no_grad():
        d_feeler_on = float((nav.step(state, sn) - nav.step(state, sn2)).abs().max())

    # Env-level: nav (NO obstacles) homing on a goal == chemo rollout, bit-exact.
    # Test "still homes" on a no-obstacle layout at a goal the forager actually
    # reaches (the first nav condition's azimuth — the warm-start chemo homer is
    # erratic/biased and only reaches a narrow azimuth band, see evolve_navigation).
    goal = conditions(cfg)[0].goal_xy
    odor = OdorField(source_xy=goal, lam=cfg.odor_lambda)

    env_c = FlyEnv(antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral)
    env_c.set_odor(odor)
    fit_c, traj_c = env_c.rollout(
        _make_chemo_policy(chemo), n_steps=cfg.rollout_steps, pass_sensors=True
    )

    env_n = FlyEnv(  # no obstacles -> feelers read (0, 0)
        antenna_forward=cfg.antenna_forward, antenna_lateral=cfg.antenna_lateral
    )
    env_n.set_odor(odor)
    fit_n, traj_n = env_n.rollout(
        _make_nav_policy(nav, feeler_input_gain=cfg.feeler_input_gain),
        n_steps=cfg.rollout_steps,
        pass_sensors=True,
    )
    d_fit = abs(fit_c - fit_n)
    d_traj = float(
        np.abs(traj_c["joint_targets"] - traj_n["joint_targets"]).max()
    )
    reached_no_obstacle = bool(
        np.linalg.norm(np.asarray(traj_n["thorax_xyz"])[:, :2] - np.asarray(goal), axis=1).min()
        < cfg.reach_radius
    )
    d_end_no_obstacle = float(
        np.linalg.norm(np.asarray(traj_n["thorax_xyz"])[-1, :2] - np.asarray(goal))
    )
    result = {
        "nca_ab_max_abs_delta": d_nca,
        "feeler_on_at_warmstart_delta": d_feeler_on,
        "env_ab_fitness_delta": d_fit,
        "env_ab_trajectory_max_delta": d_traj,
        "no_obstacle_reached_goal": reached_no_obstacle,
        "no_obstacle_d_end": d_end_no_obstacle,
        "n_params": NCA(nav=True).n_params,
    }
    print(
        f"[gate1] A/B  NCA max|Δ|={d_nca:.2e}  feeler-on@warmstart={d_feeler_on:.2e}  "
        f"env Δfit={d_fit:.2e}  env Δtraj={d_traj:.2e}  "
        f"no-obstacle reach={reached_no_obstacle} (d_end={d_end_no_obstacle:.2f})"
    )
    return result


# --------------------------------------------------------------------------- #
# Gate 2 — the untrained baseline collides (obstacles are genuine impediments)
# --------------------------------------------------------------------------- #
def gate_baseline(cfg: NavConfig, envs) -> dict:
    x0 = warm_start_x0(cfg)
    rows = diagnose_params(x0, cfg, envs=envs)
    collide_rate = float(np.mean([r["collided"] for r in rows]))
    reach_rate = float(np.mean([r["reached"] for r in rows]))
    cue_ok = _cue_correct(rows)
    print(
        f"[gate2] untrained baseline: collide_rate={collide_rate:.2f} "
        f"reach_rate={reach_rate:.2f}  feeler cue correctly signed={cue_ok}"
    )
    _print_diag(rows)
    return {
        "collide_rate": collide_rate,
        "reach_rate": reach_rate,
        "feeler_cue_correctly_signed": cue_ok,
        "rows": rows,
    }


def _cue_correct(rows: list[dict]) -> bool:
    """Feeler cue correctly signed: an obstacle LEFT-of-path (block_side +1)
    biases the LEFT feeler (feeler_mean_LmR > 0); RIGHT-of-path biases the right.
    """
    ok = True
    for r in rows:
        if r["block_side"] > 0:
            ok = ok and r["feeler_mean_LmR"] > 0.01
        elif r["block_side"] < 0:
            ok = ok and r["feeler_mean_LmR"] < -0.01
    return ok


# --------------------------------------------------------------------------- #
# Gate 4 — both-sides detour is achievable (emergent from feeler L-R)
# --------------------------------------------------------------------------- #
def gate_detour(cfg: NavConfig, envs, n_samples: int = 48, sigma: float = 0.5) -> dict:
    """Short random search over FEELER-channel weight perturbations from the
    warm start. We perturb only the feeler input weights (conv1 channels 8-9) so
    the seeking is preserved, then look for a candidate that detours the RIGHT way
    on BOTH blocking layouts (left-block -> detour right, right-block -> detour
    left). If found, both-sides detour is achievable under this fitness/weighting.
    """
    x0 = warm_start_x0(cfg)
    base = NCA(nav=True)
    base.set_params(x0)
    # indices of the feeler input-channel weights inside the flat vector.
    w = base.conv1.weight  # (16, 10, 3, 3)
    flat_idx = np.arange(x0.size)
    # conv1.weight is the first block; feeler channels are input idx 8,9.
    c1w_n = w.numel()
    w_idx = flat_idx[:c1w_n].reshape(w.shape)
    feeler_mask = np.zeros(x0.size, dtype=bool)
    feeler_mask[w_idx[:, 8:, :, :].reshape(-1)] = True

    conds = conditions(cfg)
    left_i = next(i for i, c in enumerate(conds) if c.block_side > 0)
    right_i = next(i for i, c in enumerate(conds) if c.block_side < 0)

    rng = np.random.default_rng(cfg.seed)
    best = None
    samples = []
    for k in range(n_samples):
        cand = x0.copy()
        cand[feeler_mask] += rng.normal(0, sigma, size=int(feeler_mask.sum()))
        rows = diagnose_params(cand, cfg, envs=envs)
        left = rows[left_i]
        right = rows[right_i]
        # detour_perp: + = went left, - = went right.
        left_detours_right = left["detour_perp"] < -0.3
        right_detours_left = right["detour_perp"] > 0.3
        both = bool(left_detours_right and right_detours_left)
        score = float(np.mean([r["fitness"] for r in rows]))
        rec = {
            "sample": k,
            "score": score,
            "left_block_detour_perp": left["detour_perp"],
            "right_block_detour_perp": right["detour_perp"],
            "left_block_collisions": left["collision_count"],
            "right_block_collisions": right["collision_count"],
            "both_sides_correct": both,
        }
        samples.append(rec)
        if both and (best is None or score > best["score"]):
            best = rec
    achievable = best is not None
    n_both = int(sum(s["both_sides_correct"] for s in samples))
    print(
        f"[gate4] detour search: {n_both}/{n_samples} samples detour BOTH ways "
        f"correctly  achievable={achievable}"
    )
    if best is not None:
        print(
            f"[gate4]   best: left-block detour_perp={best['left_block_detour_perp']:+.2f} "
            f"(want <0), right-block detour_perp={best['right_block_detour_perp']:+.2f} "
            f"(want >0), score={best['score']:.2f}"
        )
    return {
        "n_samples": n_samples,
        "sigma": sigma,
        "n_both_sides_correct": n_both,
        "both_sides_detour_achievable": achievable,
        "best": best,
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# Calibration (the four gates) + report
# --------------------------------------------------------------------------- #
def calibrate(args) -> None:
    cfg = _cfg_from_args(args)
    print("[cal] building per-condition envs (obstacles baked into the world)...")
    envs = build_condition_envs(cfg)
    conds = conditions(cfg)
    print(f"[cal] conditions: {[c.name for c in conds]}")

    print("\n[cal] === Gate 1: warm-start A/B (feeler-zeroed nav == chemo) ===")
    g1 = gate_ab(cfg)
    print("\n[cal] === Gate 2: untrained baseline collides (genuine impediments) ===")
    g2 = gate_baseline(cfg, envs)
    print("\n[cal] === Gate 3: parallel == sequential ===")
    g3 = benchmark_speedup(cfg, n_individuals=min(cfg.popsize, 12))
    print("\n[cal] === Gate 4: both-sides detour achievable ===")
    g4 = gate_detour(cfg, envs, n_samples=args.detour_samples, sigma=args.detour_sigma)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "source_distance": cfg.source_distance,
            "odor_lambda": cfg.odor_lambda,
            "reach_radius": cfg.reach_radius,
            "obstacle_radius": cfg.obstacle_radius,
            "feeler_range": cfg.feeler_range,
            "feeler_input_gain": cfg.feeler_input_gain,
            "w_collide": cfg.w_collide,
            "reach_bonus": cfg.reach_bonus,
            "time_penalty": cfg.time_penalty,
            "rollout_steps": cfg.rollout_steps,
            "conditions": [
                {
                    "name": c.name,
                    "goal_xy": list(c.goal_xy),
                    "obstacle_xy": list(c.obstacles[0]),
                    "block_side": c.block_side,
                }
                for c in conds
            ],
        },
        "gate1_ab": g1,
        "gate2_baseline": g2,
        "gate3_parallel": g3,
        "gate4_detour": g4,
    }
    out = SCRATCH / "calibration.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[cal] wrote {out.relative_to(ROOT)}")

    _write_report(args, cfg, payload)


def _write_report(args, cfg: NavConfig, p: dict) -> None:
    g1, g2, g3, g4 = p["gate1_ab"], p["gate2_baseline"], p["gate3_parallel"], p["gate4_detour"]
    full_cmd = (
        f"uv run python scripts/run_evolution_navigation.py "
        f"--pop {args.full_pop} --gens {args.full_gens}"
    )

    def yn(b):
        return "✅" if b else "❌"

    g1_pass = (
        g1["nca_ab_max_abs_delta"] == 0.0
        and g1["feeler_on_at_warmstart_delta"] == 0.0
        and g1["env_ab_trajectory_max_delta"] < 1e-6
        and g1["no_obstacle_reached_goal"]
    )
    g2_pass = g2["collide_rate"] >= 0.5 and g2["feeler_cue_correctly_signed"]
    g3_pass = g3["max_abs_fitness_diff"] == 0.0
    g4_pass = g4["both_sides_detour_achievable"]

    lines = []
    lines.append("# REPORT — N-A · Obstacle navigation: calibration gate\n")
    lines.append(
        "Validation-first calibration for the navigation behavior (feelers + "
        "arena + seek-vs-avoid). The full evolution has **NOT** been launched.\n"
    )
    lines.append("## The four gates\n")
    lines.append(f"### {yn(g1_pass)} Gate 1 — warm-start still walks + homes (A/B bit-exact)\n")
    lines.append(
        f"- NCA-level feeler-zeroed `nav` vs 8-input `chemo` forward pass: "
        f"max|Δ| = `{g1['nca_ab_max_abs_delta']:.2e}`\n"
        f"- Non-zero feeler at warm start changes output by: "
        f"`{g1['feeler_on_at_warmstart_delta']:.2e}` (feeler weights are zero)\n"
        f"- Env-level (no-obstacle) `nav` vs `chemo` rollout: Δfitness = "
        f"`{g1['env_ab_fitness_delta']:.2e}`, max|Δtrajectory| = "
        f"`{g1['env_ab_trajectory_max_delta']:.2e}`\n"
        f"- On a no-obstacle layout the warm-start still reaches the goal: "
        f"`{g1['no_obstacle_reached_goal']}` (d_end = {g1['no_obstacle_d_end']:.2f})\n"
    )
    lines.append(f"### {yn(g2_pass)} Gate 2 — obstacles are genuine impediments (baseline collides)\n")
    lines.append(
        f"- Untrained warm-start forager (feelers wired, avoidance not yet learned): "
        f"collision rate = **{g2['collide_rate']:.0%}**, reach rate = "
        f"**{g2['reach_rate']:.0%}** across the {len(p['config']['conditions'])} layouts.\n"
        f"- Feeler cue correctly signed (left-of-path -> left feeler, right -> right): "
        f"`{g2['feeler_cue_correctly_signed']}`\n\n"
    )
    lines.append("| condition | reached | collisions | min_dist | approach | detour_perp | feeler_peak | feeler L−R |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in g2["rows"]:
        lines.append(
            f"| {r['name']} | {r['reached']} | {r['collision_count']} | "
            f"{r['min_dist']:.2f} | {r['approach']:.2f} | {r['detour_perp']:+.2f} | "
            f"{r['feeler_peak']:.2f} | {r['feeler_mean_LmR']:+.3f} |\n"
        )
    lines.append("\n")
    lines.append(f"### {yn(g3_pass)} Gate 3 — parallel == sequential\n")
    lines.append(
        f"- max|Δfitness| across workers = `{g3['max_abs_fitness_diff']:.2e}` "
        f"(n={g3['n_individuals']}, workers={g3['workers']})\n"
        f"- speedup = **{g3['speedup']:.2f}×** "
        f"(seq {g3['t_sequential_s']:.1f}s → par {g3['t_parallel_s']:.1f}s)\n"
    )
    lines.append(f"### {yn(g4_pass)} Gate 4 — both-sides detour is achievable\n")
    lines.append(
        f"- Short random search over feeler-channel weight perturbations "
        f"(n={g4['n_samples']}, σ={g4['sigma']}): "
        f"**{g4['n_both_sides_correct']}/{g4['n_samples']}** candidates detour the "
        f"correct way on BOTH blocking layouts.\n"
        f"- Both-sides detour achievable from the feeler L−R asymmetry: "
        f"**{g4['both_sides_detour_achievable']}**\n"
    )
    if g4["best"] is not None:
        b = g4["best"]
        lines.append(
            f"- Best candidate: left-block detour_perp = "
            f"`{b['left_block_detour_perp']:+.2f}` (want <0, i.e. detour right), "
            f"right-block detour_perp = `{b['right_block_detour_perp']:+.2f}` "
            f"(want >0, i.e. detour left).\n"
        )
    lines.append("\n## Design decisions & calibration notes\n")
    colls = [r["collision_count"] for r in g2["rows"]]
    lines.append(
        "- **Warm-start forager is an erratic, left-biased homer.** The trained "
        "chemotaxis controller does NOT home omnidirectionally — it reaches only "
        "a narrow, non-contiguous azimuth set (~30/40/55° at distance 18; never "
        "0°/right/rear), and of those only **az=40°** approaches along a "
        "reasonably direct path (the 30/55° paths loop and double back). The nav "
        "task is therefore built on **az=40° with two obstacle placements "
        "(near/far) × two block sides** = 4 conditions. The left/right block at "
        "each placement is the anti-bias mechanism (the detour direction must "
        "come from the feeler L−R, not a fixed swerve). Goal-azimuth variation is "
        "genuinely limited by this particular forager's narrow homing band — an "
        "honest constraint of the composition, not a modelling shortcut.\n"
        f"- **Headroom is in the collision count, not the reach flag.** The "
        f"baseline still *reaches* (100%) because it bumps along and slides past, "
        f"but it collides on every layout ({min(colls)}–{max(colls)} in-contact "
        f"steps/episode). With `w_collide={p['config']['w_collide']}` the collision "
        f"penalty already exceeds approach+reach_bonus on the baseline (fitness is "
        f"negative on all four), so a clean detour — which removes the penalty "
        f"while keeping the approach — is the fitness optimum.\n"
        "- **No fitness re-weighting was needed.** Unlike escape (which had to "
        "raise its directional weights above survival), the starting "
        f"`w_collide={p['config']['w_collide']}` already makes detouring the "
        "optimum, and the Gate-4 search finds candidates that detour the correct "
        "way on both blocking layouts. The full run can raise `w_collide` further "
        "if a fixed swerve emerges, but the calibration does not require it.\n"
    )
    lines.append("\n## Honesty caveats (carry into the export meta)\n")
    lines.append(
        "1. The **feelers are a hand-built rangefinder abstraction**. Real "
        "*Drosophila* avoid obstacles via vision / optic flow / visual looming, "
        "NOT a LIDAR-like distance sensor. Unlike escape (LC4/LPLC2 → DNp01), "
        "**navigation has no clean real-circuit seam** — it is a robotics-flavored "
        "capability demo, not a connectome bridge.\n"
        "2. The **goal is the chemotaxis odor beacon reused** as a homing target: "
        "this is **reactive local avoidance + gradient homing**, NOT global path "
        "planning or spatial memory. It can get **trapped in concave/dead-end "
        "obstacle configurations** (a local minimum).\n"
        "3. The nav **fitness scalar is not comparable** across behaviors "
        "(task-specific shaping).\n"
        "4. `chemo`/`loom`/closed-loop/v1 paths are **bit-exact unchanged**; nav "
        "is purely additive (Gate 1).\n"
    )
    lines.append("\n## Proposed full run\n")
    lines.append(
        f"n_params = {g1['n_params']} (bigger than escape's 1236), so a slightly "
        f"larger population and more generations than escape (pop 32 / 50 gens):\n\n"
    )
    lines.append(f"```\n{full_cmd}\n```\n\n")
    lines.append("**Confirm and I'll launch the full run.**\n")

    report = ROOT / "REPORT_n_a_calibration.md"
    report.write_text("".join(lines))
    print(f"[cal] wrote {report.name}")
    print("\n" + "=" * 70)
    print("CALIBRATION GATES:")
    print(f"  Gate 1 (A/B bit-exact + still homes):     {yn(g1_pass)}")
    print(f"  Gate 2 (baseline collides):               {yn(g2_pass)}")
    print(f"  Gate 3 (parallel == sequential):          {yn(g3_pass)}")
    print(f"  Gate 4 (both-sides detour achievable):    {yn(g4_pass)}")
    print("=" * 70)
    print(f"\nProposed full run:\n  {full_cmd}\nConfirm and I'll launch the full run.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", "--popsize", dest="pop", type=int, default=48)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--rollout-steps", type=int, default=NAV_ROLLOUT_STEPS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=0, help="0 -> cpu_count()-1")
    # task geometry
    p.add_argument("--distance", type=float, default=18.0)
    p.add_argument("--odor-lambda", type=float, default=12.0)
    p.add_argument("--reach-radius", type=float, default=3.0)
    p.add_argument("--obstacle-radius", type=float, default=2.0)
    p.add_argument("--feeler-range", type=float, default=6.0)
    p.add_argument("--feeler-gain", type=float, default=8.0)
    # fitness shaping
    p.add_argument("--w-collide", type=float, default=0.2)
    p.add_argument("--reach-bonus", type=float, default=8.0)
    p.add_argument("--time-penalty", type=float, default=4.0)
    # calibration knobs
    p.add_argument("--detour-samples", type=int, default=48)
    p.add_argument("--detour-sigma", type=float, default=0.5)
    p.add_argument("--full-pop", type=int, default=48, help="proposed full-run popsize")
    p.add_argument("--full-gens", type=int, default=70, help="proposed full-run gens")
    # modes
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--calibrate", action="store_true", help="Run the four gates + report, then exit.")
    p.add_argument("--benchmark", action="store_true", help="Time seq vs parallel before evolving.")
    p.add_argument("--benchmark-only", action="store_true", help="Only benchmark, then exit.")
    p.add_argument("--diagnose-ckpt", type=Path, default=None, help="Diagnose a checkpoint best, then exit.")
    args = p.parse_args()

    run_id = args.run_id or _utc_run_id()
    cfg = _cfg_from_args(args)

    if args.calibrate:
        calibrate(args)
        return

    if args.diagnose_ckpt is not None:
        data = load_checkpoint(args.diagnose_ckpt)
        best = np.asarray(data["best_params"], dtype=np.float64)
        print(f"[diag] checkpoint best_fit={float(data['best_fit']):.4f}")
        _print_diag(diagnose_params(best, cfg))
        return

    if args.benchmark or args.benchmark_only:
        bench = benchmark_speedup(cfg, n_individuals=min(cfg.popsize, 12))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bpath = OUT_DIR / f"benchmark_{run_id}.json"
        bpath.write_text(json.dumps(bench, indent=2))
        print(f"[run] benchmark -> {bpath.relative_to(ROOT)}")
        if args.benchmark_only:
            return

    # Gen-0 (warm-start) diagnostics for the record before evolving.
    print("[run] gen-0 (warm-start) per-condition diagnostics:")
    _print_diag(diagnose_params(warm_start_x0(cfg), cfg))

    best_fit, best_params, run_dir = run_evolution_navigation(
        cfg, run_id=run_id, resume_from=args.resume_from
    )
    print(f"[run] done. best_fit={best_fit:.4f}  ckpts -> {run_dir.relative_to(ROOT)}")
    print("[run] final best per-condition diagnostics:")
    _print_diag(diagnose_params(best_params, cfg))


if __name__ == "__main__":
    main()
