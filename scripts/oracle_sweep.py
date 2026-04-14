"""
Oracle Exhaustive Sweep (no-model version)

Directly feeds fixed (dim0, intra) actions to the environment, bypassing
the DT model entirely. ~3-5x faster than the model-based version.

11 codebook × 3 intra = 33 combos × 5 scenarios × 5 eps × 1 seed = 825 eps

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/oracle_sweep.py
    .pixi/envs/default/bin/python scripts/oracle_sweep.py --full       # 5 seeds
    .pixi/envs/default/bin/python scripts/oracle_sweep.py --parallel 3
"""
import os, sys, json, time, itertools, argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTRA_NAMES = {0: "RR", 1: "PF", 2: "MT"}
SCENARIOS = [5, 6, 7, 8, 9]
N_EPS = 5
NUM_SLICES = 5


def _compute_step_metrics(info, num_slices=NUM_SLICES):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        neg = [d for d in drifts if d < 0]
        if neg:
            if is_hp:
                hp_dist += min(neg)
            else:
                nhp_dist += min(neg)
        if min(drifts) < 0:
            if is_hp:
                hp_viols += 1
            else:
                nhp_viols += 1
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def run_n_episodes(env, dim0, intra, n_eps):
    """Run n episodes with fixed action. Avoids double-reset with DummyVecEnv."""
    action = np.array([[dim0] + [intra] * NUM_SLICES], dtype=np.int32)
    env.reset()
    results = []

    for _ in range(n_eps):
        ep_reward = 0.0
        hp_dist_sum = nhp_dist_sum = 0.0
        hp_viol_sum = nhp_viol_sum = 0
        hp_active_total = nhp_active_total = 0
        done = False

        while not done:
            next_obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])

            step_info = infos[0] if isinstance(infos, (list, tuple)) else infos
            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(step_info)
            hp_dist_sum += hp_d
            nhp_dist_sum += nhp_d
            hp_viol_sum += hp_v
            nhp_viol_sum += nhp_v
            hp_active_total += hp_a
            nhp_active_total += nhp_a

            if dones[0]:
                done = True

        hp_dist_ep = hp_dist_sum / hp_active_total if hp_active_total > 0 else 0.0
        nhp_dist_ep = nhp_dist_sum / nhp_active_total if nhp_active_total > 0 else 0.0
        hp_viol_ep = hp_viol_sum / hp_active_total if hp_active_total > 0 else 0.0
        nhp_viol_ep = nhp_viol_sum / nhp_active_total if nhp_active_total > 0 else 0.0

        results.append((ep_reward, hp_viol_ep, nhp_viol_ep, hp_dist_ep, nhp_dist_ep))

    return results


def run_scenario_combos(scenario_id, combos, test_seeds, n_eps):
    """Run all combos for one scenario. Returns {label: {metric: value}}."""
    import torch
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"

    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
    from src.basic_apis.network_slicing_business.path_manager import PathManager

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())

    results = {}
    for dim0, intra in combos:
        label = f"act{dim0}_{INTRA_NAMES[intra]}"
        seed_hp_viols, seed_nhp_viols = [], []
        seed_hp_dists, seed_nhp_dists = [], []
        seed_rewards = []

        for seed in test_seeds:
            cfg = env_cfg.env_settings.copy()
            cfg.mode = "testing"
            cfg.scenario_mode = "inside"
            cfg.model_name = f"scenario_{scenario_id}"
            cfg.inside.testing.active_scenario_list = [scenario_id]
            cfg.inside.testing.max_scenario_episodes = (
                cfg.inside.testing.init_scenario_episode + n_eps
            )

            def _make_env(c=cfg, s=seed):
                def _init():
                    return HierarchicalSlicingEnv(c, np.random.default_rng(s), pm)
                return _init

            env = DummyVecEnv([_make_env()])
            try:
                ep_results = run_n_episodes(env, dim0, intra, n_eps)
            finally:
                env.close()
            ep_rw = [r[0] for r in ep_results]
            ep_hv = [r[1] for r in ep_results]
            ep_nv = [r[2] for r in ep_results]
            ep_hd = [r[3] for r in ep_results]
            ep_nd = [r[4] for r in ep_results]

            seed_hp_viols.append(np.mean(ep_hv))
            seed_nhp_viols.append(np.mean(ep_nv))
            seed_hp_dists.append(np.mean(ep_hd))
            seed_nhp_dists.append(np.mean(ep_nd))
            seed_rewards.append(np.mean(ep_rw))

        results[label] = {
            "hp_viol_mean": float(np.mean(seed_hp_viols)),
            "nhp_viol_mean": float(np.mean(seed_nhp_viols)),
            "hp_dist_mean": float(np.mean(seed_hp_dists)),
            "nhp_dist_mean": float(np.mean(seed_nhp_dists)),
            "reward_mean": float(np.mean(seed_rewards)),
        }
    return scenario_id, results


def main():
    parser = argparse.ArgumentParser(description="Oracle exhaustive sweep (no-model)")
    parser.add_argument("--full", action="store_true", help="5 seeds instead of 1")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers (CPU-only)")
    parser.add_argument("--scenarios", type=int, nargs="+", default=SCENARIOS)
    parser.add_argument("--n_eps", type=int, default=N_EPS)
    args = parser.parse_args()

    test_seeds = [0, 1, 2, 3, 4] if args.full else [0]
    combos = list(itertools.product(range(11), [0, 1, 2]))

    total_eps = len(combos) * len(args.scenarios) * args.n_eps * len(test_seeds)
    print(f"Oracle sweep (no-model): {len(combos)} combos × {len(args.scenarios)} scenarios "
          f"× {args.n_eps} eps × {len(test_seeds)} seeds = {total_eps} eps")
    print(f"Parallel workers: {args.parallel}\n")

    t0 = time.time()
    all_results = {}  # {scenario_id: {label: metrics}}

    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {}
            for scen_id in args.scenarios:
                fut = pool.submit(run_scenario_combos, scen_id, combos, test_seeds, args.n_eps)
                futures[fut] = scen_id

            for fut in as_completed(futures):
                scen_id = futures[fut]
                try:
                    sid, res = fut.result()
                    all_results[sid] = res
                    elapsed = time.time() - t0
                    print(f"  [{elapsed/60:.1f}min] Done: scenario_{sid}")
                except Exception as e:
                    print(f"  FAILED scenario_{scen_id}: {e}")
                    import traceback; traceback.print_exc()
    else:
        for scen_id in args.scenarios:
            print(f"\n{'='*60}")
            print(f"  Scenario {scen_id}")
            print(f"{'='*60}")
            sid, res = run_scenario_combos(scen_id, combos, test_seeds, args.n_eps)
            all_results[sid] = res
            elapsed = time.time() - t0
            print(f"  [{elapsed/60:.1f}min elapsed]")

    total_time = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  ORACLE SWEEP COMPLETE ({total_time/60:.1f} min)")
    print(f"{'='*80}\n")

    save_path = "data/channel_generality/oracle_sweep"
    os.makedirs(save_path, exist_ok=True)

    structured = {}
    for scen_id, combo_results in all_results.items():
        for label, metrics in combo_results.items():
            if label not in structured:
                structured[label] = {}
            structured[label][f"scenario_{scen_id}"] = metrics

    with open(os.path.join(save_path, "oracle_results.json"), "w") as f:
        json.dump(structured, f, indent=2)
    print(f"Results saved to {save_path}/oracle_results.json\n")

    print_results_table(structured, args.scenarios)
    print_best_combos(structured, args.scenarios)
    print_baseline_comparison(args.scenarios, test_seeds)


def print_results_table(structured, scenarios):
    for scen_id in scenarios:
        scen_key = f"scenario_{scen_id}"
        print(f"\n{'='*80}")
        print(f"  Scenario {scen_id} — Top 10 by (hp_viol + nhp_viol)")
        print(f"{'='*80}")

        rows = []
        for label, scen_data in structured.items():
            if scen_key not in scen_data:
                continue
            d = scen_data[scen_key]
            combined = d["hp_viol_mean"] + d["nhp_viol_mean"]
            rows.append((label, d["hp_viol_mean"], d["nhp_viol_mean"],
                         d["hp_dist_mean"], d["nhp_dist_mean"], combined))

        rows.sort(key=lambda r: r[5])

        header = f"{'Combo':>12} | {'hp_viol':>9} | {'nhp_viol':>9} | {'hp_dist':>9} | {'nhp_dist':>9} | {'combined':>9}"
        print(header)
        print("-" * len(header))
        for label, hv, nv, hd, nd, comb in rows[:10]:
            print(f"{label:>12} | {hv:>9.4f} | {nv:>9.4f} | {hd:>9.4f} | {nd:>9.4f} | {comb:>9.4f}")
        if len(rows) > 10:
            print(f"  ... ({len(rows) - 10} more rows, worst combined = {rows[-1][5]:.4f})")


def print_best_combos(structured, scenarios):
    print(f"\n{'='*80}")
    print("  BEST COMBOS PER SCENARIO (by hp_viol + nhp_viol)")
    print(f"{'='*80}")

    overall_scores = {}
    for scen_id in scenarios:
        scen_key = f"scenario_{scen_id}"
        best_label, best_score = None, float("inf")
        for label, scen_data in structured.items():
            if scen_key not in scen_data:
                continue
            d = scen_data[scen_key]
            score = d["hp_viol_mean"] + d["nhp_viol_mean"]
            if label not in overall_scores:
                overall_scores[label] = 0.0
            overall_scores[label] += score
            if score < best_score:
                best_score = score
                best_label = label
                best_data = d
        if best_label:
            print(f"  S{scen_id}: {best_label:>12}  hp_v={best_data['hp_viol_mean']:.4f}  "
                  f"nhp_v={best_data['nhp_viol_mean']:.4f}  combined={best_score:.4f}")

    print(f"\n  OVERALL RANKING (sum of combined across {len(scenarios)} scenarios):")
    top10 = sorted(overall_scores.items(), key=lambda x: x[1])[:10]
    for rank, (label, score) in enumerate(top10, 1):
        print(f"    #{rank:>2}  {label:>12}: {score:.4f}")


def print_baseline_comparison(scenarios, test_seeds):
    print(f"\n{'='*80}")
    print("  BASELINE COMPARISON (seed 0)")
    print(f"{'='*80}")

    baselines = [
        ("ppo_ha_wt", "data/channel_generality/ppo_ha_weighted"),
        ("ppo_multi", "data/channel_generality/ppo_multi"),
        ("dt_fixed_rr", "data/channel_generality/dt_fixed_rr"),
        ("dt_original", "data/channel_generality/dt"),
    ]

    header = f"{'Method':>15}"
    for s in scenarios:
        header += f" | S{s}_hp  S{s}_nhp"
    header += " | sum_combined"
    print(header)
    print("-" * len(header))

    for name, bpath in baselines:
        row = f"{name:>15}"
        total_combined = 0.0
        all_valid = True
        for scen in scenarios:
            hp_vals, nhp_vals = [], []
            for seed in test_seeds:
                sp = os.path.join(bpath, "metric_json", f"scenario_{scen}", f"seed_{seed}")
                hp_fp = os.path.join(sp, "hp_violations.json")
                nhp_fp = os.path.join(sp, "nhp_violations.json")
                if os.path.exists(hp_fp) and os.path.exists(nhp_fp):
                    with open(hp_fp) as f:
                        hp_vals.append(json.load(f).get("mean", float("nan")))
                    with open(nhp_fp) as f:
                        nhp_vals.append(json.load(f).get("mean", float("nan")))
            if hp_vals:
                hp_mean = np.mean(hp_vals)
                nhp_mean = np.mean(nhp_vals)
                row += f" | {hp_mean:.4f} {nhp_mean:.4f}"
                total_combined += hp_mean + nhp_mean
            else:
                row += f" |    N/A     N/A"
                all_valid = False
        if all_valid:
            row += f" | {total_combined:.4f}"
        else:
            row += f" |          N/A"
        print(row)


if __name__ == "__main__":
    main()
