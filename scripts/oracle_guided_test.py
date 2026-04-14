"""
Oracle-Guided Test (parallel version): run per-scenario Oracle-best codebook + RR
under the same evaluation conditions (20 eps × 5 seeds), parallelized by scenario.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/oracle_guided_test.py
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_SLICES = 5
SCENARIOS = [5, 6, 7, 8, 9]
ORACLE_BEST = {5: 5, 6: 3, 7: 6, 8: 2, 9: 9}
OVERALL_BEST_DIM0 = 2


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
            hp_dist_sum += hp_d; nhp_dist_sum += nhp_d
            hp_viol_sum += hp_v; nhp_viol_sum += nhp_v
            hp_active_total += hp_a; nhp_active_total += nhp_a
            if dones[0]:
                done = True
        hp_viol_ep = hp_viol_sum / hp_active_total if hp_active_total > 0 else 0.0
        nhp_viol_ep = nhp_viol_sum / nhp_active_total if nhp_active_total > 0 else 0.0
        hp_dist_ep = hp_dist_sum / hp_active_total if hp_active_total > 0 else 0.0
        nhp_dist_ep = nhp_dist_sum / nhp_active_total if nhp_active_total > 0 else 0.0
        results.append((ep_reward, hp_viol_ep, nhp_viol_ep, hp_dist_ep, nhp_dist_ep))
    return results


def run_scenario_batch(scenario_id, dim0_list, seeds, n_eps):
    """Run multiple dim0 values for one scenario (in a single process)."""
    import torch
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
    from src.basic_apis.network_slicing_business.path_manager import PathManager

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())
    all_results = {}

    for dim0 in dim0_list:
        seed_results = []
        for seed in seeds:
            cfg = env_cfg.env_settings.copy()
            cfg.mode = "testing"
            cfg.scenario_mode = "inside"
            cfg.model_name = f"scenario_{scenario_id}"
            cfg.inside.testing.active_scenario_list = [scenario_id]
            cfg.inside.testing.max_scenario_episodes = (
                cfg.inside.testing.init_scenario_episode + n_eps
            )

            def _make(c=cfg, s=seed):
                def _init():
                    return HierarchicalSlicingEnv(c, np.random.default_rng(s), pm)
                return _init

            env = DummyVecEnv([_make()])
            try:
                eps = run_n_episodes(env, dim0, 0, n_eps)
            finally:
                env.close()

            seed_results.append({
                "hp_viol": np.mean([e[1] for e in eps]),
                "nhp_viol": np.mean([e[2] for e in eps]),
                "hp_dist": np.mean([e[3] for e in eps]),
                "nhp_dist": np.mean([e[4] for e in eps]),
                "ep_hp_viols": [e[1] for e in eps],
                "ep_nhp_viols": [e[2] for e in eps],
            })

        all_results[dim0] = {
            "hp_viol_mean": float(np.mean([r["hp_viol"] for r in seed_results])),
            "nhp_viol_mean": float(np.mean([r["nhp_viol"] for r in seed_results])),
            "hp_dist_mean": float(np.mean([r["hp_dist"] for r in seed_results])),
            "nhp_dist_mean": float(np.mean([r["nhp_dist"] for r in seed_results])),
        }

    return scenario_id, all_results


def main():
    seeds = [0, 1, 2, 3, 4]
    n_eps = 20

    tasks = {}
    for scen in SCENARIOS:
        oracle_d0 = ORACLE_BEST[scen]
        dim0_list = sorted(set([oracle_d0, OVERALL_BEST_DIM0]))
        if scen == 5:
            dim0_list = list(range(11))
        tasks[scen] = dim0_list

    total = sum(len(v) * n_eps * len(seeds) for v in tasks.values())
    print(f"Oracle-guided test (parallel): {total} total episodes")
    print(f"Configs per scenario: {[(s, len(v)) for s, v in tasks.items()]}")
    print()

    t0 = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {}
        for scen, d0_list in tasks.items():
            fut = pool.submit(run_scenario_batch, scen, d0_list, seeds, n_eps)
            futures[fut] = scen

        for fut in as_completed(futures):
            scen = futures[fut]
            try:
                sid, res = fut.result()
                results[sid] = res
                elapsed = time.time() - t0
                print(f"  [{elapsed/60:.1f}min] Done: scenario_{sid} ({len(res)} configs)")
            except Exception as e:
                print(f"  FAILED scenario_{scen}: {e}")
                import traceback; traceback.print_exc()

    total_time = time.time() - t0
    print(f"\nAll done in {total_time/60:.1f} min\n")

    save_dir = "data/channel_generality/oracle_guided"
    os.makedirs(save_dir, exist_ok=True)
    save_data = {}
    for scen, dim0_results in results.items():
        for d0, metrics in dim0_results.items():
            save_data[f"S{scen}_act{d0}_RR"] = metrics
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print_comparison(results)


def print_comparison(results):
    dt_base = "data/channel_generality/dt_fixed_rr/metric_json"
    ppo_base = "data/channel_generality/ppo_ha_weighted/metric_json"

    print(f"{'='*95}")
    print("  MAIN COMPARISON TABLE (20 eps × 5 seeds, same evaluation conditions)")
    print(f"{'='*95}")

    methods_data = {}

    # Oracle per-scenario best
    oracle_ps = {}
    for s in SCENARIOS:
        d0 = ORACLE_BEST[s]
        if s in results and d0 in results[s]:
            oracle_ps[s] = results[s][d0]
    methods_data["(A) oracle_per_scen"] = oracle_ps

    # Overall best act2_RR
    overall = {}
    for s in SCENARIOS:
        if s in results and OVERALL_BEST_DIM0 in results[s]:
            overall[s] = results[s][OVERALL_BEST_DIM0]
    methods_data["(B) act2_RR_fixed"] = overall

    # Load baselines
    for name, path in [("(C) dt_fixed_rr", dt_base), ("(D) ppo_ha_weighted", ppo_base)]:
        data = {}
        for s in SCENARIOS:
            fp = f"{path}/scenario_{s}/summary.json"
            if os.path.exists(fp):
                d = json.load(open(fp))
                data[s] = {"hp_viol_mean": d["hp_viol_mean"], "nhp_viol_mean": d["nhp_viol_mean"]}
        methods_data[name] = data

    header = f"{'Method':>25}"
    for s in SCENARIOS:
        header += f" | S{s:>1}_comb"
    header += " |   TOTAL"
    print(header)
    print("-" * len(header))

    for name, data in methods_data.items():
        row = f"{name:>25}"
        total = 0
        for s in SCENARIOS:
            if s in data:
                comb = data[s]["hp_viol_mean"] + data[s]["nhp_viol_mean"]
                total += comb
                row += f" | {comb:>7.4f}"
            else:
                row += f" |     N/A"
        row += f" | {total:>7.4f}"
        print(row)

    print()
    print("  (A) = per-scenario Oracle-best codebook + RR = theoretical upper bound for fixed-action policy")
    print("  (B) = single best overall codebook (act2) + RR = best single fixed policy")
    print("  (C) = DT model predicts dim0 + forced RR = current DT approach")
    print("  (D) = PPO-HA-weighted end-to-end policy")

    # S5 deep dive
    if 5 in results:
        print(f"\n{'='*75}")
        print("  S5 DEEP DIVE: All 11 codebooks (20 eps × 5 seeds)")
        print(f"{'='*75}")
        s5 = results[5]
        rows = []
        for d0, m in s5.items():
            comb = m["hp_viol_mean"] + m["nhp_viol_mean"]
            rows.append((f"act{d0}_RR", m["hp_viol_mean"], m["nhp_viol_mean"], m["nhp_dist_mean"], comb))
        rows.sort(key=lambda x: x[4])
        print(f"{'Combo':>12} | {'hp_viol':>8} | {'nhp_viol':>8} | {'nhp_dist':>9} | {'combined':>8}")
        print("-" * 60)
        for label, hv, nv, nd, c in rows:
            print(f"{label:>12} | {hv:>8.4f} | {nv:>8.4f} | {nd:>9.4f} | {c:>8.4f}")

        dt_s5 = json.load(open(f"{dt_base}/scenario_5/summary.json"))
        ppo_s5 = json.load(open(f"{ppo_base}/scenario_5/summary.json"))
        print("-" * 60)
        dt_c = dt_s5["hp_viol_mean"] + dt_s5["nhp_viol_mean"]
        ppo_c = ppo_s5["hp_viol_mean"] + ppo_s5["nhp_viol_mean"]
        print(f"{'dt_fixed_rr':>12} | {dt_s5['hp_viol_mean']:>8.4f} | {dt_s5['nhp_viol_mean']:>8.4f} | {dt_s5['nhp_dist_mean']:>9.4f} | {dt_c:>8.4f}")
        print(f"{'ppo_ha_wt':>12} | {ppo_s5['hp_viol_mean']:>8.4f} | {ppo_s5['nhp_viol_mean']:>8.4f} | {ppo_s5['nhp_dist_mean']:>9.4f} | {ppo_c:>8.4f}")

    # Gap analysis
    print(f"\n{'='*75}")
    print("  GAP ANALYSIS")
    print(f"{'='*75}")
    oracle_total = sum(
        methods_data["(A) oracle_per_scen"][s]["hp_viol_mean"] + methods_data["(A) oracle_per_scen"][s]["nhp_viol_mean"]
        for s in SCENARIOS if s in methods_data["(A) oracle_per_scen"]
    )
    overall_total = sum(
        methods_data["(B) act2_RR_fixed"][s]["hp_viol_mean"] + methods_data["(B) act2_RR_fixed"][s]["nhp_viol_mean"]
        for s in SCENARIOS if s in methods_data["(B) act2_RR_fixed"]
    )
    dt_total = sum(
        methods_data["(C) dt_fixed_rr"][s]["hp_viol_mean"] + methods_data["(C) dt_fixed_rr"][s]["nhp_viol_mean"]
        for s in SCENARIOS if s in methods_data["(C) dt_fixed_rr"]
    )
    ppo_total = sum(
        methods_data["(D) ppo_ha_weighted"][s]["hp_viol_mean"] + methods_data["(D) ppo_ha_weighted"][s]["nhp_viol_mean"]
        for s in SCENARIOS if s in methods_data["(D) ppo_ha_weighted"]
    )

    print(f"  (A) Oracle per-scenario: {oracle_total:.4f}")
    print(f"  (B) Best single fixed:   {overall_total:.4f}   gap from (A): {overall_total - oracle_total:+.4f}")
    print(f"  (D) PPO-HA-weighted:     {ppo_total:.4f}   gap from (A): {ppo_total - oracle_total:+.4f}")
    print(f"  (C) DT + fixed RR:       {dt_total:.4f}   gap from (A): {dt_total - oracle_total:+.4f}")
    print()
    print(f"  DT vs PPO gap:                  {dt_total - ppo_total:+.4f}")
    print(f"  DT's improvable gap to (A):     {dt_total - oracle_total:+.4f}")
    print(f"  Of which Dim0 selection:        ~{dt_total - overall_total:+.4f}")
    print(f"  Of which scene-adaptive Dim0:   ~{overall_total - oracle_total:+.4f}")


if __name__ == "__main__":
    main()
