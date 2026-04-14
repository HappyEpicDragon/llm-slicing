"""
Phase 0: Verify that slice ordering has high-SNR impact on violations.

Test all 5! = 120 permutations of slice ordering with fixed codebook (act5)
and intra=RR, on test scenarios 5-9. Compare ordering spread vs codebook spread.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/phase0_ordering_sweep.py
"""
import os, sys, json, time, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_SLICES = 5
CODEBOOK = 5  # act5
INTRA = 0     # RR
SCENARIOS = [5, 6, 7, 8, 9]


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
        if min(drifts) < 0:
            if is_hp:
                hp_viols += 1
            else:
                nhp_viols += 1
            neg = [d for d in drifts if d < 0]
            if is_hp:
                hp_dist += min(neg)
            else:
                nhp_dist += min(neg)
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def run_scenario_orderings(scenario_id):
    """Run all 120 orderings + urgency_rank baseline for one scenario."""
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
    from src.basic_apis.network_slicing_business.path_manager import PathManager

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())

    action = np.array([[CODEBOOK] + [INTRA] * NUM_SLICES], dtype=np.int32)
    all_perms = list(itertools.permutations(range(NUM_SLICES)))
    results = {}

    def run_one_episode(env, ordering_label):
        env.reset()
        hp_viol_sum = nhp_viol_sum = 0
        hp_active_t = nhp_active_t = 0
        done = False
        while not done:
            next_obs, rewards, dones, infos = env.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            _, _, hv, nv, ha, na = _compute_step_metrics(info)
            hp_viol_sum += hv; nhp_viol_sum += nv
            hp_active_t += ha; nhp_active_t += na
            if dones[0]:
                done = True
        hp_v = hp_viol_sum / hp_active_t if hp_active_t > 0 else 0.0
        nhp_v = nhp_viol_sum / nhp_active_t if nhp_active_t > 0 else 0.0
        return hp_v + nhp_v

    # --- 1. Run urgency_rank baseline ---
    cfg_ur = env_cfg.env_settings.copy()
    cfg_ur.mode = "testing"
    cfg_ur.scenario_mode = "inside"
    cfg_ur.model_name = f"scenario_{scenario_id}"
    cfg_ur.inside.testing.active_scenario_list = [scenario_id]
    cfg_ur.inside.testing.init_scenario_episode = 0
    cfg_ur.inside.testing.max_scenario_episodes = 1
    cfg_ur.mapping_mode = "urgency_rank"

    def _make_ur(c=cfg_ur):
        def _init():
            return HierarchicalSlicingEnvV2(c, np.random.default_rng(42), pm)
        return _init

    env_ur = DummyVecEnv([_make_ur()])
    results["urgency_rank"] = run_one_episode(env_ur, "urgency_rank")
    env_ur.close()

    # --- 2. Run all 120 orderings ---
    for perm in all_perms:
        cfg = env_cfg.env_settings.copy()
        cfg.mode = "testing"
        cfg.scenario_mode = "inside"
        cfg.model_name = f"scenario_{scenario_id}"
        cfg.inside.testing.active_scenario_list = [scenario_id]
        cfg.inside.testing.init_scenario_episode = 0
        cfg.inside.testing.max_scenario_episodes = 1

        def _make(c=cfg, p=perm):
            def _init():
                e = HierarchicalSlicingEnvV2(c, np.random.default_rng(42), pm)
                e.set_forced_ordering(list(p))
                return e
            return _init

        env = DummyVecEnv([_make()])
        results[str(perm)] = run_one_episode(env, str(perm))
        env.close()

    return scenario_id, results


def main():
    print(f"Phase 0: Ordering Sweep (120 perms × {len(SCENARIOS)} scenarios)")
    print(f"Codebook: act{CODEBOOK}, Intra: RR, 1 episode each\n")

    t0 = time.time()
    all_results = {}

    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(run_scenario_orderings, s): s for s in SCENARIOS}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                sid, res = fut.result()
                all_results[sid] = res
                elapsed = time.time() - t0
                print(f"  [{elapsed/60:.1f}min] S{sid} done ({len(res)} configs)")
            except Exception as e:
                print(f"  FAILED S{s}: {e}")
                import traceback; traceback.print_exc()

    print(f"\nDone in {(time.time()-t0)/60:.1f} min\n")

    save_dir = "data/channel_generality/phase0_ordering"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Analysis
    print("="*80)
    print("  RESULTS: Ordering Spread vs Codebook Spread")
    print("="*80)

    for scen in SCENARIOS:
        if scen not in all_results:
            continue
        res = all_results[scen]
        ur_val = res.pop("urgency_rank")
        perm_vals = list(res.values())

        best_perm = min(res.items(), key=lambda x: x[1])
        worst_perm = max(res.items(), key=lambda x: x[1])
        spread = worst_perm[1] - best_perm[1]
        ur_rank = sum(1 for v in perm_vals if v < ur_val) + 1

        print(f"\n  S{scen}:")
        print(f"    Ordering spread:   {spread:.4f}  (best={best_perm[1]:.4f}, worst={worst_perm[1]:.4f})")
        print(f"    urgency_rank:      {ur_val:.4f}  (rank #{ur_rank}/120)")
        print(f"    Best ordering:     {best_perm[0]}")
        print(f"    Worst ordering:    {worst_perm[0]}")

    print(f"\n  Reference: Codebook spread (from Oracle analysis) ≈ 0.3-0.4")
    print(f"  If ordering spread >> 0.4, then ordering is a high-SNR learning target.")


if __name__ == "__main__":
    main()
