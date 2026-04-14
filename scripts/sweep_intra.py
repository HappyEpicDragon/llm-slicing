"""
Intra-Slice Scheduling Sensitivity Sweep (快速诊断版)

3 scenarios × 5 episodes × 4 configs × 1 seed ≈ 8 分钟

用法:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/sweep_intra.py
"""
import os, sys, json, time
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"

from src.basic_apis.dt_utils.test import test_dt_process
from src.basic_apis.network_slicing_business.path_manager import PathManager


def build_cfg(scenarios, n_episodes=5, force_dim0=None, force_intra=None):
    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    dt_cfg = OmegaConf.load("conf/simulation/channel_generality/dt_testing.yaml")
    cfg = OmegaConf.merge(dt_cfg.dt_testing, {"environment": env_cfg})

    cfg.test_seeds = [0]
    cfg.save_results = False
    cfg.force_dim0 = int(force_dim0) if force_dim0 is not None else None
    cfg.force_intra = int(force_intra) if force_intra is not None else None

    cfg.env_updates.inside.testing.active_scenario_list = scenarios
    cfg.environment.env_settings.inside.testing.max_scenario_episodes = (
        cfg.environment.env_settings.inside.testing.init_scenario_episode + n_episodes
    )
    return cfg


def extract_metrics(results):
    """从 test_dt_process 返回值提取关键指标"""
    hp_v, nhp_v, hp_d, nhp_d = [], [], [], []
    for scen_key, data in results.items():
        hp_v.append(data["hp_viol_mean"])
        nhp_v.append(data["nhp_viol_mean"])
        hp_d.append(data["hp_dist_mean"])
        nhp_d.append(data["nhp_dist_mean"])
    return {
        "hp_viol": np.mean(hp_v), "nhp_viol": np.mean(nhp_v),
        "hp_dist": np.mean(hp_d), "nhp_dist": np.mean(nhp_d),
    }


def main():
    pm = PathManager(os.getcwd())
    scenarios = [5, 6, 9]
    n_eps = 5

    configs = [
        ("DT original",  None, None),
        ("All RR (0)",    None, 0),
        ("All PF (1)",    None, 1),
        ("All MT (2)",    None, 2),
    ]

    total_eps = len(configs) * len(scenarios) * n_eps
    print(f"Intra sweep: {len(configs)} configs × {len(scenarios)} scenarios × {n_eps} eps = {total_eps} eps")
    print(f"Estimated: ~{total_eps * 8 / 60:.0f} min\n")

    all_results = {}
    t0 = time.time()

    for label, fd0, fi in configs:
        print(f"--- {label} ---")
        cfg = build_cfg(scenarios, n_eps, force_dim0=fd0, force_intra=fi)
        res = test_dt_process(cfg, pm)
        m = extract_metrics(res)
        all_results[label] = m
        elapsed = time.time() - t0
        print(f"  hp_v={m['hp_viol']:.4f} nhp_v={m['nhp_viol']:.4f} "
              f"hp_d={m['hp_dist']:.4f} nhp_d={m['nhp_dist']:.4f}  [{elapsed/60:.1f}min]\n")

    print(f"\n{'='*72}")
    print(f"  INTRA SWEEP RESULTS (scenarios {scenarios}, {n_eps} eps, seed=0)")
    print(f"{'='*72}\n")

    header = f"{'Config':>15} | {'hp_viol':>10} | {'nhp_viol':>10} | {'hp_dist':>10} | {'nhp_dist':>10}"
    print(header)
    print("-" * len(header))
    for label, m in all_results.items():
        print(f"{label:>15} | {m['hp_viol']:>10.4f} | {m['nhp_viol']:>10.4f} | "
              f"{m['hp_dist']:>10.4f} | {m['nhp_dist']:>10.4f}")

    # Baseline comparison (same scenarios, seed 0)
    print("-" * len(header))
    for baseline, bpath in [
        ("ppo_ha_wt", "data/channel_generality/ppo_ha_weighted"),
        ("ppo_multi", "data/channel_generality/ppo_multi"),
    ]:
        vals = {"hp_viol": [], "nhp_viol": [], "hp_dist": [], "nhp_dist": []}
        for scen in scenarios:
            sp = os.path.join(bpath, "metric_json", f"scenario_{scen}", "seed_0")
            for mk, fname in [("hp_viol", "hp_violations.json"), ("nhp_viol", "nhp_violations.json"),
                              ("hp_dist", "hp_distance.json"), ("nhp_dist", "nhp_distance.json")]:
                fp = os.path.join(sp, fname)
                if os.path.exists(fp):
                    with open(fp) as f:
                        vals[mk].append(json.load(f).get("mean", float("nan")))
        print(f"{baseline:>15} | {np.mean(vals['hp_viol']):>10.4f} | {np.mean(vals['nhp_viol']):>10.4f} | "
              f"{np.mean(vals['hp_dist']):>10.4f} | {np.mean(vals['nhp_dist']):>10.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
