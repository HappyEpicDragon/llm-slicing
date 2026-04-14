"""
Dim0 Sensitivity Sweep (快速版)

用法:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/sweep_dim0.py

快速模式: 1 seed × 5 scenarios × 20 episodes × 4 dim0 值 = 400 episodes ≈ 55 分钟
完整模式: .pixi/envs/default/bin/python scripts/sweep_dim0.py --full
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single_dim0(dim0_value, save_root, test_seeds):
    """在独立进程中跑一个 dim0 值的完整测试"""
    import torch
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"

    from omegaconf import OmegaConf
    from src.basic_apis.dt_utils.test import test_dt_process
    from src.basic_apis.network_slicing_business.path_manager import PathManager

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    dt_cfg = OmegaConf.load("conf/simulation/channel_generality/dt_testing.yaml")
    cfg = OmegaConf.merge(dt_cfg.dt_testing, {"environment": env_cfg})

    cfg.force_dim0 = int(dim0_value) if dim0_value is not None else None
    cfg.save_root = save_root
    cfg.save_results = True
    cfg.clean_before_save = True
    cfg.test_seeds = test_seeds

    pm = PathManager(os.getcwd())
    results = test_dt_process(cfg, pm)
    return results


def collect_summary(base_save, label):
    """从已保存的 JSON 中读取汇总"""
    metrics = {}
    for scen in range(5, 10):
        sp = os.path.join(base_save, label, "metric_json", f"scenario_{scen}", "summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                metrics[f"scenario_{scen}"] = json.load(f)
    return metrics


def print_comparison(all_results, labels):
    metric_keys = ["hp_viol_mean", "nhp_viol_mean", "hp_dist_mean", "nhp_dist_mean"]
    header = f"{'Dim0':>15}"
    for m in metric_keys:
        header += f" | {m:>14}"
    print(header)
    print("-" * (16 + 17 * len(metric_keys)))

    for label in labels:
        if label not in all_results or not all_results[label]:
            continue
        res = all_results[label]
        row = f"{label:>15}"
        for m in metric_keys:
            vals = [v[m] for v in res.values() if m in v]
            avg = np.mean(vals) if vals else float("nan")
            row += f" | {avg:>14.4f}"
        print(row)

    print("-" * (16 + 17 * len(metric_keys)))
    for baseline, bpath in [
        ("ppo_ha_wt", "data/channel_generality/ppo_ha_weighted"),
        ("ppo_multi", "data/channel_generality/ppo_multi"),
    ]:
        row = f"{baseline:>15}"
        for m in metric_keys:
            vals = []
            for scen in range(5, 10):
                sp = os.path.join(bpath, "metric_json", f"scenario_{scen}", "summary.json")
                if os.path.exists(sp):
                    with open(sp) as f:
                        d = json.load(f)
                    if m in d:
                        vals.append(d[m])
            avg = np.mean(vals) if vals else float("nan")
            row += f" | {avg:>14.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="完整模式: 5 seeds, 7 dim0 值")
    parser.add_argument("--parallel", type=int, default=1, help="并行进程数 (注意 GPU 显存)")
    args = parser.parse_args()

    base_save = "data/channel_generality/dt_dim0_sweep"

    if args.full:
        dim0_values = [None, 0, 1, 3, 5, 7, 10]
        test_seeds = [0, 1, 2, 3, 4]
    else:
        dim0_values = [None, 0, 5, 7]
        test_seeds = [0]

    tasks = []
    for dim0 in dim0_values:
        label = "dt_original" if dim0 is None else f"act{dim0}"
        save_root = os.path.join(base_save, label)
        tasks.append((dim0, label, save_root))

    n_eps = len(dim0_values) * 5 * len(test_seeds) * 20
    print(f"Sweep config: {len(dim0_values)} dim0 values × 5 scenarios × {len(test_seeds)} seeds × 20 eps = {n_eps} episodes")
    print(f"Estimated time: ~{n_eps * 8 / 60:.0f} minutes (at 8s/ep)")
    print(f"Parallel workers: {args.parallel}")
    print()

    t0 = time.time()
    all_results = {}
    labels_order = []

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {}
            for dim0, label, save_root in tasks:
                fut = pool.submit(run_single_dim0, dim0, save_root, test_seeds)
                futures[fut] = label
                labels_order.append(label)

            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    res = fut.result()
                    all_results[label] = res
                    elapsed = time.time() - t0
                    print(f"\n  [{elapsed/60:.1f}min] Done: {label}")
                except Exception as e:
                    print(f"\n  FAILED: {label}: {e}")
    else:
        for dim0, label, save_root in tasks:
            labels_order.append(label)
            print(f"\n{'='*60}")
            print(f"  Testing Dim0 = {label}")
            print(f"{'='*60}")
            try:
                res = run_single_dim0(dim0, save_root, test_seeds)
                all_results[label] = res
            except Exception as e:
                print(f"  FAILED: {e}")
            elapsed = time.time() - t0
            print(f"  [{elapsed/60:.1f}min elapsed]")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    print_comparison(all_results, labels_order)
    print()


if __name__ == "__main__":
    main()
