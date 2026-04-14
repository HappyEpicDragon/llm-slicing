"""
构建多教师数据集：保留所有 agent 的数据，仅在 agent 内做 best-of-N

从 training/ 和 discarded_training/ 中恢复所有轨迹，
按 (scenario, agent, physical_episode) 分组，每组保留回报最高的轨迹。
输出到新目录，不覆盖原始数据集。

用法:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/build_multi_teacher_dataset.py
"""
import os
import sys
import glob
import json
import shutil
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_KEPT = "data/channel_generality/dt/dataset/training"
SRC_DISC = "data/channel_generality/dt/dataset/discarded_training"
SRC_META = "data/channel_generality/dt/dataset/metadata.json"
DST_BASE = "data/channel_generality/dt_multi_teacher/dataset"
DST_TRAIN = os.path.join(DST_BASE, "training")
DST_EVAL = os.path.join(DST_BASE, "evaluating")


def parse_filename(path):
    name = os.path.basename(path)
    parts = name.replace(".pkl", "").split("_")
    scen = int(parts[0][1:])
    agent = int(parts[1][1:])
    ep = int(parts[2 + 1])  # skip "ep" token
    ret_str = parts[-1]
    try:
        ret = float(ret_str)
    except ValueError:
        ret = float("-inf")
    return scen, agent, ep, ret


def main():
    os.makedirs(DST_TRAIN, exist_ok=True)
    os.makedirs(DST_EVAL, exist_ok=True)

    # Collect all trajectory files
    print("Scanning source directories...")
    all_files = []
    for src in [SRC_KEPT, SRC_DISC]:
        if os.path.exists(src):
            files = glob.glob(os.path.join(src, "*.pkl"))
            all_files.extend(files)
            print(f"  {src}: {len(files)} files")
    print(f"  Total: {len(all_files)} files")

    # Group by (scenario, agent, physical_episode)
    print("\nGrouping by (scenario, agent, episode)...")
    groups = defaultdict(list)
    for f in all_files:
        scen, agent, ep, ret = parse_filename(f)
        groups[(scen, agent, ep)].append((ret, f))

    # Best-of-N within each (scenario, agent, episode)
    print("Selecting best-of-N per (scenario, agent, episode)...")
    selected = []
    scen_agent_counts = defaultdict(int)
    scen_agent_returns = defaultdict(list)

    for (scen, agent, ep), candidates in groups.items():
        best_ret, best_file = max(candidates, key=lambda x: x[0])
        selected.append(best_file)
        scen_agent_counts[(scen, agent)] += 1
        scen_agent_returns[(scen, agent)].append(best_ret)

    print(f"\nSelected {len(selected)} trajectories (from {len(all_files)} total)")
    print("\nPer (scenario, agent) distribution:")
    for (s, a), cnt in sorted(scen_agent_counts.items()):
        rets = scen_agent_returns[(s, a)]
        print(f"  S{s} A{a}: {cnt} trajs, return mean={np.mean(rets):.0f}, p95={np.percentile(rets, 95):.0f}")

    # Copy selected files to new directory
    print(f"\nCopying to {DST_TRAIN}...")
    for f in tqdm(selected, desc="Copying"):
        dst = os.path.join(DST_TRAIN, os.path.basename(f))
        shutil.copy2(f, dst)

    # Copy evaluating data (unchanged)
    eval_src = "data/channel_generality/dt/dataset/evaluating"
    if os.path.exists(eval_src):
        print(f"Copying evaluating data...")
        if os.path.exists(DST_EVAL):
            shutil.rmtree(DST_EVAL)
        shutil.copytree(eval_src, DST_EVAL)

    # Recompute metadata
    print("\nRecomputing metadata...")
    recompute_metadata(DST_TRAIN, os.path.join(DST_BASE, "metadata.json"))

    print(f"\nDone! New dataset at: {DST_BASE}")
    print(f"  Training: {len(os.listdir(DST_TRAIN))} trajectories")
    print(f"  To train DT on this dataset:")
    print(f"    pixi run sim dt_training \\")
    print(f"      dt_training.train_dt.dataset.base_dir={DST_BASE}/ \\")
    print(f"      dt_training.train_dt.saving.asset.model_root=data/channel_generality/dt_multi_teacher/model")


def recompute_metadata(train_dir, meta_path):
    """Recompute obs normalization stats and scenario return stats."""
    from src.basic_apis.dt_utils.dataset import HierarchicalDTDataset

    files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))

    # Collect obs stats with correct shapes:
    #   inter_feat: (5, 4) → mean/std shape (4,)  (average over slices)
    #   intra_feat: (25, 5) → mean/std shape (5,)  (average over users)
    #   global_feat: (2,) → mean/std shape (2,)
    inter_all, intra_all, global_all = [], [], []
    scenario_returns = defaultdict(list)
    action_counts = defaultdict(lambda: defaultdict(int))
    expert_contrib = defaultdict(lambda: defaultdict(int))

    for f in tqdm(files, desc="Computing stats"):
        with open(f, "rb") as fh:
            traj = pickle.load(fh)

        scen = traj.get("meta_scen", -1)
        agent = traj.get("meta_agent", -1)
        rewards = np.nan_to_num(traj["rewards"], nan=0.0)
        total_ret = float(np.sum(rewards))
        scenario_returns[scen].append(total_ret)
        expert_contrib[scen][agent] += 1

        obs_list = traj["observations"]
        for obs in obs_list:
            if isinstance(obs, dict):
                if "inter_feat" in obs:
                    v = obs["inter_feat"]
                    inter_all.append(v.reshape(-1, v.shape[-1]))
                if "intra_feat" in obs:
                    v = obs["intra_feat"]
                    intra_all.append(v.reshape(-1, v.shape[-1]))
                if "global_feat" in obs:
                    global_all.append(obs["global_feat"].reshape(1, -1))

        actions = traj["actions"]
        for t in range(len(actions)):
            a = actions[t]
            for dim_i in range(len(a)):
                action_counts[dim_i][str(int(a[dim_i]))] += 1

    inter_arr = np.concatenate(inter_all, axis=0)
    intra_arr = np.concatenate(intra_all, axis=0)
    global_arr = np.concatenate(global_all, axis=0)

    meta = {
        "obs_stats": {
            "inter": {"mean": inter_arr.mean(axis=0).tolist(), "std": inter_arr.std(axis=0).tolist()},
            "intra": {"mean": intra_arr.mean(axis=0).tolist(), "std": intra_arr.std(axis=0).tolist()},
            "global": {"mean": global_arr.mean(axis=0).tolist(), "std": global_arr.std(axis=0).tolist()},
        },
        "scenario_stats": {
            str(s): {
                "min": float(np.min(rets)), "mean": float(np.mean(rets)),
                "max": float(np.max(rets)), "p95": float(np.percentile(rets, 95)),
            }
            for s, rets in scenario_returns.items()
        },
        "action_dist": {str(k): dict(v) for k, v in action_counts.items()},
        "expert_contrib": {
            str(s): {str(a): c for a, c in agents.items()}
            for s, agents in expert_contrib.items()
        },
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
