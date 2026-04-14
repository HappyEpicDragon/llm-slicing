"""
构建 violation-RTG 数据集：用基于 metric 的 reward 替代原始 reward

从现有 pkl 轨迹中重建 step-level violation reward:
  viol_reward_t = sum_over_slices(drift * weight if drift < 0)
  weight = 10 for HP, 1 for NHP

用法:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/build_viol_rtg_dataset.py [--src small|multi]
"""
import os, sys, glob, shutil, pickle, argparse
import numpy as np
from tqdm import tqdm

HP_WEIGHT = 10.0
NHP_WEIGHT = 1.0


def compute_viol_rewards(traj):
    """从轨迹的 observations 中重建 step-level violation reward"""
    obs_list = traj["observations"]
    T = len(obs_list)
    viol_rewards = np.zeros(T, dtype=np.float64)

    for t in range(T):
        obs = obs_list[t]
        if isinstance(obs, dict):
            inter = obs["inter_feat"]  # (5, 4): [priority, drift, traffic, alloc]
        else:
            continue

        step_reward = 0.0
        for s in range(inter.shape[0]):
            prio = inter[s, 0]
            drift = inter[s, 1]
            if drift < 0:
                w = HP_WEIGHT if prio > 0 else NHP_WEIGHT
                step_reward += drift * w
        viol_rewards[t] = step_reward

    return viol_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", choices=["small", "multi"], default="small",
                        help="small=原始2039条, multi=多教师58K条")
    args = parser.parse_args()

    if args.src == "small":
        src_train = "data/channel_generality/dt/dataset/training"
        src_meta = "data/channel_generality/dt/dataset/metadata.json"
        dst_base = "data/channel_generality/dt_viol_rtg/dataset"
    else:
        src_train = "data/channel_generality/dt_multi_teacher/dataset/training"
        src_meta = "data/channel_generality/dt_multi_teacher/dataset/metadata.json"
        dst_base = "data/channel_generality/dt_multi_teacher_viol_rtg/dataset"

    dst_train = os.path.join(dst_base, "training")
    os.makedirs(dst_train, exist_ok=True)

    # Copy metadata (obs stats unchanged)
    shutil.copy2(src_meta, os.path.join(dst_base, "metadata.json"))

    # Copy evaluating dir
    eval_src = os.path.dirname(src_train) + "/evaluating"
    eval_dst = os.path.join(dst_base, "evaluating")
    if os.path.exists(eval_src):
        if os.path.exists(eval_dst):
            shutil.rmtree(eval_dst)
        shutil.copytree(eval_src, eval_dst)

    # Process training trajectories
    pkl_files = sorted(glob.glob(os.path.join(src_train, "*.pkl")))
    print(f"Processing {len(pkl_files)} trajectories from {src_train}")

    stats = {"orig_return": [], "viol_return": [], "hp_viols": [], "nhp_viols": []}

    for f in tqdm(pkl_files, desc="Building viol-RTG"):
        with open(f, "rb") as fh:
            traj = pickle.load(fh)

        viol_rewards = compute_viol_rewards(traj)

        stats["orig_return"].append(float(traj["rewards"].sum()))
        stats["viol_return"].append(float(viol_rewards.sum()))
        stats["hp_viols"].append(traj.get("hp_viols", 0))
        stats["nhp_viols"].append(traj.get("nhp_viols", 0))

        traj["rewards"] = viol_rewards
        traj["episode_returns"] = float(viol_rewards.sum())

        dst_path = os.path.join(dst_train, os.path.basename(f))
        with open(dst_path, "wb") as fh:
            pickle.dump(traj, fh)

    # Print stats
    orig = np.array(stats["orig_return"])
    viol = np.array(stats["viol_return"])
    print(f"\nOriginal reward: mean={orig.mean():.0f}, std={orig.std():.0f}")
    print(f"Violation reward: mean={viol.mean():.1f}, std={viol.std():.1f}")
    print(f"  range: [{viol.min():.1f}, {viol.max():.1f}]")
    print(f"  correlation with orig: {np.corrcoef(orig, viol)[0,1]:.3f}")
    print(f"  correlation with hp_viols: {np.corrcoef(stats['hp_viols'], viol)[0,1]:.3f}")
    print(f"  correlation with nhp_viols: {np.corrcoef(stats['nhp_viols'], viol)[0,1]:.3f}")
    print(f"\nDone! Dataset at: {dst_base}")
    print(f"Train command:")
    print(f"  pixi run sim dt_training \\")
    print(f"    dt_training.train_dt.dataset.base_dir={dst_base}/ \\")
    print(f"    dt_training.train_dt.saving.asset.model_root=data/channel_generality/dt_viol_rtg/model")


if __name__ == "__main__":
    main()
