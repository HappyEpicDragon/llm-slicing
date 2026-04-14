import pickle
import numpy as np
import torch


def inspect_pickle(path, name):
    print(f"\n🔍 Inspecting {name}: {path}")
    try:
        with open(path, "rb") as f:
            trajs = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return

    if len(trajs) == 0:
        print("⚠️  Dataset is EMPTY!")
        return

    print(f"   Count: {len(trajs)} trajectories")

    # 1. 检查 Actions (排查 Train Loss = 0)
    all_actions = np.concatenate([t['actions'] for t in trajs], axis=0)
    print(f"   [Actions] Shape: {all_actions.shape}")
    print(f"             Mean: {all_actions.mean():.6f} | Std: {all_actions.std():.6f}")
    print(f"             Min:  {all_actions.min():.6f} | Max: {all_actions.max():.6f}")

    if np.allclose(all_actions, 0, atol=1e-5):
        print("   🚨 ALARM: Actions are effectively ALL ZEROS! (Explains Train Loss 0.0003)")

    # 2. 检查 Rewards / RTG (排查 Val Loss = NaN)
    # 我们现场算一下 RTG 看看有没有 Inf
    all_rewards = np.concatenate([t['rewards'] for t in trajs], axis=0)
    print(f"   [Rewards] Min: {all_rewards.min():.2f} | Max: {all_rewards.max():.2f}")

    if np.isinf(all_rewards).any() or np.isnan(all_rewards).any():
        print("   🚨 ALARM: Rewards contain NaN or Inf!")

    # 检查 Observations 是否有坏值
    has_nan_obs = False
    for t in trajs:
        for k, v in t['observations'].items():
            if np.isnan(v).any() or np.isinf(v).any():
                has_nan_obs = True
                print(f"   🚨 ALARM: Observation '{k}' contains NaN/Inf!")
                break
        if has_nan_obs: break

    if not has_nan_obs:
        print("   ✅ Observations look clean (no NaN/Inf).")


if __name__ == "__main__":
    # 请修改为你的实际路径
    inspect_pickle("/root/decision_transformer_slicing/data/channel_generality/dt/dataset/scenario_0/dt_dataset_training.pkl", "TRAIN SET")
    inspect_pickle("/root/decision_transformer_slicing/data/channel_generality/dt/dataset/scenario_0/dt_dataset_evaluating.pkl", "VAL SET")