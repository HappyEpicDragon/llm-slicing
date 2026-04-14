import pickle
import numpy as np


if __name__ == '__main__':
    # 加载训练集
    with open("/root/decision_transformer_slicing/data/channel_generality/dt/dataset/scenario_0/dt_dataset_training.pkl", "rb") as f:
        trajectories = pickle.load(f)

    # 提取所有动作
    all_actions = np.concatenate([t['actions'] for t in trajectories], axis=0)

    print(f"Action Shape: {all_actions.shape}")
    print(f"Max Action: {all_actions.max()}")
    print(f"Min Action: {all_actions.min()}")
    print(f"Mean Action: {all_actions.mean()}")
    print(f"Std Action: {all_actions.std()}")

    # 检查是否全是 0
    if np.allclose(all_actions, 0):
        print("❌ 严重警告：数据集中所有动作几乎全为 0！")
        print("   -> 原因可能是 collect_v3.py 中模型未正确加载，或者环境处于无效状态。")
    else:
        print("✅ 动作数据看起来有分布，非全 0。")