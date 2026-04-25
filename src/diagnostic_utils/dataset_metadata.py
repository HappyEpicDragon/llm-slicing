import os
import pickle
import numpy as np
import json
from tqdm import tqdm
from omegaconf import OmegaConf


def compute_stats():
    # 配置路径
    train_path = '/root/decision_transformer_slicing/data/channel_generality/dt/dataset/scenario_0/dt_dataset_training.pkl'

    print(f"📊 Scanning Dataset: {train_path}")

    with open(train_path, "rb") as f:
        trajs = pickle.load(f)

    # 容器：用于累积 sum 和 sum_sq
    stats = {
        "inter": {"sum": 0, "sq_sum": 0, "count": 0},
        "intra": {"sum": 0, "sq_sum": 0, "count": 0},
        "global": {"sum": 0, "sq_sum": 0, "count": 0}
    }

    # 1. 遍历所有数据
    for t in tqdm(trajs, desc="Computing Statistics"):
        obs = t['observations']

        # Inter: [T, 5, 4] -> Flatten to [N, 4]
        inter = obs['inter_feat'].reshape(-1, 4)
        stats["inter"]["sum"] += inter.sum(axis=0)
        stats["inter"]["sq_sum"] += (inter ** 2).sum(axis=0)
        stats["inter"]["count"] += inter.shape[0]

        # Intra: [T, 25, 5] -> Flatten to [N, 5]
        intra = obs['intra_feat'].reshape(-1, 5)
        stats["intra"]["sum"] += intra.sum(axis=0)
        stats["intra"]["sq_sum"] += (intra ** 2).sum(axis=0)
        stats["intra"]["count"] += intra.shape[0]

        # Global: [T, 2] -> Flatten to [N, 2]
        glob = obs['global_feat'].reshape(-1, 2)
        stats["global"]["sum"] += glob.sum(axis=0)
        stats["global"]["sq_sum"] += (glob ** 2).sum(axis=0)
        stats["global"]["count"] += glob.shape[0]

    # 2. 计算 Mean / Std
    metadata = {}
    for key in stats:
        N = stats[key]["count"]
        mean = stats[key]["sum"] / N
        # Var = E[X^2] - (E[X])^2
        var = (stats[key]["sq_sum"] / N) - (mean ** 2)
        # 加上 epsilon 防止除零
        std = np.sqrt(np.maximum(var, 1e-6))

        # 转为 list 存 JSON
        metadata[key] = {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
        print(f"✅ {key.capitalize()} - Mean: {np.round(mean, 2)} | Std: {np.round(std, 2)}")

    # 3. 保存
    output_path = "dataset_metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Metadata saved to {output_path}")


if __name__ == "__main__":
    compute_stats()