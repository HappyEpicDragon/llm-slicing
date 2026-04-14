"""
S3 数据集子采样：从 D-B（900 traj）按比例随机采样，生成 10%/30%/50% 子集。

输出目录：data/channel_generality/sensitivity/s3_dataset/dataset_{pct}pct/
  - training/（符号链接原始 pkl，节省磁盘）
  - metadata.json（从原始 D-B 复制，obs_stats 不变；scenario_stats 取对应子集的均值）
"""

import os
import json
import shutil
import random
import numpy as np
from pathlib import Path

SEED = 42
SRC_DIR = Path("data/channel_generality/dt_v2_8exp/dataset_D-B")
OUT_BASE = Path("data/channel_generality/sensitivity/s3_dataset")
PERCENTAGES = [10, 30, 50]


def subsample_dataset(src_dir: Path, out_dir: Path, pct: int, seed: int = 42):
    src_train = src_dir / "training"
    all_files = sorted(f for f in src_train.iterdir() if f.suffix == ".pkl")
    total = len(all_files)
    n_sample = max(1, int(total * pct / 100))

    rng = random.Random(seed)
    selected = sorted(rng.sample(all_files, n_sample), key=lambda x: x.name)

    out_train = out_dir / "training"
    out_train.mkdir(parents=True, exist_ok=True)

    # 使用符号链接，节省磁盘空间
    for f in selected:
        link = out_train / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())

    # 复制 metadata（obs_stats 不变，更新 scenario_stats 估计）
    with open(src_dir / "metadata.json") as fp:
        meta = json.load(fp)

    # 从选中文件名解析 scenario return（保留原始 obs_stats）
    import pickle
    scenario_returns: dict[str, list] = {}
    for f in selected:
        try:
            with open(f, "rb") as fp:
                traj = pickle.load(fp)
            rewards = np.array(traj.get("rewards", []), dtype=np.float32)
            ep_ret = float(rewards.sum())
            # 从文件名推断 scenario（如 S0_single_...）
            s_id = f.name.split("_")[0].lstrip("S")
            scenario_returns.setdefault(s_id, []).append(ep_ret)
        except Exception as e:
            print(f"  [WARN] skip {f.name}: {e}")

    updated_stats = {}
    for s_id, rets in scenario_returns.items():
        updated_stats[s_id] = {
            "min": float(np.min(rets)),
            "mean": float(np.mean(rets)),
            "max": float(np.max(rets)),
        }
    meta["scenario_stats"] = updated_stats

    with open(out_dir / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    # 复制 manifest（如有）
    manifest_src = src_dir / "manifest.json"
    if manifest_src.exists():
        shutil.copy(manifest_src, out_dir / "manifest.json")

    print(f"  [{pct}%] {total} → {n_sample} traj, 写入 {out_dir}")
    return n_sample


if __name__ == "__main__":
    print(f"=== S3 数据集子采样（D-B 共 900 traj，seed={SEED}）===")
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    for pct in PERCENTAGES:
        out_dir = OUT_BASE / f"dataset_{pct}pct"
        n = subsample_dataset(SRC_DIR, out_dir, pct, seed=SEED)
        print(f"  => {out_dir}  ({n} 个 pkl 符号链接)")

    print("\n=== 完成 ===")
    print("训练时指定 dataset_dir=data/channel_generality/sensitivity/s3_dataset/dataset_Npct")
