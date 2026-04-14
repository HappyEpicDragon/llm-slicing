"""
将 curated dataset 的动作格式从语义 token（template_id）转换为 raw hybrid action（α_1..α_5）。
用于 A1 消融实验：IDT-RawAction 变体。

原始动作格式：[template_id, j1, j2, j3, j4, j5]
转换后格式：  [α_1, α_2, α_3, α_4, α_5, j1, j2, j3, j4, j5]
  其中 α_i = inter_quota_i / total_prbs 为连续 resource fraction

用法：
    python scripts/convert_dataset_to_raw_action.py \
        --src_dir data/channel_generality/dt/dataset/training/ \
        --dst_dir data/channel_generality/dt/dataset_raw_action/training/
"""
import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from src.basic_apis.codebook_utils import build_dirichlet_inter_quota_codebook


# 与 HierarchicalSlicingEnv 保持一致的 template codebook
def _build_inter_quota_patterns(num_prbs: int = 135) -> list:
    patterns = build_dirichlet_inter_quota_codebook(
        num_slices=5,
        num_prbs=num_prbs,
        c_min=10,
        num_concentration_levels=10,
        num_mc_samples=10_000,
        beta_min=0.1,
        beta_max=50.0,
        seed=2025,
    )
    return [np.array(p, dtype=np.float32) for p in patterns]


INTER_QUOTA_PATTERNS = _build_inter_quota_patterns(135)
TOTAL_PRBS = 135.0


def convert_action(action: np.ndarray) -> np.ndarray:
    """
    将单步动作从 [template_id, j1..j5] 转换为 [α_1..α_5, j1..j5]。
    """
    template_id = int(action[0])
    intra_modes = action[1:]  # [j1..j5]

    pattern = INTER_QUOTA_PATTERNS[template_id]
    alpha = pattern / TOTAL_PRBS  # shape=(5,), normalized fraction

    return np.concatenate([alpha, intra_modes.astype(np.float32)])  # shape=(10,)


def convert_trajectory(traj: dict) -> dict:
    """将一条轨迹中所有 actions 字段转换为 raw action 格式"""
    new_traj = {k: v for k, v in traj.items()}  # 浅拷贝
    if 'actions' in traj:
        new_traj['actions'] = [convert_action(np.array(a)) for a in traj['actions']]
    return new_traj


def convert_dataset(src_dir: str, dst_dir: str) -> None:
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    pkl_files = sorted([f for f in src_path.iterdir() if f.suffix == '.pkl'])
    if not pkl_files:
        raise FileNotFoundError(f"在 {src_dir} 中未找到任何 .pkl 文件")

    print(f"Converting {len(pkl_files)} trajectories: {src_dir} → {dst_dir}")
    print(f"  Action format: [template_id, j1..j5] → [α_1..α_5, j1..j5]")

    dst_path.mkdir(parents=True, exist_ok=True)

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            traj = pickle.load(f)

        new_traj = convert_trajectory(traj)

        out_path = dst_path / pkl_file.name
        with open(out_path, 'wb') as f:
            pickle.dump(new_traj, f)

    print(f"Done. {len(pkl_files)} files written to {dst_dir}")
    # 验证示例
    first_file = dst_path / pkl_files[0].name
    with open(first_file, 'rb') as f:
        sample = pickle.load(f)
    if sample.get('actions'):
        print(f"  Sample action shape: {np.array(sample['actions'][0]).shape} "
              f"(expected: (10,))")


def main():
    parser = argparse.ArgumentParser(
        description="将 IDT dataset 动作格式转换为 raw hybrid action（A1 消融实验）"
    )
    parser.add_argument('--src_dir', required=True,
                        help='源 curated dataset 目录（含 .pkl 文件）')
    parser.add_argument('--dst_dir', required=True,
                        help='目标 raw_action dataset 目录')
    args = parser.parse_args()
    convert_dataset(args.src_dir, args.dst_dir)


if __name__ == '__main__':
    main()
