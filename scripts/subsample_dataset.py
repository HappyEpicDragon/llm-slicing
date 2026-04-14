"""
数据集规模子采样脚本（对应实验 N1）。

从已有的 curated dataset 中按 sort_key 截断到指定比例，生成更小规模的数据集变体。

用法：
    python scripts/subsample_dataset.py --ratio 0.1 \
        --src_dir data/channel_generality/dt/dataset/training/ \
        --dst_dir data/channel_generality/dt/dataset_10pct/training/

    # 批量生成多个比例
    for ratio in 0.1 0.3 0.5; do
        python scripts/subsample_dataset.py --ratio $ratio \
            --src_dir data/channel_generality/dt/dataset/training/ \
            --dst_dir data/channel_generality/dt/dataset_${ratio/./}pct/training/
    done
"""
import os
import math
import shutil
import argparse
import pickle
from pathlib import Path


def subsample_dataset(src_dir: str, dst_dir: str, ratio: float) -> None:
    """
    从 src_dir 中读取所有 pkl 文件，按文件名（字典序，即原始 sort_key 顺序）
    截断到 ratio * total 个文件，写入 dst_dir。

    Args:
        src_dir: 源 curated dataset 目录
        dst_dir: 目标子采样数据集目录
        ratio:   保留比例（0 < ratio <= 1.0）
    """
    if not 0 < ratio <= 1.0:
        raise ValueError(f"ratio 必须在 (0, 1] 之间，实际传入: {ratio}")

    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    pkl_files = sorted([f for f in src_path.iterdir() if f.suffix == '.pkl'])
    if not pkl_files:
        raise FileNotFoundError(f"在 {src_dir} 中未找到任何 .pkl 文件")

    total = len(pkl_files)
    keep_count = max(1, math.ceil(total * ratio))
    selected = pkl_files[:keep_count]

    print(f"Subsampling dataset: {total} → {keep_count} files (ratio={ratio:.0%})")
    print(f"  Source: {src_dir}")
    print(f"  Target: {dst_dir}")

    dst_path.mkdir(parents=True, exist_ok=True)

    for pkl_file in selected:
        shutil.copy2(pkl_file, dst_path / pkl_file.name)

    print(f"Done. {keep_count} files written to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="从 curated dataset 子采样到指定比例（N1 实验）"
    )
    parser.add_argument('--ratio', type=float, required=True,
                        help='保留比例，例如 0.1 表示 10%%')
    parser.add_argument('--src_dir', required=True,
                        help='源 curated dataset 目录（包含 .pkl 文件）')
    parser.add_argument('--dst_dir', required=True,
                        help='目标子采样数据集目录')
    args = parser.parse_args()

    subsample_dataset(args.src_dir, args.dst_dir, args.ratio)


if __name__ == '__main__':
    main()
