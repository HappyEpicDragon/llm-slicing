"""
一次性将 data/static/channel/ 下所有 .mat 文件转换为 .npz 格式。
原始 .mat 文件保留不动。

Usage:
    cd /root/decision_transformer_slicing
    pixi run python scripts/convert_channel_mat_to_npz.py [--workers 32] [--dry-run]
"""
import os
import sys
import argparse
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNEL_DIR = os.path.join(ROOT, "data", "static", "channel")


def convert_one(mat_path: str) -> tuple[str, bool, str]:
    """返回 (mat_path, success, error_msg)"""
    npz_path = mat_path.replace(".mat", ".npz")
    if os.path.exists(npz_path):
        return mat_path, True, "already_exists"
    try:
        with h5py.File(mat_path, "r") as f:
            data = f["target_cell_power"][:]
        np.savez_compressed(npz_path, target_cell_power=data)
        return mat_path, True, ""
    except Exception as e:
        return mat_path, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32, help="并行进程数")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件不执行转换")
    args = parser.parse_args()

    mat_files = sorted(Path(CHANNEL_DIR).rglob("*.mat"))
    total = len(mat_files)
    print(f"Found {total} .mat files in {CHANNEL_DIR}")

    already = sum(1 for f in mat_files if f.with_suffix(".npz").exists())
    print(f"Already converted: {already}, remaining: {total - already}")

    if args.dry_run:
        print("[dry-run] exit without converting.")
        return

    mat_strs = [str(f) for f in mat_files]
    failed = []
    skipped = 0

    with Pool(processes=args.workers) as pool:
        for path, ok, msg in tqdm(
            pool.imap_unordered(convert_one, mat_strs),
            total=total,
            desc="Converting",
        ):
            if msg == "already_exists":
                skipped += 1
            elif not ok:
                failed.append((path, msg))

    print(f"\nDone. skipped={skipped}, failed={len(failed)}, converted={total - skipped - len(failed)}")
    if failed:
        print("Failed files:")
        for p, e in failed:
            print(f"  {p}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
