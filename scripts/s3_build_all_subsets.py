"""
为 S3 穷举实验构建所有 C(5,k) 子集数据集 (k=1..4)。
共 30 个子集，跳过已存在的。
"""
import json
import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

SRC_DIR = Path("data/channel_generality/dt_v2_8exp/dataset_D-B")
OUT_BASE = Path("data/channel_generality/sensitivity/s3_exhaustive")

ALL_SCENARIOS = [0, 1, 2, 3, 4]


def subset_key(scenario_ids: list[int]) -> str:
    return "_".join(str(s) for s in sorted(scenario_ids))


def build_subset(src_dir: Path, out_dir: Path, scenario_ids: list[int]):
    src_train = src_dir / "training"
    all_files = sorted(f for f in src_train.iterdir() if f.suffix == ".pkl")

    prefixes = {f"S{s}_" for s in scenario_ids}
    selected = [f for f in all_files if any(f.name.startswith(p) for p in prefixes)]

    out_train = out_dir / "training"
    out_train.mkdir(parents=True, exist_ok=True)

    for f in selected:
        link = out_train / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())

    with open(src_dir / "metadata.json") as fp:
        meta = json.load(fp)

    scenario_returns: dict[str, list] = {}
    for f in selected:
        try:
            with open(f, "rb") as fp2:
                traj = pickle.load(fp2)
            rewards = np.array(traj.get("rewards", []), dtype=np.float32)
            ep_ret = float(rewards.sum())
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
            "p95": float(np.percentile(rets, 95)),
        }
    meta["scenario_stats"] = updated_stats

    with open(out_dir / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    return len(selected)


if __name__ == "__main__":
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    total = 0
    for k in range(1, 5):
        for combo in combinations(ALL_SCENARIOS, k):
            key = subset_key(list(combo))
            out_dir = OUT_BASE / f"k{k}_{key}"
            if (out_dir / "metadata.json").exists():
                print(f"  [SKIP] k{k}_{key} already exists")
                total += 1
                continue
            n = build_subset(SRC_DIR, out_dir, list(combo))
            print(f"  [OK] k{k}_{key}: {n} traj")
            total += 1
    print(f"\n=== Done: {total} subsets ===")
