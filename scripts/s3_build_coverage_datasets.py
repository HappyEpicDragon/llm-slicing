"""
S3 场景覆盖度子数据集构建：按累加训练场景数拆分 D-B 数据集。

覆盖方式（确定性，无随机采样）：
  scen1: S0 only        → 180 traj
  scen2: S0 + S1        → 360 traj
  scen3: S0 + S1 + S2   → 540 traj
  scen4: S0 + S1 + S2 + S3 → 720 traj
  scen5: 全部（=D-B 原始） → 复用 b2_e2 结果，不创建

obs_stats 使用 D-B 全集统计量（消融只改场景数，不改 normalization）。
scenario_stats 仅统计子集包含的场景。
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

SRC_DIR = Path("data/channel_generality/dt_v2_8exp/dataset_D-B")
OUT_BASE = Path("data/channel_generality/sensitivity/s3_coverage")

COVERAGE_CONFIGS = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 2],
    4: [0, 1, 2, 3],
}


def build_coverage_dataset(src_dir: Path, out_dir: Path, scenario_ids: list[int]):
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

    print(f"  [scen{len(scenario_ids)}] scenarios={scenario_ids} → {len(selected)} traj → {out_dir}")
    return len(selected)


if __name__ == "__main__":
    print("=== S3 场景覆盖度子数据集构建 ===")
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    for n_scen, scen_ids in COVERAGE_CONFIGS.items():
        out_dir = OUT_BASE / f"scen{n_scen}"
        build_coverage_dataset(SRC_DIR, out_dir, scen_ids)

    print("\n=== 完成 ===")
