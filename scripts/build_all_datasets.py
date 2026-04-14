"""
构建 5 个正式数据集 D-A/B/C/D/E，每个数据集 180 条/场景 × 5 场景 = 900 条。
所有数据集大小一致，确保 10 组 DT 实验的数据量公平可比。

D-A: mlp_expert only                      (180/scen from mlp_expert)
D-B: attn_expert only                     (180/scen from attn_expert)
D-C: attn_expert + attn_joint mix 1:1     (90/scen each)
D-E: mlp_expert + mlp_joint mix 1:1      (90/scen each)
D-D: all 4 pools mix 1:1:1:1             (45/scen each)
"""
import argparse
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.collect_data_v2 import compute_metadata

SRC  = os.path.join(ROOT, "data/channel_generality/dt_v2_8exp/dataset_sources")
DEST = os.path.join(ROOT, "data/channel_generality/dt_v2_8exp")
SCENARIOS = [0, 1, 2, 3, 4]
# Use all available trajectories per pool (no artificial budget cap).
# D-A/D-B: 900 each; D-C/D-E: 1800 each; D-D: 3600.
# This matches real-world usage where all data is used.
TRAJ_PER_SCEN = None   # None = use all available
SEED = 42


def scan_pool(pool_dir):
    """Return {scenario: [file_entry, ...]} for all pkl files in pool_dir/training/."""
    train_dir = os.path.join(pool_dir, "training")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"training dir not found: {train_dir}")
    grouped = defaultdict(list)
    for fname in sorted(f for f in os.listdir(train_dir) if f.endswith(".pkl")):
        fpath = os.path.join(train_dir, fname)
        with open(fpath, "rb") as fp:
            t = pickle.load(fp)
        scen = int(t.get("meta_scen", -1))
        grouped[scen].append({
            "path": fpath,
            "scenario": scen,
            "teacher_kind": t.get("meta_teacher_kind", "unknown"),
            "teacher_id":   t.get("meta_teacher_id",   "unknown"),
            "episode_return": float(t.get("episode_returns", 0.0)),
            "basename": fname,
        })
    return grouped


def safe_tag(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def build_dataset(name, pool_configs, out_dir):
    """
    pool_configs: list of (pool_grouped, n_per_scen)
    Samples n_per_scen trajectories from each pool per scenario,
    copies them into out_dir/training/ with renamed files.
    """
    if os.path.exists(out_dir):
        print(f"  [SKIP] already exists: {out_dir}")
        return
    train_dir = os.path.join(out_dir, "training")
    os.makedirs(train_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    selected = []
    summary = defaultdict(lambda: defaultdict(int))

    for scen in SCENARIOS:
        for pool_grouped, n_need in pool_configs:
            entries = pool_grouped.get(scen, [])
            if n_need is None:
                chosen = entries          # use all
            else:
                if len(entries) < n_need:
                    raise ValueError(
                        f"Dataset {name} S{scen}: need {n_need} but pool only has {len(entries)}"
                    )
                idx = rng.choice(len(entries), size=n_need, replace=False)
                chosen = [entries[i] for i in sorted(idx)]
            selected.extend(chosen)
            for e in chosen:
                summary[scen][e["teacher_id"]] += 1

    manifest = {
        "dataset_name": name,
        "traj_per_scenario": "all",
        "n_trajectories": len(selected),
        "selection_summary": {
            str(s): dict(sorted(v.items())) for s, v in sorted(summary.items())
        },
        "files": [],
    }

    for idx, e in enumerate(tqdm(selected, desc=f"Copy {name}")):
        pfx = f"S{e['scenario']}_{safe_tag(e['teacher_kind'])}_{safe_tag(e['teacher_id'])}"
        dst_name = f"{pfx}_idx{idx:05d}_{e['basename']}"
        dst_path = os.path.join(train_dir, dst_name)
        shutil.copy2(e["path"], dst_path)
        manifest["files"].append({
            "dst_name": dst_name,
            "scenario": e["scenario"],
            "teacher_kind": e["teacher_kind"],
            "teacher_id": e["teacher_id"],
            "episode_return": e["episode_return"],
        })

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Recomputing metadata for {name}...")
    compute_metadata(out_dir)
    print(f"  Done: {len(selected)} trajectories → {out_dir}")


def main():
    print("Scanning source pools...")
    mlp_e  = scan_pool(os.path.join(SRC, "mlp_expert"))
    attn_e = scan_pool(os.path.join(SRC, "attn_expert"))
    mlp_j  = scan_pool(os.path.join(SRC, "mlp_joint"))
    attn_j = scan_pool(os.path.join(SRC, "attn_joint"))

    # Verify pool sizes
    for pname, pool in [("mlp_expert", mlp_e), ("attn_expert", attn_e),
                        ("mlp_joint", mlp_j), ("attn_joint", attn_j)]:
        total = sum(len(v) for v in pool.values())
        print(f"  {pname}: {total} total trajs ({[len(pool.get(s,[])) for s in SCENARIOS]} per scen)")

    # None = use all available trajectories from each pool (no subsampling)
    N = None
    datasets = [
        # name,  pool_configs (list of (pool, n_per_scen or None=all))
        ("D-A",  [(mlp_e,  N)]),                              # 900
        ("D-B",  [(attn_e, N)]),                              # 900
        ("D-C",  [(attn_e, N), (attn_j, N)]),                 # 1800
        ("D-E",  [(mlp_e,  N), (mlp_j,  N)]),                 # 1800
        ("D-D",  [(attn_e, N), (attn_j, N), (mlp_e, N), (mlp_j, N)]),  # 3600
    ]

    print("\nBuilding datasets...")
    for name, pool_configs in datasets:
        out = os.path.join(DEST, f"dataset_{name}")
        print(f"\n[{name}] → {out}")
        build_dataset(name, pool_configs, out)

    print("\n=== All 5 datasets built ===")
    for name, _ in datasets:
        out = os.path.join(DEST, f"dataset_{name}", "training")
        n = len(os.listdir(out)) if os.path.isdir(out) else 0
        print(f"  {name}: {n} trajectories")


if __name__ == "__main__":
    main()
