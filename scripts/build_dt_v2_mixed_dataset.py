"""
Build DT V2 dataset variants from separate per-scenario and joint teacher pools.

The script keeps source pools immutable and materializes formal training datasets
with freshly recomputed metadata for each variant.

Examples:
    pixi run python -u scripts/build_dt_v2_mixed_dataset.py --variant per_scenario
    pixi run python -u scripts/build_dt_v2_mixed_dataset.py --variant joint
    pixi run python -u scripts/build_dt_v2_mixed_dataset.py --variant mix --ratio 1:1
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


def parse_ratio(raw):
    try:
        left, right = raw.split(":")
        left = int(left)
        right = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Ratio must look like 1:1 or 2:1") from exc

    if left < 0 or right < 0 or (left + right) == 0:
        raise argparse.ArgumentTypeError("Ratio must be non-negative and not all-zero")
    return left, right


def safe_tag(raw):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))


def scan_source_pool(source_dir):
    train_dir = os.path.join(source_dir, "training")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"training dir not found: {train_dir}")

    grouped = defaultdict(list)
    files = sorted(f for f in os.listdir(train_dir) if f.endswith(".pkl"))
    print(f"Scanning {train_dir} ({len(files)} files)")

    for file_name in tqdm(files, desc=f"Scan {os.path.basename(source_dir)}"):
        full_path = os.path.join(train_dir, file_name)
        with open(full_path, "rb") as f:
            traj = pickle.load(f)

        scen = int(traj.get("meta_scen", -1))
        entry = {
            "path": full_path,
            "scenario": scen,
            "teacher_kind": traj.get("meta_teacher_kind", "unknown"),
            "teacher_id": traj.get("meta_teacher_id", "unknown"),
            "episode_return": float(traj.get("episode_returns", 0.0)),
            "basename": file_name,
        }
        grouped[scen].append(entry)

    return grouped


def choose_entries(entries, n_select, rng):
    if n_select == 0:
        return []
    if len(entries) < n_select:
        raise ValueError(f"Need {n_select} entries but only found {len(entries)}")
    picked = rng.choice(len(entries), size=n_select, replace=False)
    return [entries[i] for i in sorted(picked)]


def build_selection(variant, per_pool, joint_pool, scenarios, total_per_scenario, ratio, seed):
    rng = np.random.default_rng(seed)
    selected = []
    summary = defaultdict(lambda: defaultdict(int))

    per_weight, joint_weight = ratio
    total_weight = per_weight + joint_weight

    for scen in scenarios:
        per_entries = per_pool.get(scen, [])
        joint_entries = joint_pool.get(scen, [])

        if variant == "per_scenario":
            per_need, joint_need = total_per_scenario, 0
        elif variant == "joint":
            per_need, joint_need = 0, total_per_scenario
        else:
            per_need = int(round(total_per_scenario * per_weight / total_weight))
            joint_need = total_per_scenario - per_need

        per_selected = choose_entries(per_entries, per_need, rng)
        joint_selected = choose_entries(joint_entries, joint_need, rng)

        selected.extend(per_selected)
        selected.extend(joint_selected)
        summary[scen]["per_scenario"] += len(per_selected)
        summary[scen]["joint"] += len(joint_selected)

    return selected, summary


def derive_output_dir(output_base, variant, ratio):
    if variant == "per_scenario":
        name = "dataset_per_scenario"
    elif variant == "joint":
        name = "dataset_joint"
    else:
        name = f"dataset_mix_{ratio[0]}to{ratio[1]}"
    return os.path.join(output_base, name)


def write_dataset(output_dir, variant, ratio, total_per_scenario, selected, summary):
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output already exists: {output_dir}")

    train_dir = os.path.join(output_dir, "training")
    os.makedirs(train_dir, exist_ok=True)

    manifest = {
        "variant": variant,
        "ratio": f"{ratio[0]}:{ratio[1]}",
        "total_per_scenario": total_per_scenario,
        "n_trajectories": len(selected),
        "selection_summary": {
            str(scen): dict(sorted(source_counts.items()))
            for scen, source_counts in sorted(summary.items())
        },
        "files": [],
    }

    for idx, entry in enumerate(tqdm(selected, desc=f"Copy {os.path.basename(output_dir)}")):
        prefix = (
            f"S{entry['scenario']}_{safe_tag(entry['teacher_kind'])}_"
            f"{safe_tag(entry['teacher_id'])}"
        )
        dst_name = f"{prefix}_idx{idx:05d}_{entry['basename']}"
        dst_path = os.path.join(train_dir, dst_name)
        shutil.copy2(entry["path"], dst_path)
        manifest["files"].append(
            {
                "dst_name": dst_name,
                "scenario": entry["scenario"],
                "teacher_kind": entry["teacher_kind"],
                "teacher_id": entry["teacher_id"],
                "episode_return": entry["episode_return"],
                "src_path": os.path.relpath(entry["path"], ROOT),
            }
        )

    with open(os.path.join(output_dir, "selection_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nRecomputing metadata...")
    compute_metadata(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per_source",
        type=str,
        default="data/channel_generality/dt_v2_mlp_ent05/dataset_sources/per_scenario",
    )
    parser.add_argument(
        "--joint_source",
        type=str,
        default="data/channel_generality/dt_v2_mlp_ent05/dataset_sources/joint",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="data/channel_generality/dt_v2_mlp_ent05",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional explicit output directory; otherwise derived from variant",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["per_scenario", "joint", "mix"],
        required=True,
    )
    parser.add_argument("--ratio", type=parse_ratio, default=(1, 1))
    parser.add_argument("--total_per_scenario", type=int, default=200)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or derive_output_dir(args.output_base, args.variant, args.ratio)

    if args.variant == "per_scenario":
        per_pool = scan_source_pool(os.path.join(ROOT, args.per_source))
        joint_pool = {}
    elif args.variant == "joint":
        per_pool = {}
        joint_pool = scan_source_pool(os.path.join(ROOT, args.joint_source))
    else:
        per_pool = scan_source_pool(os.path.join(ROOT, args.per_source))
        joint_pool = scan_source_pool(os.path.join(ROOT, args.joint_source))

    selected, summary = build_selection(
        variant=args.variant,
        per_pool=per_pool,
        joint_pool=joint_pool,
        scenarios=args.scenarios,
        total_per_scenario=args.total_per_scenario,
        ratio=args.ratio,
        seed=args.seed,
    )

    print("\nSelection summary:")
    for scen in sorted(summary.keys()):
        counts = summary[scen]
        print(
            f"  S{scen}: per_scenario={counts.get('per_scenario', 0)} "
            f"joint={counts.get('joint', 0)}"
        )
    print(f"  Total trajectories: {len(selected)}")
    print(f"  Output dir: {output_dir}")

    write_dataset(
        output_dir=output_dir,
        variant=args.variant,
        ratio=args.ratio,
        total_per_scenario=args.total_per_scenario,
        selected=selected,
        summary=summary,
    )


if __name__ == "__main__":
    main()
