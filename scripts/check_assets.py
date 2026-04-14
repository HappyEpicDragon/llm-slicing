#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _check_exists(path: Path, missing: list[str], label: str) -> None:
    if not path.exists():
        missing.append(f"{label}: {path}")


def check_model(path: str) -> list[str]:
    missing: list[str] = []
    _check_exists(Path(path), missing, "model")
    return missing


def check_metric_tree(
    save_root: str,
    scenarios: list[int],
    seeds: list[int],
    require_raw: bool = True,
    require_global: bool = True,
) -> list[str]:
    missing: list[str] = []
    root = Path(save_root)
    required_json = [
        "hp_violations.json",
        "nhp_violations.json",
        "hp_distance.json",
        "nhp_distance.json",
        "episode_rewards.json",
    ]
    for scen in scenarios:
        for seed in seeds:
            seed_dir = root / "metric_json" / f"scenario_{scen}" / f"seed_{seed}"
            for filename in required_json:
                _check_exists(seed_dir / filename, missing, "metric_json")
            if require_raw:
                _check_exists(root / "metric_raw" / f"scenario_{scen}" / f"ep_seed{seed}.npz", missing, "metric_raw")
        _check_exists(root / "metric_json" / f"scenario_{scen}" / "summary.json", missing, "scenario_summary")
    if require_global:
        _check_exists(root / "metric_json" / "global_summary.json", missing, "global_summary")
    return missing


def check_dataset(root: str) -> list[str]:
    missing: list[str] = []
    base = Path(root)
    _check_exists(base / "training", missing, "dataset_training")
    _check_exists(base / "evaluating", missing, "dataset_evaluating")
    _check_exists(base / "metadata.json", missing, "dataset_metadata")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Asset integrity checker")
    parser.add_argument("--check", choices=["model", "metrics", "dataset"], required=True)
    parser.add_argument("--path", required=True, help="Path for model or dataset, or save_root for metrics")
    parser.add_argument("--scenarios", default="5,6,7,8,9", help="Comma-separated scenario IDs for metrics")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed IDs for metrics")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    parser.add_argument("--require-raw", action="store_true", default=False, help="Require metric_raw/*.npz files")
    parser.add_argument("--require-global", action="store_true", default=False, help="Require metric_json/global_summary.json")
    args = parser.parse_args()

    if args.check == "model":
        missing = check_model(args.path)
    elif args.check == "dataset":
        missing = check_dataset(args.path)
    else:
        scenarios = [int(x) for x in args.scenarios.split(",") if x.strip()]
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        missing = check_metric_tree(
            args.path,
            scenarios,
            seeds,
            require_raw=bool(args.require_raw),
            require_global=bool(args.require_global),
        )

    ok = len(missing) == 0
    payload = {"ok": ok, "missing_count": len(missing), "missing": missing}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if ok:
            print("OK: all required assets found.")
        else:
            print(f"MISSING: {len(missing)} items")
            for item in missing:
                print(f"- {item}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

