#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path

from check_assets import check_model, check_dataset, check_metric_tree


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _run_checks(
    pre_models: list[str],
    pre_datasets: list[str],
    post_metrics: list[str],
    post_models: list[str],
    post_datasets: list[str],
    scenarios: list[int],
    seeds: list[int],
    require_raw: bool,
    require_global: bool,
    stage: str,
) -> bool:
    missing: list[str] = []
    for m in pre_models if stage == "pre" else post_models:
        missing.extend(check_model(m))
    for d in pre_datasets if stage == "pre" else post_datasets:
        missing.extend(check_dataset(d))
    if stage == "post":
        for root in post_metrics:
            missing.extend(check_metric_tree(root, scenarios, seeds, require_raw=require_raw, require_global=require_global))

    if missing:
        print(f"[{stage}] asset checks failed: {len(missing)} missing items")
        for item in missing:
            print(f"- {item}")
        return False
    print(f"[{stage}] asset checks passed.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run command with pre/post asset checks.")
    parser.add_argument("--pre-model", action="append", default=[])
    parser.add_argument("--pre-dataset", action="append", default=[])
    parser.add_argument("--post-model", action="append", default=[])
    parser.add_argument("--post-dataset", action="append", default=[])
    parser.add_argument("--post-metrics", action="append", default=[])
    parser.add_argument("--scenarios", default="5,6,7,8,9")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--require-raw", action="store_true", default=False)
    parser.add_argument("--require-global", action="store_true", default=False)
    parser.add_argument("--command", required=True, help="Full command string to execute.")
    args = parser.parse_args()

    scenarios = _parse_csv_ints(args.scenarios)
    seeds = _parse_csv_ints(args.seeds)

    if not _run_checks(
        pre_models=args.pre_model,
        pre_datasets=args.pre_dataset,
        post_metrics=args.post_metrics,
        post_models=args.post_model,
        post_datasets=args.post_dataset,
        scenarios=scenarios,
        seeds=seeds,
        require_raw=bool(args.require_raw),
        require_global=bool(args.require_global),
        stage="pre",
    ):
        return 2

    print(f"[run] {args.command}")
    result = subprocess.run(shlex.split(args.command), cwd=str(Path.cwd()))
    if result.returncode != 0:
        print(f"[run] command failed with exit code {result.returncode}")
        return result.returncode

    if not _run_checks(
        pre_models=args.pre_model,
        pre_datasets=args.pre_dataset,
        post_metrics=args.post_metrics,
        post_models=args.post_model,
        post_datasets=args.post_dataset,
        scenarios=scenarios,
        seeds=seeds,
        require_raw=bool(args.require_raw),
        require_global=bool(args.require_global),
        stage="post",
    ):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

