#!/usr/bin/env python3
"""Phase 3 Tab.A: main comparison table (8 methods, mean ± std over s5-s9)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SCENARIOS = [5, 6, 7, 8, 9]
METHODS = ["M1", "M2", "M3", "M4", "M5", "M5b", "M6", "M6b"]
METHOD_LABEL = {
    "M1": "IDT",
    "M2": "DT-Mix",
    "M3": "PPO-Discrete",
    "M4": "PPO-baseline",
    "M5": "CQL-Discrete",
    "M5b": "CQL-Mix",
    "M6": "PPO-Lagrangian-Discrete",
    "M6b": "PPO-Lagrangian-Mix",
}
ROOTS = {
    "M1": "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100",
    "M2": "data/channel_generality/dt_baseline_v2/eval_ood",
    "M3": "data/channel_generality/ppo_teacher_5seed/eval_ood",
    "M4": "data/channel_generality/ppo_multi",
    "M5": "data/channel_generality/cql_v3_10/cql_expert",
    "M5b": "data/channel_generality/cql_baseline/cql_expert",
    "M6": "data/channel_generality/ppo_lagrangian_v2",
    "M6b": "data/channel_generality/ppo_lagrangian_baseline",
}

METRICS = ["hp_dist_mean", "nhp_dist_mean", "hp_viol_mean", "nhp_viol_mean"]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _safe(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def load_scenario_metric(base: Path, scenario: int, metric: str) -> float:
    # 1) direct summary under scenario_x
    d = load_json(base / f"scenario_{scenario}" / "summary.json")
    if d is not None and metric in d:
        return float(d[metric])

    # 2) ppo_multi global
    g = load_json(base / "metric_json" / "global_summary.json")
    if g and "scenarios" in g:
        row = g["scenarios"].get(f"scenario_{scenario}")
        if row and metric in row:
            return float(row[metric])

    # 3) metric_json/scenario_x/seed_*/summary.json
    sdir = base / "metric_json" / f"scenario_{scenario}"
    vals: list[float] = []
    for sd in sorted(sdir.glob("seed_*")):
        jd = load_json(sd / "summary.json")
        if jd is None or metric not in jd:
            continue
        vals.append(float(jd[metric]))
    if vals:
        return float(np.mean(vals))

    return float("nan")


def build_rows() -> list[dict[str, str]]:
    rr = repo_root()
    rows: list[dict[str, str]] = []
    for m in METHODS:
        base = rr / ROOTS[m]
        row: dict[str, str] = {"method_id": m, "method": METHOD_LABEL[m]}
        for metric in METRICS:
            vals = [load_scenario_metric(base, s, metric) for s in SCENARIOS]
            vals = [v for v in vals if not np.isnan(v)]
            mean, std = _safe(vals) if vals else (float("nan"), float("nan"))
            row[metric + "_mean"] = f"{mean:.6f}" if not np.isnan(mean) else "nan"
            row[metric + "_std"] = f"{std:.6f}" if not np.isnan(std) else "nan"
            row[metric + "_mean_pm_std"] = (
                f"{mean:.6f} ± {std:.6f}" if not np.isnan(mean) else "nan"
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/tables/channel_generality/main_table.csv",
    )
    args = parser.parse_args()

    out_csv = (repo_root() / args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    fields = [
        "method_id",
        "method",
        "hp_dist_mean_pm_std",
        "nhp_dist_mean_pm_std",
        "hp_viol_mean_pm_std",
        "nhp_viol_mean_pm_std",
        "hp_dist_mean",
        "nhp_dist_mean",
        "hp_viol_mean",
        "nhp_viol_mean",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    print(f"Saved: {out_csv}")
    print("[Tab.A] methods:", ", ".join(r["method"] for r in rows))


if __name__ == "__main__":
    main()

