#!/usr/bin/env python3
"""Phase 3 Fig.A: cross-scenario bar charts with fixed method mapping."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = [5, 6, 7, 8, 9]
SCENARIO_LABEL = {5: "A", 6: "B", 7: "C", 8: "D", 9: "E"}

METHODS = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M5b",
    "M6",
    "M6b",
]

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

METRICS = [
    ("nhp_dist_mean", "NHP distance"),
    ("nhp_viol_mean", "NHP violations"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=float)))


def _scenario_from_seed_summary(seed_summary: dict) -> dict[str, float]:
    return {
        "hp_dist_mean": float(seed_summary.get("hp_dist_mean", float("nan"))),
        "nhp_dist_mean": float(seed_summary.get("nhp_dist_mean", float("nan"))),
        "hp_viol_mean": float(seed_summary.get("hp_viol_mean", float("nan"))),
        "nhp_viol_mean": float(seed_summary.get("nhp_viol_mean", float("nan"))),
    }


RAW_ROOTS = {
    "M1": "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100/metric_raw",
    "M3": "data/channel_generality/ppo_teacher_5seed/eval_ood/metric_raw",
    "M4": "data/channel_generality/ppo_multi/metric_raw",
}
_NPZ_METRIC_MAP = {
    "nhp_viol_mean": "nhp_viols",
    "hp_viol_mean":  "hp_viols",
    "nhp_dist_mean": "step_nhp_dist",
    "hp_dist_mean":  "step_hp_dist",
}


def _load_raw_per_seed(raw_root: Path, scenario: int) -> dict[str, list[float]]:
    """Return per-seed means for each metric by reading ep_seed*.npz files."""
    out: dict[str, list[float]] = {k: [] for k in _NPZ_METRIC_MAP}
    for p in sorted(raw_root.glob(f"scenario_{scenario}/ep_seed*.npz")):
        npz = np.load(p)
        for metric_key, npz_key in _NPZ_METRIC_MAP.items():
            if npz_key not in npz.files:
                continue
            arr = npz[npz_key].astype(float).ravel()
            out[metric_key].append(float(np.mean(arr)))
    return out


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=0))


def load_series_for_method(
    base: Path, method: str, repo: Path
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """
    Returns (means_per_scenario, stds_per_scenario).
    Each is a dict {scenario_id -> {metric_key -> float}}.
    """
    means: dict[int, dict[str, float]] = {}
    stds: dict[int, dict[str, float]] = {}

    # M1/M3/M4: per-seed npz files in metric_raw.
    if method in RAW_ROOTS:
        raw_root = repo / RAW_ROOTS[method]
        for s in SCENARIOS:
            per_seed = _load_raw_per_seed(raw_root, s)
            m_row, s_row = {}, {}
            for mk in _NPZ_METRIC_MAP:
                mn, sd = _mean_std(per_seed[mk])
                m_row[mk] = mn
                s_row[mk] = sd
            means[s] = m_row
            stds[s] = s_row
        return means, stds

    # M5/M5b/M6/M6b: metric_json/scenario_X/seed_*/summary.json.
    metric_json = base / "metric_json"
    if metric_json.is_dir():
        for s in SCENARIOS:
            sdir = metric_json / f"scenario_{s}"
            seed_rows: dict[str, list[float]] = {k: [] for k in _NPZ_METRIC_MAP}
            for seed_dir in sorted(sdir.glob("seed_*")):
                d = load_json(seed_dir / "summary.json")
                if d is None:
                    continue
                for mk in _NPZ_METRIC_MAP:
                    v = d.get(mk)
                    if v is not None:
                        seed_rows[mk].append(float(v))
            m_row, s_row = {}, {}
            for mk in _NPZ_METRIC_MAP:
                mn, sd = _mean_std(seed_rows[mk])
                m_row[mk] = mn
                s_row[mk] = sd
            means[s] = m_row
            stds[s] = s_row
        return means, stds

    # M2: single-seed direct summary.json under scenario_X/.
    for s in SCENARIOS:
        p = base / f"scenario_{s}" / "summary.json"
        d = load_json(p)
        if d is None:
            means[s] = {k: float("nan") for k in _NPZ_METRIC_MAP}
        else:
            means[s] = _scenario_from_seed_summary(d)
        stds[s] = {k: float("nan") for k in _NPZ_METRIC_MAP}

    # ppo_multi fallback: global_summary with std fields.
    if not any(np.isfinite(v) for r in means.values() for v in r.values()):
        gsum = load_json(base / "metric_json" / "global_summary.json")
        if gsum and "scenarios" in gsum:
            for s in SCENARIOS:
                row = gsum["scenarios"].get(f"scenario_{s}")
                if row is None:
                    continue
                means[s] = _scenario_from_seed_summary(row)
                stds[s] = {
                    "hp_dist_mean":  float(row.get("hp_dist_std",  float("nan"))),
                    "nhp_dist_mean": float(row.get("nhp_dist_std", float("nan"))),
                    "hp_viol_mean":  float(row.get("hp_viol_std",  float("nan"))),
                    "nhp_viol_mean": float(row.get("nhp_viol_std", float("nan"))),
                }

    return means, stds


def draw(out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    rr = repo_root()
    all_means: dict[str, dict[int, dict[str, float]]] = {}
    all_stds: dict[str, dict[int, dict[str, float]]] = {}
    for m in METHODS:
        mn, sd = load_series_for_method(rr / ROOTS[m], m, rr)
        if not mn:
            print(f"[warn] no data for {m}: {ROOTS[m]}")
            continue
        all_means[m] = mn
        all_stds[m] = sd

    if not all_means:
        raise RuntimeError("No method data loaded for Fig.A.")

    methods_loaded = [m for m in METHODS if m in all_means]
    # Keep only NHP panels and widen bars for readability.
    n = len(methods_loaded)
    width = 0.17
    group_span = n * width
    group_gap = 0.55
    centers = np.arange(len(SCENARIOS)) * (group_span + group_gap)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
    ]
    hatches = ["", "///", "\\\\\\", "xxx", "..", "++", "oo", "**"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes = np.atleast_1d(axes)
    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        # Alternate background by scenario group for easier visual grouping.
        for gi, c in enumerate(centers):
            left = c - group_span / 2 - width * 0.6
            right = c + group_span / 2 + width * 0.6
            if gi % 2 == 0:
                ax.axvspan(left, right, color="gray", alpha=0.06, zorder=0)

        for i, m in enumerate(methods_loaded):
            ys  = [all_means[m].get(s, {}).get(metric_key, float("nan")) for s in SCENARIOS]
            errs = [all_stds[m].get(s, {}).get(metric_key, float("nan"))  for s in SCENARIOS]
            # Replace nan std with 0 so bar() doesn't crash; skip errorbar if all nan.
            errs_clean = [e if np.isfinite(e) else 0.0 for e in errs]
            has_err = any(np.isfinite(e) and e > 0 for e in errs)
            xpos = centers + offsets[i]
            bars = ax.bar(
                xpos,
                ys,
                width=width * 0.9,
                label=METHOD_LABEL[m],
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.6,
                hatch=hatches[i % len(hatches)],
                zorder=3,
            )
            if has_err:
                ax.errorbar(
                    xpos,
                    ys,
                    yerr=errs_clean,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=0.9,
                    zorder=5,
                )
            # Zero values are hard to see as bars; add baseline marker for visibility.
            for b, y in zip(bars, ys):
                if np.isfinite(y) and abs(y) < 1e-12:
                    ax.plot(
                        [b.get_x() + b.get_width() / 2],
                        [0.0],
                        marker="o",
                        markersize=3.8,
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markeredgewidth=0.7,
                        zorder=6,
                    )
        ax.set_xticks(centers)
        ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIOS])
        ax.set_xlabel("Scenario")
        ax.set_title(metric_title)
        ax.grid(axis="y", alpha=0.28, zorder=1)

        # Keep some side margins to avoid clipped first/last groups.
        ax.set_xlim(centers[0] - group_span / 2 - 0.25, centers[-1] + group_span / 2 + 0.25)
        all_vals = [
            all_means[m].get(s, {}).get(metric_key, float("nan"))
            for m in methods_loaded
            for s in SCENARIOS
        ]
        finite_vals = [v for v in all_vals if np.isfinite(v)]

        if finite_vals:
            vmin, vmax = min(finite_vals), max(finite_vals)
            margin = (vmax - vmin) * 0.1 if vmax != vmin else abs(vmax) * 0.1 or 0.1

            if vmin >= 0:
                # 全正值：底部从 0 开始，顶部留 margin
                ax.set_ylim(bottom=0, top=vmax + margin)
            elif vmax <= 0:
                # 全负值：顶部到 0，底部留 margin
                ax.set_ylim(bottom=vmin - margin, top=0)
            else:
                # 正负混合：两端都留 margin
                ax.set_ylim(bottom=vmin - margin, top=vmax + margin)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    print("\n[Fig.A] loaded methods:", ", ".join(METHOD_LABEL[m] for m in methods_loaded))
    for s in SCENARIOS:
        print(f"  {SCENARIO_LABEL[s]}(s{s}) ok")
    print("[Fig.A] error bars: available for methods with >1 seed; M2 has single seed → no errorbar")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures/channel_generality",
        help="Figure output directory",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="cross_scenario_bar",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    out_dir = (repo_root() / args.output_dir).resolve()
    draw(out_dir / f"{args.out_prefix}.pdf")


if __name__ == "__main__":
    main()
