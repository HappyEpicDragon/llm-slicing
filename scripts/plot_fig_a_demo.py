#!/usr/bin/env python3
"""Demo: Plan A-v2 – line chart (difficulty-sorted) + robustness scatter."""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = [5, 6, 7, 8, 9]
SCENARIO_LABEL = {5: "A", 6: "B", 7: "C", 8: "D", 9: "E"}

METHOD_ORDER = [
    "IDT", "PPO-Discrete", "PPO-baseline", "CQL-Discrete",
    "CQL-Mix", "PPO-Lag-Discrete", "DT-Mix", "PPO-Lag-Mix",
]

ROOTS = {
    "IDT":               "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100",
    "DT-Mix":            "data/channel_generality/dt_baseline_v2/eval_ood",
    "PPO-Discrete":      "data/channel_generality/ppo_teacher_5seed/eval_ood",
    "PPO-baseline":      "data/channel_generality/ppo_multi",
    "CQL-Discrete":      "data/channel_generality/cql_v3_10/cql_expert",
    "CQL-Mix":           "data/channel_generality/cql_baseline/cql_expert",
    "PPO-Lag-Discrete":  "data/channel_generality/ppo_lagrangian_v2",
    "PPO-Lag-Mix":       "data/channel_generality/ppo_lagrangian_baseline",
}

IS_RAW = {"IDT", "PPO-Discrete"}
WEAK = {"DT-Mix", "PPO-Lag-Mix"}

NPZ_MAP = {"nhp_viol": "nhp_viols", "nhp_dist": "step_nhp_dist"}
JSON_MAP = {"nhp_viol": "nhp_violations", "nhp_dist": "nhp_distance"}

STYLE = {
    "IDT":               {"color": "#D62728", "marker": "o",  "lw": 2.8, "ms": 9,   "zorder": 10},
    "PPO-Discrete":      {"color": "#2CA02C", "marker": "s",  "lw": 1.6, "ms": 6,   "zorder": 5},
    "PPO-baseline":      {"color": "#1F77B4", "marker": "^",  "lw": 1.6, "ms": 6,   "zorder": 5},
    "CQL-Discrete":      {"color": "#9467BD", "marker": "D",  "lw": 1.6, "ms": 5.5, "zorder": 5},
    "CQL-Mix":           {"color": "#8C564B", "marker": "p",  "lw": 1.6, "ms": 6,   "zorder": 5},
    "PPO-Lag-Discrete":  {"color": "#E377C2", "marker": "v",  "lw": 1.6, "ms": 6,   "zorder": 5},
    "DT-Mix":            {"color": "#AAAAAA", "marker": "x",  "lw": 1.2, "ms": 6,   "zorder": 2, "ls": "--"},
    "PPO-Lag-Mix":       {"color": "#BBBBBB", "marker": "+",  "lw": 1.2, "ms": 7,   "zorder": 2, "ls": "--"},
}

RC = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9.5,
    "font.family": "sans-serif",
}

# ── data loading ──────────────────────────────────────────────────────────

def _load_seed_means_raw(root: Path, scenario: int, metric: str) -> list[float]:
    npz_key = NPZ_MAP[metric]
    vals = []
    for p in sorted(root.glob(f"metric_raw/scenario_{scenario}/ep_seed*.npz")):
        npz = np.load(p)
        if npz_key in npz.files:
            vals.append(float(np.mean(npz[npz_key])))
    return vals


def _load_seed_means_json(root: Path, scenario: int, metric: str) -> list[float]:
    json_key = JSON_MAP[metric]
    vals = []
    mj = root / "metric_json" / f"scenario_{scenario}"
    for seed_dir in sorted(mj.glob("seed_*")):
        # Try individual metric JSON first (nhp_violations.json, etc.)
        metric_file = seed_dir / f"{json_key}.json"
        if metric_file.is_file():
            d = json.load(open(metric_file))
            if "mean" in d:
                vals.append(float(d["mean"]))
            elif json_key in d:
                vals.append(float(np.mean(d[json_key])))
            continue
        # Fallback: summary.json with nhp_viol_mean / nhp_dist_mean keys
        sm = seed_dir / "summary.json"
        if sm.is_file():
            d = json.load(open(sm))
            key_alt = metric.replace("nhp_viol", "nhp_viol_mean").replace("nhp_dist", "nhp_dist_mean")
            if key_alt in d:
                vals.append(float(d[key_alt]))
    return vals


def load_all() -> dict[str, dict[int, dict[str, list[float]]]]:
    """Returns {method -> {scenario -> {metric -> [per_seed_values]}}}."""
    data: dict[str, dict[int, dict[str, list[float]]]] = {}
    for name in METHOD_ORDER:
        root = REPO / ROOTS[name]
        data[name] = {}
        for s in SCENARIOS:
            row = {}
            for mk in ["nhp_viol", "nhp_dist"]:
                if name in IS_RAW:
                    row[mk] = _load_seed_means_raw(root, s, mk)
                else:
                    row[mk] = _load_seed_means_json(root, s, mk)
            data[name][s] = row
    return data


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=0))

# ── Plan A-v2: line chart + robustness scatter ────────────────────────────

def plan_a_v2(data: dict, out_dir: Path):
    plt.rcParams.update(RC)

    mk = "nhp_viol"

    # Compute per-method, per-scenario means (seed-averaged)
    method_scenario_mean: dict[str, dict[int, float]] = {}
    method_scenario_std: dict[str, dict[int, float]] = {}
    for name in METHOD_ORDER:
        method_scenario_mean[name] = {}
        method_scenario_std[name] = {}
        for s in SCENARIOS:
            m, sd = mean_std(data[name][s].get(mk, []))
            method_scenario_mean[name][s] = m
            method_scenario_std[name][s] = sd

    # Sort scenarios by baseline average (difficulty)
    baseline_names = [n for n in METHOD_ORDER if n != "IDT"]
    difficulty = {}
    for s in SCENARIOS:
        vals = [method_scenario_mean[bn][s] for bn in baseline_names
                if np.isfinite(method_scenario_mean[bn][s])]
        difficulty[s] = np.mean(vals) if vals else 0.0
    sorted_scenarios = sorted(SCENARIOS, key=lambda s: difficulty[s])
    x_labels = [SCENARIO_LABEL[s] for s in sorted_scenarios]

    # ── Figure layout: line chart (left, wide) + scatter (right, narrow)
    fig = plt.figure(figsize=(12.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.3], wspace=0.35)
    ax_line = fig.add_subplot(gs[0])
    ax_scatter = fig.add_subplot(gs[1])

    # ── Left: Line chart ──
    for name in METHOD_ORDER:
        st = STYLE[name]
        ys = [method_scenario_mean[name][s] for s in sorted_scenarios]
        errs = [method_scenario_std[name][s] for s in sorted_scenarios]
        ys_arr = np.array(ys)
        errs_arr = np.array(errs)
        is_weak = name in WEAK

        kw = dict(color=st["color"], marker=st["marker"],
                  linewidth=st["lw"], markersize=st["ms"],
                  zorder=st["zorder"], label=name,
                  linestyle=st.get("ls", "-"),
                  alpha=0.35 if is_weak else 1.0,
                  clip_on=True)
        ax_line.plot(range(len(sorted_scenarios)), ys_arr, **kw)

        has_err = np.any(np.isfinite(errs_arr) & (errs_arr > 0))
        if has_err and not is_weak:
            ax_line.fill_between(range(len(sorted_scenarios)),
                                 ys_arr - errs_arr, ys_arr + errs_arr,
                                 color=st["color"], alpha=0.10, zorder=1)

    ax_line.set_xticks(range(len(sorted_scenarios)))
    ax_line.set_xticklabels(x_labels)
    ax_line.set_xlabel("Scenario (sorted by avg. baseline violation →)")
    ax_line.set_ylabel("NHP Violations")
    ax_line.set_title("NHP Violations across OOD Scenarios")
    ax_line.grid(axis="y", alpha=0.25, zorder=0)
    ax_line.grid(axis="x", alpha=0.12, linestyle=":", zorder=0)
    ax_line.set_ylim(bottom=-0.008, top=0.42)

    ax_line.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92,
                   edgecolor="#ccc", fontsize=9, columnspacing=0.8)

    # ── Right: Mean vs Worst-case scatter ──
    for name in METHOD_ORDER:
        st = STYLE[name]
        is_weak = name in WEAK
        scenario_vals = [method_scenario_mean[name][s] for s in SCENARIOS]
        finite_vals = [v for v in scenario_vals if np.isfinite(v)]
        if not finite_vals:
            continue
        avg_val = np.mean(finite_vals)
        worst_val = np.max(finite_vals)

        ax_scatter.scatter(
            avg_val, worst_val,
            color=st["color"], marker=st["marker"],
            s=120 if name == "IDT" else 70,
            edgecolors="black" if name == "IDT" else "white",
            linewidths=1.5 if name == "IDT" else 0.5,
            alpha=0.35 if is_weak else 1.0,
            zorder=st["zorder"],
        )
        # Label offset
        offx, offy = 0.003, 0.003
        if name == "IDT":
            offx, offy = 0.005, -0.012
        elif name == "PPO-Lag-Mix":
            offx, offy = -0.10, 0.01
        elif name == "DT-Mix":
            offx, offy = -0.08, 0.01

        short = name.replace("PPO-Lag-", "Lag-")
        ax_scatter.annotate(
            short, (avg_val, worst_val),
            xytext=(avg_val + offx, worst_val + offy),
            fontsize=8.5, color=st["color"],
            fontweight="bold" if name == "IDT" else "normal",
            alpha=0.40 if is_weak else 0.90,
        )

    # Ideal region highlight
    ax_scatter.axhspan(-0.01, 0.03, color="#D6272820", zorder=0)
    ax_scatter.axvspan(-0.01, 0.03, color="#D6272820", zorder=0)
    ax_scatter.text(0.012, 0.015, "Ideal\nregion", fontsize=8, color="#D62728",
                    ha="center", va="center", alpha=0.6, style="italic")

    # y = x reference line (worst = mean → perfect consistency)
    lim_max = 0.45
    ax_scatter.plot([0, lim_max], [0, lim_max], "k--", alpha=0.15, lw=0.8, zorder=0)

    ax_scatter.set_xlabel("Mean Violation (across scenarios)")
    ax_scatter.set_ylabel("Worst-case Violation")
    ax_scatter.set_title("Robustness Profile")
    ax_scatter.set_xlim(-0.01, lim_max)
    ax_scatter.set_ylim(-0.01, lim_max)
    ax_scatter.set_aspect("equal")
    ax_scatter.grid(alpha=0.2, zorder=0)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        p = out_dir / f"demo_plan_a_v2_line_robust.{ext}"
        fig.savefig(p, format=ext, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plan A-v2] saved to {out_dir / 'demo_plan_a_v2_line_robust.png'}")


# ── Plan C: Strip + box plot ──────────────────────────────────────────────

def plan_c(data: dict, out_dir: Path):
    plt.rcParams.update(RC)

    metrics = [("nhp_dist", "NHP Distance (closer to 0 is better)"),
               ("nhp_viol", "NHP Violations (lower is better)")]

    scenario_markers = {5: "o", 6: "s", 7: "^", 8: "D", 9: "v"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    for ax, (mk, title) in zip(axes, metrics):
        box_data = []
        for name in METHOD_ORDER:
            vals = []
            for s in SCENARIOS:
                seed_vals = data[name][s].get(mk, [])
                if seed_vals:
                    vals.append(np.mean(seed_vals))
            box_data.append(vals if vals else [float("nan")])

        bp = ax.boxplot(box_data, positions=range(len(METHOD_ORDER)), widths=0.45,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(linewidth=0.9),
                        capprops=dict(linewidth=0.9), zorder=3)

        for i, (patch, name) in enumerate(zip(bp["boxes"], METHOD_ORDER)):
            c = STYLE[name]["color"]
            is_weak = name in WEAK
            is_ours = name == "IDT"
            patch.set_facecolor(c)
            patch.set_alpha(0.20 if is_weak else (0.50 if is_ours else 0.30))
            patch.set_edgecolor(c)
            patch.set_linewidth(2.5 if is_ours else 1.0)

        for i, name in enumerate(METHOD_ORDER):
            is_weak = name in WEAK
            for s in SCENARIOS:
                seed_vals = data[name][s].get(mk, [])
                if seed_vals:
                    y = np.mean(seed_vals)
                    jitter = np.random.default_rng(s + i * 10).uniform(-0.12, 0.12)
                    ax.scatter(i + jitter, y, marker=scenario_markers[s], s=40,
                               color=STYLE[name]["color"],
                               edgecolors="white", linewidths=0.5,
                               alpha=0.30 if is_weak else 0.85, zorder=5)

        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([n.replace("PPO-Lag-", "Lag-") for n in METHOD_ORDER],
                           rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.axhline(0, color="gray", linewidth=0.5, zorder=1)

        if mk == "nhp_dist":
            ax.set_ylim(top=0.005)
        if mk == "nhp_viol":
            ax.set_ylim(bottom=-0.01)

    import matplotlib.patches as mpatches
    legend_elements = [mpatches.Patch(facecolor=STYLE["IDT"]["color"], alpha=0.5,
                                       edgecolor=STYLE["IDT"]["color"], linewidth=2.5,
                                       label="IDT (ours)")]
    for s in SCENARIOS:
        legend_elements.append(
            plt.scatter([], [], marker=scenario_markers[s], s=40,
                        color="gray", label=f"Scenario {SCENARIO_LABEL[s]}"))
    fig.legend(handles=legend_elements, loc="upper center", ncol=6,
               frameon=True, framealpha=0.9, edgecolor="#ccc",
               bbox_to_anchor=(0.5, 1.03), columnspacing=1.2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        p = out_dir / f"demo_plan_c_box.{ext}"
        fig.savefig(p, format=ext, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plan C] saved to {out_dir / 'demo_plan_c_box.png'}")


# ── Plan A-v2-rank: line chart + avg rank bar ─────────────────────────────

def plan_a_v2_rank(data: dict, out_dir: Path):
    """Left: difficulty-sorted line chart.  Right: cross-scenario avg rank bar."""
    plt.rcParams.update(RC)

    mk = "nhp_viol"

    method_scenario_mean: dict[str, dict[int, float]] = {}
    method_scenario_std: dict[str, dict[int, float]] = {}
    for name in METHOD_ORDER:
        method_scenario_mean[name] = {}
        method_scenario_std[name] = {}
        for s in SCENARIOS:
            m, sd = mean_std(data[name][s].get(mk, []))
            method_scenario_mean[name][s] = m
            method_scenario_std[name][s] = sd

    sorted_scenarios = SCENARIOS  # keep original A–E order
    x_labels = [SCENARIO_LABEL[s] for s in sorted_scenarios]

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.2], wspace=0.30)
    ax_main = fig.add_subplot(gs[0])
    ax_rank = fig.add_subplot(gs[1])

    # ── Left: Line chart ──
    for name in METHOD_ORDER:
        st = STYLE[name]
        ys = [method_scenario_mean[name][s] for s in sorted_scenarios]
        errs = [method_scenario_std[name][s] for s in sorted_scenarios]
        ys_arr = np.array(ys)
        errs_arr = np.array(errs)
        is_weak = name in WEAK

        kw = dict(color=st["color"], marker=st["marker"],
                  linewidth=st["lw"], markersize=st["ms"],
                  zorder=st["zorder"], label=name,
                  linestyle=st.get("ls", "-"),
                  alpha=0.35 if is_weak else 1.0)
        ax_main.plot(range(len(sorted_scenarios)), ys_arr, **kw)

        has_err = np.any(np.isfinite(errs_arr) & (errs_arr > 0))
        if has_err and not is_weak:
            ax_main.fill_between(range(len(sorted_scenarios)),
                                 ys_arr - errs_arr, ys_arr + errs_arr,
                                 color=st["color"], alpha=0.10, zorder=1)

    ax_main.set_xticks(range(len(sorted_scenarios)))
    ax_main.set_xticklabels(x_labels)
    ax_main.set_xlabel("Scenario")
    ax_main.set_ylabel("NHP Violations")
    ax_main.set_title("NHP Violations across OOD Scenarios")
    ax_main.grid(axis="y", alpha=0.25, zorder=0)
    ax_main.grid(axis="x", alpha=0.12, linestyle=":", zorder=0)
    ax_main.set_ylim(bottom=-0.008, top=0.42)

    ax_main.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92,
                   edgecolor="#ccc", fontsize=9, columnspacing=0.8)

    # ── Right: Cross-scenario mean ± std (actual metric values) ──
    cross_mean: dict[str, float] = {}
    cross_std: dict[str, float] = {}
    for n in METHOD_ORDER:
        vals = [method_scenario_mean[n][s] for s in SCENARIOS
                if np.isfinite(method_scenario_mean[n][s])]
        cross_mean[n] = np.mean(vals) if vals else float("nan")
        cross_std[n] = np.std(vals, ddof=0) if vals else float("nan")

    value_sorted = sorted(METHOD_ORDER, key=lambda n: cross_mean[n])
    y_pos = np.arange(len(value_sorted))

    bars = ax_rank.barh(
        y_pos, [cross_mean[n] for n in value_sorted],
        xerr=[cross_std[n] for n in value_sorted],
        color=[STYLE[n]["color"] for n in value_sorted],
        edgecolor="white", linewidth=0.5, height=0.6, zorder=3,
        error_kw=dict(ecolor="black", elinewidth=0.9, capsize=3, capthick=0.8),
    )
    for bar, n in zip(bars, value_sorted):
        if n in WEAK:
            bar.set_alpha(0.35)
        if n == "IDT":
            bar.set_edgecolor(STYLE[n]["color"])
            bar.set_linewidth(2.0)

    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(
        [n.replace("PPO-Lag-", "Lag-") for n in value_sorted], fontsize=10)
    ax_rank.set_xlabel("Mean NHP Violations\n(across scenarios, lower is better)", fontsize=10)
    ax_rank.set_title("Cross-scenario Average", fontsize=12)
    ax_rank.grid(axis="x", alpha=0.25, zorder=0)

    for i, n in enumerate(value_sorted):
        val = cross_mean[n]
        sd = cross_std[n]
        x_text = val + sd + 0.003 if np.isfinite(sd) else val + 0.003
        ax_rank.text(
            x_text, i, f"{val:.3f}",
            va="center", fontsize=9, color=STYLE[n]["color"],
            fontweight="bold" if n == "IDT" else "normal",
            alpha=0.40 if n in WEAK else 1.0,
        )

    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        p = out_dir / f"demo_plan_a_v2_line_rank.{ext}"
        fig.savefig(p, format=ext, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plan A-v2-rank] saved to {out_dir / 'demo_plan_a_v2_line_rank.png'}")

    print(f"  Scenario order: {' '.join(x_labels)}")
    for n in value_sorted:
        print(f"    {n:22s}: mean={cross_mean[n]:.4f} ± {cross_std[n]:.4f}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    data = load_all()
    out_dir = REPO / "outputs/figures/channel_generality"

    for name in METHOD_ORDER:
        found = sum(1 for s in SCENARIOS if data[name][s].get("nhp_viol"))
        seeds_example = len(data[name][5].get("nhp_viol", []))
        print(f"  {name:22s}: {found}/5 scenarios, {seeds_example} seeds in s5")

    plan_a_v2(data, out_dir)
    plan_a_v2_rank(data, out_dir)
    plan_c(data, out_dir)


if __name__ == "__main__":
    main()
