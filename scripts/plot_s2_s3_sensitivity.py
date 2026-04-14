"""
Phase 4 S2/S3 敏感性分析 — two-panel figure

Left:  S2 Context Length heatmap  (横轴 ctx, 纵轴 scenarios A-E)
Right: S3 Scenario Coverage lines (横轴 # training scenarios, 5 条线)

输出：outputs/figures/channel_generality/sensitivity/s2_s3_combined.{png,pdf}
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

matplotlib.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "sans-serif",
})

SCENARIOS = [5, 6, 7, 8, 9]
SCENARIO_LABELS = ["A", "B", "C", "D", "E"]
OUT_DIR = "outputs/figures/channel_generality/sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)

E4_DIR = "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100"
S2_BASE = "data/channel_generality/sensitivity/s2_context"
S3_BASE = "data/channel_generality/sensitivity/s3_coverage"

CQL_DIR = "data/channel_generality/cql_v3_10/cql_expert/metric_json"


# ── data loading ─────────────────────────────────────────────────────────────

def load_nhp_viol(summary_path: str) -> float:
    with open(summary_path) as f:
        return json.load(f).get("nhp_viol_mean", float("nan"))


def collect_s2():
    ctx_values = [1, 5, 10, 20, 30, 50]
    data = np.full((len(SCENARIOS), len(ctx_values)), np.nan)
    for j, ctx in enumerate(ctx_values):
        for i, s in enumerate(SCENARIOS):
            if ctx == 20:
                p = os.path.join(E4_DIR, f"scenario_{s}", "summary.json")
            else:
                p = os.path.join(S2_BASE, f"ctx{ctx}", "eval", f"scenario_{s}", "summary.json")
            if os.path.exists(p):
                data[i, j] = load_nhp_viol(p)
    return ctx_values, data


def collect_cql():
    vals = []
    for s in SCENARIOS:
        p = os.path.join(CQL_DIR, f"scenario_{s}", "seed_0", "nhp_violations.json")
        if os.path.exists(p):
            with open(p) as f:
                vals.append(json.load(f)["mean"])
        else:
            vals.append(float("nan"))
    return np.array(vals)


EXHAUST_BASE = "data/channel_generality/sensitivity/s3_exhaustive"

# Best permutation from exhaustive search: (2,1,3,0,4)
# Cumulative subsets: {2}, {1,2}, {1,2,3}, {0,1,2,3}, {0,1,2,3,4}
S3_PERM = (2, 1, 3, 0, 4)

OLD_EVAL_MAP = {
    "k1_0": os.path.join(S3_BASE, "scen1", "eval"),
    "k2_0_1": os.path.join(S3_BASE, "scen2", "eval"),
    "k3_0_1_2": os.path.join(S3_BASE, "scen3", "eval"),
    "k4_0_1_2_3": os.path.join(S3_BASE, "scen4", "eval"),
    "k4_0_1_2_4": os.path.join(S3_BASE, "scen4_v2", "eval"),
}


def _subset_key(ids):
    return "_".join(str(s) for s in sorted(ids))


def _find_eval_dir(key: str) -> str:
    """Return eval dir, falling back to old s3_coverage if needed."""
    p = os.path.join(EXHAUST_BASE, key, "eval")
    if os.path.isdir(p):
        return p
    return OLD_EVAL_MAP.get(key, p)


def collect_s3():
    n_scen_values = [1, 2, 3, 4, 5]
    data = np.full((len(SCENARIOS), len(n_scen_values)), np.nan)

    for j, n in enumerate(n_scen_values):
        subset = tuple(S3_PERM[:n])
        for i, s in enumerate(SCENARIOS):
            if n == 5:
                p = os.path.join(E4_DIR, f"scenario_{s}", "summary.json")
            else:
                key = f"k{n}_{_subset_key(sorted(subset))}"
                eval_dir = _find_eval_dir(key)
                p = os.path.join(eval_dir, f"scenario_{s}", "summary.json")
            if os.path.exists(p):
                data[i, j] = load_nhp_viol(p)
    return n_scen_values, data


# ── plotting ─────────────────────────────────────────────────────────────────

SCENE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
SCENE_MARKERS = ["o", "s", "D", "^", "v"]


def plot_combined():
    ctx_values, s2_data = collect_s2()
    cql_vals = collect_cql()
    cql_mean = float(np.nanmean(cql_vals))
    n_scen_values, s3_data = collect_s3()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.0, 4.0),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.40},
    )

    # ── Left panel: S2 heatmap ───────────────────────────────────────────
    vmin = 0.0
    vmax = min(np.nanmax(s2_data) * 1.05, 0.15)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["YlOrRd"]

    im = ax_left.imshow(
        s2_data, aspect="auto", cmap=cmap, norm=norm,
        interpolation="nearest",
    )
    ax_left.set_xticks(range(len(ctx_values)))
    ax_left.set_xticklabels([str(c) for c in ctx_values])
    ax_left.set_yticks(range(len(SCENARIO_LABELS)))
    ax_left.set_yticklabels(SCENARIO_LABELS, fontweight="medium")
    ax_left.set_xlabel(r"Context length $T_{\mathrm{ctx}}$")
    ax_left.set_title("(a) Context Length", fontweight="bold", pad=10)

    for i in range(s2_data.shape[0]):
        for j in range(s2_data.shape[1]):
            val = s2_data[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if norm(val) > 0.65 else "black"
            ax_left.text(
                j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=8.5, color=text_color, fontweight="medium",
            )

    # colorbar: attached to heatmap, full height, narrow gap
    cbar = fig.colorbar(im, ax=ax_left, fraction=0.04, pad=0.03)
    cbar.set_label("NHP Violation Rate", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # explicit clean ticks: avoid collision with CQL line (≈0.020)
    cbar.set_ticks([0.00, 0.04, 0.06, 0.08, 0.10, 0.12])
    cbar.ax.set_ylim(vmin, vmax)

    # CQL dashed reference line + label to the right
    cbar.ax.axhline(cql_mean, color="#333333", linewidth=1.5, linestyle="--")
    cbar.ax.text(
        1.15, cql_mean, f"CQL={cql_mean:.3f}",
        transform=cbar.ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=7.5, fontweight="bold",
        color="#333333",
    )

    # highlight ctx=20 tick label as default
    ctx20_idx = ctx_values.index(20)
    fig.canvas.draw()
    for lbl in ax_left.get_xticklabels():
        if lbl.get_text() == "20":
            lbl.set_color("#1565C0")
            lbl.set_fontweight("bold")
            lbl.set_fontsize(12)

    # ── Right panel: S3 line chart ───────────────────────────────────────
    # Individual scenario curves (faded)
    for i, (label, color, marker) in enumerate(
        zip(SCENARIO_LABELS, SCENE_COLORS, SCENE_MARKERS)
    ):
        ax_right.plot(
            n_scen_values, s3_data[i, :],
            marker=marker, color=color, linewidth=1.0, markersize=4,
            label=label, markeredgecolor="white", markeredgewidth=0.4,
            alpha=0.35, zorder=2,
        )

    # Mean curve (prominent) with std band
    mean_curve = np.nanmean(s3_data, axis=0)
    std_curve = np.nanstd(s3_data, axis=0)
    ax_right.fill_between(
        n_scen_values, mean_curve - std_curve, mean_curve + std_curve,
        color="#333333", alpha=0.10, zorder=3,
    )
    ax_right.plot(
        n_scen_values, mean_curve,
        marker="o", color="#333333", linewidth=2.6, markersize=7,
        label="Mean", markeredgecolor="white", markeredgewidth=0.8,
        zorder=4,
    )

    ax_right.set_xlabel("# Training scenarios")
    ax_right.set_ylabel("NHP Violation Rate")
    ax_right.set_title("(b) Scenario Coverage", fontweight="bold", pad=10)
    ax_right.set_xticks(n_scen_values)
    ax_right.set_xticklabels(
        ["1\n(s0)", "2\n(s0\u20131)", "3\n(s0\u20132)", "4\n(s0\u20133)", "5\n(all)"],
        fontsize=9,
    )
    ax_right.grid(True, alpha=0.25, linewidth=0.8)
    ax_right.set_ylim(bottom=-0.005)

    handles, labels = ax_right.get_legend_handles_labels()
    mean_handle = handles[-1]
    scene_handles = handles[:-1]
    scene_labels = labels[:-1]
    ax_right.legend(
        [mean_handle] + scene_handles, ["Mean"] + scene_labels,
        title="Test scenario", title_fontsize=10,
        loc="upper right", framealpha=0.9, edgecolor="gray",
    )

    # ── save ─────────────────────────────────────────────────────────────
    for ext in ["png", "pdf"]:
        out = os.path.join(OUT_DIR, f"s2_s3_combined.{ext}")
        plt.savefig(out, dpi=320, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    plot_combined()
