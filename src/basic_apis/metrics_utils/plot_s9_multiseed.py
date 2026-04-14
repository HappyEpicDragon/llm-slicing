"""
多 seed 对比绘图：从 per-seed JSON 读取数据，绘制均值 ± 标准差曲线。

用法（在项目根目录）：
    pixi run python src/basic_apis/metrics_utils/plot_s9_multiseed.py \
        --scenario 9 --seeds 0 1 2 3 4 \
        --methods ppo_ha_weighted dt_baseline dt \
        --outdir outputs/plots_s9
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.basic_apis.metrics_utils.plot_style import get_style, setup_style

# ─── 注意：指标计算方式 ─────────────────────────────────────────────────────────
# 所有测试脚本已对齐 metric_value.py 的计算标准（2026-03-13）：
#   Distance: 每个 slice 仅计入违约（负值）drift 的最差指标（min），按 active_slices×steps 归一化
#   Violation: 每个 slice 整体判断（min(drifts) < 0 → 1次），按 active_slices×steps 归一化
# ⚠️  已有旧数据需重新运行对应 test 命令才能更新为新计算方式的结果。

BASE_DIR = "data/channel_generality"


def _save_figure_both(fig, png_path: str, dpi: int = 180):
    """同时保存 PNG 和 PDF。"""
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def load_metric_per_seed(method: str, scenario: int, seeds: list, metric_file: str, value_key: str):
    """
    读取 data/<method>/metric_json/scenario_<N>/seed_<S>/<metric_file>.json
    返回 shape (n_seeds, n_episodes) 的 numpy 数组。
    """
    arrays = []
    for seed in seeds:
        path = os.path.join(
            BASE_DIR, method, "metric_json",
            f"scenario_{scenario}", f"seed_{seed}", metric_file
        )
        if not os.path.exists(path):
            print(f"  [WARN] missing: {path}")
            return None
        with open(path) as f:
            data = json.load(f)
        arrays.append(np.array(data[value_key], dtype=np.float64))
    return np.array(arrays)   # (n_seeds, n_episodes)


def plot_metric(ax, methods, scenario, seeds, metric_file, value_key, title, ylabel, yref=None):
    for method in methods:
        arr = load_metric_per_seed(method, scenario, seeds, metric_file, value_key)
        if arr is None:
            continue
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        x    = np.arange(len(mean))
        sty   = get_style(method)
        color = sty.get("color")
        label = sty.get("label", method)
        ls    = sty.get("ls", "-")
        marker = sty.get("marker", "o") or "o"

        ax.plot(x, mean, label=label, color=color, linewidth=2,
                linestyle=ls, marker=marker,
                markevery=max(1, len(x) // 8), markersize=5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)

    if yref is not None:
        ax.axhline(yref, color="gray", linestyle=":", linewidth=1.2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, frameon=False)


def make_comparison_plot(methods, scenario, seeds, outdir):
    setup_style()
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Scenario {scenario} — Multi-seed Comparison  (seeds={seeds})",
                 fontsize=14, fontweight="bold", y=0.98)

    plot_metric(axes[0, 0], methods, scenario, seeds,
                "hp_violations.json", "hp_violations",
                "HP Violation Rate (↓ better)", "HP Viol. Rate", yref=0.0)

    plot_metric(axes[0, 1], methods, scenario, seeds,
                "nhp_violations.json", "nhp_violations",
                "NHP Violation Rate (↓ better)", "NHP Viol. Rate", yref=0.0)

    plot_metric(axes[1, 0], methods, scenario, seeds,
                "hp_distance.json", "hp_distance",
                "HP Distance (drift, ↑ better)", "HP Dist (mean drift)", yref=0.0)

    plot_metric(axes[1, 1], methods, scenario, seeds,
                "nhp_distance.json", "nhp_distance",
                "NHP Distance (drift, ↑ better)", "NHP Dist (mean drift)", yref=0.0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(outdir, f"scenario_{scenario}_multiseed.png")
    _save_figure_both(fig, out_path, dpi=180)
    plt.close(fig)

    # ── 也单独保存 reward 图 ───────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.suptitle(f"Scenario {scenario} — Episode Reward  (seeds={seeds})",
                  fontsize=13, fontweight="bold")
    plot_metric(ax2, methods, scenario, seeds,
                "episode_rewards.json", "rewards",
                "Episode Reward (↑ better)", "Reward")
    plt.tight_layout()
    out_path2 = os.path.join(outdir, f"scenario_{scenario}_rewards.png")
    _save_figure_both(fig2, out_path2, dpi=180)
    plt.close(fig2)


# ─── Step-level 时序图（Fig.4 / Fig.5）：从 NPZ 读 step_hp_dist / step_nhp_dist ───
ROLLING_WINDOW = 500


def _load_step_level_npz(method: str, scenario: int, seeds: list, field: str):
    """从 metric_raw/scenario_<N>/ep_seed<S>.npz 加载 step 级数组，返回 list of arrays (每 seed 一个)."""
    out = []
    for seed in seeds:
        path = os.path.join(BASE_DIR, method, "metric_raw", f"scenario_{scenario}", f"ep_seed{seed}.npz")
        if not os.path.exists(path):
            continue
        try:
            npz = np.load(path, allow_pickle=True)
            if field not in npz:
                continue
            arr = npz[field].ravel()
            if "viol" in field and arr.dtype != np.float64:
                arr = (arr < 0).astype(np.float64)  # 从 distance 派生 violation
            out.append(arr)
        except Exception as e:
            print(f"  [WARN] {path}: {e}")
    return out


def _smooth_and_band(arrays, window=ROLLING_WINDOW):
    """多 seed 数组：各 seed 先 rolling 平滑，再求 mean ± std。"""
    if not arrays:
        return None, None, None
    min_len = min(len(a) for a in arrays)
    smoothed = np.array([
        pd.Series(np.asarray(a)[:min_len]).rolling(window=window, min_periods=1).mean().values
        for a in arrays
    ])
    mean_curve = smoothed.mean(axis=0)
    std_curve = smoothed.std(axis=0)
    x = np.arange(min_len)
    return x, mean_curve, std_curve


def make_step_level_plots(methods, scenario, seeds, outdir):
    """
    绘制 Step-level 时序图（对应 experiment_manual_final 中 Fig.4 / Fig.5）。
    - 数据来源：metric_raw/scenario_<N>/ep_seed<S>.npz 的 step_hp_dist / step_nhp_dist
    - Violation 由 distance<0 派生（瞬时违约指示）
    - 5 seeds 平滑后均值 ± std，rolling window=500
    """
    setup_style()
    os.makedirs(outdir, exist_ok=True)

    # Fig.4 — Instantaneous Distance
    fig4, (ax4_hp, ax4_nhp) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig4.suptitle(f"Scenario {scenario} — Instantaneous Distance (Step-level, seeds={seeds})", fontsize=13, fontweight="bold")
    plt.subplots_adjust(hspace=0.08)

    for method in methods:
        hp_arrays = _load_step_level_npz(method, scenario, seeds, "step_hp_dist")
        nhp_arrays = _load_step_level_npz(method, scenario, seeds, "step_nhp_dist")
        if not hp_arrays and not nhp_arrays:
            continue
        sty = get_style(method)
        x_hp, mean_hp, std_hp = _smooth_and_band(hp_arrays) if hp_arrays else (None, None, None)
        x_nhp, mean_nhp, std_nhp = _smooth_and_band(nhp_arrays) if nhp_arrays else (None, None, None)
        if x_hp is not None:
            ax4_hp.fill_between(x_hp, mean_hp - std_hp, mean_hp + std_hp, color=sty["color"], alpha=0.18)
            ax4_hp.plot(x_hp, mean_hp, color=sty["color"], ls=sty.get("ls", "-"), lw=2, label=sty.get("label", method))
        if x_nhp is not None:
            ax4_nhp.fill_between(x_nhp, mean_nhp - std_nhp, mean_nhp + std_nhp, color=sty["color"], alpha=0.18)
            ax4_nhp.plot(x_nhp, mean_nhp, color=sty["color"], ls=sty.get("ls", "-"), lw=2, label=sty.get("label", method))

    ax4_hp.set_ylabel("HP Distance", fontsize=11)
    ax4_hp.axhline(0, color="gray", ls=":", lw=1)
    ax4_hp.grid(True, alpha=0.4)
    ax4_hp.legend(fontsize=9, frameon=False)
    ax4_nhp.set_ylabel("NHP Distance", fontsize=11)
    ax4_nhp.set_xlabel("Simulation Steps", fontsize=11)
    ax4_nhp.axhline(0, color="gray", ls=":", lw=1)
    ax4_nhp.grid(True, alpha=0.4)
    ax4_nhp.legend(fontsize=9, frameon=False)
    out4 = os.path.join(outdir, f"scenario_{scenario}_step_distance.png")
    _save_figure_both(fig4, out4, dpi=180)
    plt.close(fig4)

    # Fig.5 — Instantaneous Violation（由 step distance < 0 派生）
    fig5, (ax5_hp, ax5_nhp) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig5.suptitle(f"Scenario {scenario} — Instantaneous Violation (Step-level, seeds={seeds})", fontsize=13, fontweight="bold")
    plt.subplots_adjust(hspace=0.08)

    for method in methods:
        hp_arrays = _load_step_level_npz(method, scenario, seeds, "step_hp_dist")
        nhp_arrays = _load_step_level_npz(method, scenario, seeds, "step_nhp_dist")
        if not hp_arrays and not nhp_arrays:
            continue
        hp_viol = [(a < 0).astype(np.float64) for a in hp_arrays]
        nhp_viol = [(a < 0).astype(np.float64) for a in nhp_arrays]
        sty = get_style(method)
        x_hp, mean_hp, std_hp = _smooth_and_band(hp_viol)
        x_nhp, mean_nhp, std_nhp = _smooth_and_band(nhp_viol)
        if x_hp is not None:
            ax5_hp.fill_between(x_hp, mean_hp - std_hp, mean_hp + std_hp, color=sty["color"], alpha=0.18)
            ax5_hp.plot(x_hp, mean_hp, color=sty["color"], ls=sty.get("ls", "-"), lw=2, label=sty.get("label", method))
        if x_nhp is not None:
            ax5_nhp.fill_between(x_nhp, mean_nhp - std_nhp, mean_nhp + std_nhp, color=sty["color"], alpha=0.18)
            ax5_nhp.plot(x_nhp, mean_nhp, color=sty["color"], ls=sty.get("ls", "-"), lw=2, label=sty.get("label", method))

    ax5_hp.set_ylabel("HP Violation Ratio", fontsize=11)
    ax5_hp.set_ylim(-0.05, 1.05)
    ax5_hp.grid(True, alpha=0.4)
    ax5_hp.legend(fontsize=9, frameon=False)
    ax5_nhp.set_ylabel("NHP Violation Ratio", fontsize=11)
    ax5_nhp.set_xlabel("Simulation Steps", fontsize=11)
    ax5_nhp.set_ylim(-0.05, 1.05)
    ax5_nhp.grid(True, alpha=0.4)
    ax5_nhp.legend(fontsize=9, frameon=False)
    out5 = os.path.join(outdir, f"scenario_{scenario}_step_violation.png")
    _save_figure_both(fig5, out5, dpi=180)
    plt.close(fig5)


def plot_multiseed_from_cfg(cfg):
    """Hydra cfg 入口，供 channel_generality.py 调用。"""
    plot_cfg = cfg.get("plot_multiseed", cfg)
    # 兼容 scenario_ids（手册命名）与 scenario（脚本命名）
    scenario_ids = plot_cfg.get("scenario_ids", None)
    if scenario_ids:
        scenario = int(list(scenario_ids)[0])
    else:
        scenario = int(plot_cfg.get("scenario", 9))
    seeds    = list(plot_cfg.get("seeds",   [0, 1, 2, 3, 4]))
    # 兼容 model_keys（手册命名）与 methods（脚本命名）
    methods  = list(plot_cfg.get("methods", plot_cfg.get("model_keys", ["ppo_ha_weighted", "dt_baseline", "dt"])))
    # 兼容 output_dir（手册命名）与 outdir（脚本命名）
    outdir   = str(plot_cfg.get("outdir", plot_cfg.get("output_dir", "outputs/plots_s9")))
    print(f"Plotting scenario={scenario}, seeds={seeds}, methods={methods}, outdir={outdir}")
    make_comparison_plot(methods, scenario, seeds, outdir)


def plot_step_multiseed_from_cfg(cfg):
    """Step-level 时序图（Fig.4/5）入口，与 plot_multiseed 共用 scenario/seeds/methods/outdir 配置。"""
    plot_cfg = cfg.get("plot_step_multiseed", cfg.get("plot_multiseed", cfg))
    scenario_ids = plot_cfg.get("scenario_ids", None)
    if scenario_ids:
        scenario = int(list(scenario_ids)[0])
    else:
        scenario = int(plot_cfg.get("scenario", 9))
    seeds    = list(plot_cfg.get("seeds",   [0, 1, 2, 3, 4]))
    methods  = list(plot_cfg.get("methods", plot_cfg.get("model_keys", ["ppo_ha_weighted", "dt_baseline", "dt"])))
    outdir   = str(plot_cfg.get("outdir", plot_cfg.get("output_dir", "outputs/plots_s9")))
    print(f"Step-level plots: scenario={scenario}, seeds={seeds}, methods={methods}, outdir={outdir}")
    make_step_level_plots(methods, scenario, seeds, outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=9)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+",
                        default=["ppo_ha_weighted", "dt_baseline", "dt"])
    parser.add_argument("--outdir", default="outputs/plots_s9")
    args = parser.parse_args()

    print(f"Plotting scenario={args.scenario}, seeds={args.seeds}, methods={args.methods}")
    make_comparison_plot(args.methods, args.scenario, args.seeds, args.outdir)


if __name__ == "__main__":
    main()
