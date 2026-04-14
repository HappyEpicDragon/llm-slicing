"""
Phase 4 S1: target_rtg 敏感性分析出图

输入：data/channel_generality/dt_v2_tiny/sensitivity/rtg_*/scenario_*/summary.json
输出：outputs/figures/channel_generality/sensitivity/rtg_sensitivity.{png,pdf}

RTG 范围：[-200, -100, -50, -20, -10, -5, 0, 10, 50]
  - [-200, 0]：训练分布内（log-transform 后 [-5.3, 0]）
  - [10, 50]：超出分布的正值外推点

x 轴：target_rtg（原始值，symlog 便于 0 附近可读）
y 轴：NHP 两项主指标（nhp_viol, -nhp_dist）跨场景 A-E 的均值
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

BASE_DIR = "data/channel_generality/dt_v2_tiny/sensitivity"
OUT_DIR = "outputs/figures/channel_generality/sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)

RTG_VALUES = [-200, -100, -50, -20, -10, -5, 0, 10, 50]
SCENARIOS = [5, 6, 7, 8, 9]
METRICS = [("nhp_viol_mean", "NHP Violations (↓)")]
DIST_KEY = "nhp_dist_mean"


def load_rtg_results(rtg):
    """加载某个 RTG 值在 s5-s9 的平均指标。返回 None 表示结果尚不存在。"""
    results = {k: [] for k, _ in METRICS}
    results[DIST_KEY] = []
    found = False
    for s in SCENARIOS:
        path = os.path.join(BASE_DIR, f"rtg_{rtg}", f"scenario_{s}", "summary.json")
        if not os.path.exists(path):
            continue
        found = True
        with open(path) as f:
            d = json.load(f)
        for key, _ in METRICS:
            results[key].append(d.get(key, np.nan))
        results[DIST_KEY].append(d.get(DIST_KEY, np.nan))
    if not found:
        return None
    return {k: np.nanmean(v) if v else np.nan for k, v in results.items()}


def collect_sweep():
    """收集所有可用 RTG 点的结果。"""
    xs, ys = [], {k: [] for k, _ in METRICS}
    ys[DIST_KEY] = []
    missing = []
    for rtg in RTG_VALUES:
        r = load_rtg_results(rtg)
        if r is None:
            missing.append(rtg)
            continue
        xs.append(rtg)
        for key, _ in METRICS:
            ys[key].append(r[key])
        ys[DIST_KEY].append(r[DIST_KEY])
    return xs, ys, missing


def plot_sensitivity(xs, ys, missing):
    if not xs:
        print("[WARN] 没有可用数据，请等待扫描完成后重新运行")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    fig.suptitle("Target RTG Sensitivity (IDT, Scenarios A-E)", fontweight="bold", fontsize=14, y=0.975)

    # 标注训练分布边界（rtg=0 是分布上界，负值在分布内）
    dist_boundary = 0

    ax1, ax2 = axes

    # Left: NHP violations
    ax1.plot(xs, ys["nhp_viol_mean"], "o-", color="#1f77b4", linewidth=2.0, markersize=6.5, zorder=3, label="IDT")
    ax1.axvline(dist_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xscale("symlog", linthresh=10)
    ax1.set_title("NHP Violations (↓)")
    ax1.set_xlabel("target_rtg")
    ax1.set_ylabel("NHP Violations")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(v) for v in xs], rotation=30, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(xs, ys[DIST_KEY], "o-", color="#ff7f0e", linewidth=2.0, markersize=6.5, zorder=3, label="IDT")
    ax2.axvline(dist_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xscale("symlog", linthresh=10)
    ax2.set_title("NHP Distance (↑)")
    ax2.set_xlabel("target_rtg")
    ax2.set_ylabel("NHP Distance")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(v) for v in xs], rotation=30, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    if missing:
        fig.text(0.5, 0.01, f"[待运行] RTG 点: {missing}", ha="center",
                 fontsize=10, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.985])
    for ext in ["png", "pdf"]:
        out_path = os.path.join(OUT_DIR, f"rtg_sensitivity.{ext}")
        plt.savefig(out_path, dpi=320, bbox_inches="tight")
        print(f"[S1] Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    print("=== 收集 RTG 敏感性数据 ===")
    xs, ys, missing = collect_sweep()
    print(f"  已有: {xs}")
    if missing:
        print(f"  尚缺: {missing}（扫描进行中，可先用已有点出图）")

    print("\n=== 生成敏感性曲线图 ===")
    plot_sensitivity(xs, ys, missing)

    print("\n=== 完成 ===")
