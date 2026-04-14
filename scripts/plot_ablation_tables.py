"""
Phase 4 S4/S5 敏感性分析：encoder 消融 与 dataset mixing 消融

输入：data/channel_generality/dt_v2_8exp/eval_ood/E*/scenario_*/summary.json
输出：
  outputs/figures/channel_generality/sensitivity/ablation_encoder.{png,pdf}
  outputs/figures/channel_generality/sensitivity/ablation_dataset.{png,pdf}
  outputs/tables/channel_generality/sensitivity/ablation_table.csv

实验矩阵（10 组，20ep per scenario，s5-s9）：
  E1:  D-A + MLP          E2:  D-A + SliceAttn
  E3:  D-B + MLP          E4:  D-B + SliceAttn  ← 主方法
  E5:  D-C + MLP          E6:  D-C + SliceAttn
  E7:  D-D + MLP          E8:  D-D + SliceAttn
  E9:  D-E + MLP (epoch5) E10: D-E + SliceAttn (epoch5)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

BASE_DIR = "data/channel_generality/dt_v2_8exp/eval_ood"
OUT_FIG = "outputs/figures/channel_generality/sensitivity"
OUT_TAB = "outputs/tables/channel_generality/sensitivity"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

SCENARIOS = [5, 6, 7, 8, 9]

EXPS = {
    "E1":  ("D-A", "MLP"),
    "E2":  ("D-A", "SliceAttn"),
    "E3":  ("D-B", "MLP"),
    "E4":  ("D-B", "SliceAttn"),
    "E5":  ("D-C", "MLP"),
    "E6":  ("D-C", "SliceAttn"),
    "E7":  ("D-D", "MLP"),
    "E8":  ("D-D", "SliceAttn"),
    "E9":  ("D-E", "MLP",),
    "E10": ("D-E", "SliceAttn"),
}

DATASET_ORDER = ["D-A", "D-B", "D-C", "D-D", "D-E"]
DATASET_LABELS = {
    "D-A": "D-A\n(single expert)",
    "D-B": "D-B\n(8-expert mix)",
    "D-C": "D-C",
    "D-D": "D-D",
    "D-E": "D-E",
}
ENCODER_COLORS = {"MLP": "#4C72B0", "SliceAttn": "#DD8452"}
ENCODER_HATCHES = {"MLP": "//", "SliceAttn": ""}

METRICS = [
    ("nhp_viol_mean", "NHP Violations (↓)"),
    ("nhp_dist_mean", "NHP Distance (↓, closer to 0)"),
    ("hp_viol_mean", "HP Violations (↓)"),
    ("hp_dist_mean", "HP Distance (↓, closer to 0)"),
]


def load_exp_results(exp_id):
    """加载一组实验在 s5-s9 上的平均指标。"""
    results = {k: [] for k, _ in METRICS}
    for s in SCENARIOS:
        path = os.path.join(BASE_DIR, exp_id, f"scenario_{s}", "summary.json")
        if not os.path.exists(path):
            print(f"  [WARN] missing: {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        for key, _ in METRICS:
            results[key].append(d.get(key, np.nan))
    return {k: np.nanmean(v) if v else np.nan for k, v in results.items()}


def collect_all():
    """收集所有实验结果。"""
    data = {}
    for exp_id, info in EXPS.items():
        dataset, encoder = info[0], info[1]
        r = load_exp_results(exp_id)
        r["dataset"] = dataset
        r["encoder"] = encoder
        r["exp_id"] = exp_id
        data[exp_id] = r
        print(f"  {exp_id} ({dataset}/{encoder}): nhp_viol={r['nhp_viol_mean']:.5f}  nhp_dist={r['nhp_dist_mean']:.6f}")
    return data


def plot_encoder_ablation(data):
    """S4: MLP vs SliceAttn（固定 D-B 数据集）"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("S4: Encoder Ablation (D-B Dataset, s5–s9)", fontweight="bold")

    db_exps = {k: v for k, v in data.items() if v["dataset"] == "D-B"}
    encoders = ["MLP", "SliceAttn"]

    for ax, (metric_key, metric_label) in zip(axes, [
        ("nhp_viol_mean", "NHP Violations (↓)"),
        ("nhp_dist_mean", "NHP Distance (↓)"),
    ]):
        vals = [db_exps.get(e, {}).get(metric_key, np.nan)
                for e in ["E3", "E4"]]
        colors = [ENCODER_COLORS[enc] for enc in encoders]
        bars = ax.bar(encoders, vals, color=colors, width=0.4, edgecolor="k", linewidth=0.8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(bar.get_height()) * 0.02,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xlabel("Encoder")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(OUT_FIG, f"ablation_encoder.{ext}"), dpi=150, bbox_inches="tight")
    print(f"[S4] Saved ablation_encoder.{{png,pdf}}")
    plt.close()


def plot_dataset_ablation(data):
    """S5: 各数据集 × encoder 的组合比较"""
    metric_pairs = [
        ("nhp_viol_mean", "NHP Violations (↓)"),
        ("nhp_dist_mean", "NHP Distance (↓)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("S5: Dataset Mixing Ablation (s5–s9 OOD)", fontweight="bold")

    x = np.arange(len(DATASET_ORDER))
    width = 0.35

    for ax, (metric_key, metric_label) in zip(axes, metric_pairs):
        for i, encoder in enumerate(["MLP", "SliceAttn"]):
            vals = []
            for ds in DATASET_ORDER:
                found = [v for v in data.values() if v["dataset"] == ds and v["encoder"] == encoder]
                vals.append(found[0][metric_key] if found else np.nan)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=encoder,
                          color=ENCODER_COLORS[encoder],
                          hatch=ENCODER_HATCHES[encoder],
                          edgecolor="k", linewidth=0.7)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ypos = bar.get_height() + abs(bar.get_height()) * 0.02
                    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in DATASET_ORDER], fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()
        ax.set_xlabel("Dataset")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(OUT_FIG, f"ablation_dataset.{ext}"), dpi=150, bbox_inches="tight")
    print(f"[S5] Saved ablation_dataset.{{png,pdf}}")
    plt.close()


def save_csv(data):
    """保存完整消融表到 CSV。"""
    path = os.path.join(OUT_TAB, "ablation_table.csv")
    fieldnames = ["exp_id", "dataset", "encoder"] + [k for k, _ in METRICS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp_id in sorted(data.keys(), key=lambda x: int(x[1:])):
            row = {k: data[exp_id].get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"[CSV] Saved {path}")


if __name__ == "__main__":
    print("=== 收集实验数据 ===")
    all_data = collect_all()

    print("\n=== S4: Encoder 消融图 ===")
    plot_encoder_ablation(all_data)

    print("\n=== S5: Dataset Mixing 消融图 ===")
    plot_dataset_ablation(all_data)

    print("\n=== 保存 CSV 表格 ===")
    save_csv(all_data)

    print("\n=== 完成 ===")
    print(f"图表输出至: {OUT_FIG}/")
    print(f"CSV 输出至:  {OUT_TAB}/ablation_table.csv")
