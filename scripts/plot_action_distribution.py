"""
X3: 动作分布分析 — codebook / ordering / intra 直方图

从 D-B 训练数据集加载所有专家轨迹，统计三类动作的分布：
  1. Codebook（切片间资源分配模式），共 11 种选项
  2. Ordering（各切片被赋予的优先级得分），每切片 5 个离散值
  3. Intra（切片内资源分配策略），每切片 3 种模式

同时绘制「按场景」分解的分布，展示跨场景策略差异。

用法：
    pixi run python scripts/plot_action_distribution.py
"""
import os
import sys
import pickle
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────── 配置 ──────────────────────────────
DATASET_DIR = "data/channel_generality/dt_v2_8exp/dataset_D-B/training"
OUTPUT_DIR  = "outputs/figures/channel_generality/xai"
ROLLOUT_RAW_DIR = "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2/metric_raw"

NUM_SLICES    = 5
CODEBOOK_SIZE = 11         # action[0], 0–10
ORDER_SIZE    = 5          # action[1..5], 每切片 0–4
INTRA_SIZE    = 3          # action[6..10], 每切片 0–2

# Codebook 模式标注（根据 build_dirichlet_inter_quota_codebook 的实际值）
CODEBOOK_LABELS = [
    "0\n(uniform)", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "10\n(extreme)"
]
ORDER_LABELS  = [f"rank {i}" for i in range(ORDER_SIZE)]
INTRA_LABELS  = ["conservative", "moderate", "aggressive"]

SCENARIO_COLORS = {0: "#2196F3", 1: "#4CAF50", 2: "#FF9800", 3: "#9C27B0", 4: "#F44336"}
# ───────────────────────────────────────────────────────────────


def get_scenario_color(scen: int) -> str:
    palette = [
        "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728",
        "#17becf", "#bcbd22", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    if scen in SCENARIO_COLORS:
        return SCENARIO_COLORS[scen]
    return palette[scen % len(palette)]


def parse_scenario_from_filename(fn):
    """从文件名解析场景编号（如 S0_single_... → 0）"""
    try:
        return int(os.path.basename(fn).split("_single_")[0].lstrip("S"))
    except Exception:
        return -1


def load_all_actions(dataset_dir):
    """
    加载所有 pkl 轨迹，返回：
      all_actions: (N, 11) numpy array  [全部时间步]
      scen_actions: dict {scenario: (n_steps, 11) array}
    """
    all_acts = []
    scen_acts = defaultdict(list)

    pkl_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} trajectory files")

    for fn in pkl_files:
        scen = parse_scenario_from_filename(fn)
        try:
            traj = pickle.load(open(fn, "rb"))
            acts = traj["actions"].astype(np.int32)   # (1000, 11)
            all_acts.append(acts)
            if scen >= 0:
                scen_acts[scen].append(acts)
        except Exception as e:
            print(f"  Warning: {fn}: {e}")

    all_actions = np.concatenate(all_acts, axis=0)
    scen_actions = {s: np.concatenate(v, axis=0) for s, v in scen_acts.items()}
    print(f"Total timesteps: {all_actions.shape[0]}")
    for s, a in sorted(scen_actions.items()):
        print(f"  Scenario {s}: {a.shape[0]} steps")
    return all_actions, scen_actions


def load_actions_from_metric_raw(metric_raw_dir, scenarios=None):
    """
    从 metric_raw/scenario_*/ep_seed*.npz 读取动作。
    期望每个 npz 含键 `actions`，shape=(n_steps, 11)。
    """
    if scenarios is None:
        scenarios = [5, 6, 7, 8, 9]

    all_acts = []
    scen_acts = defaultdict(list)
    n_files = 0

    for scen in scenarios:
        scen_dir = os.path.join(metric_raw_dir, f"scenario_{scen}")
        npz_files = sorted(glob.glob(os.path.join(scen_dir, "ep_seed*.npz")))
        if not npz_files:
            print(f"  Warning: no npz found in {scen_dir}")
            continue
        for fn in npz_files:
            with np.load(fn) as data:
                if "actions" not in data:
                    raise KeyError(f"`actions` not found in {fn}. Please rerun test_dt_v2 after action logging patch.")
                acts = data["actions"].astype(np.int32)
            all_acts.append(acts)
            scen_acts[scen].append(acts)
            n_files += 1

    if not all_acts:
        raise RuntimeError(f"No valid rollout actions loaded from {metric_raw_dir}")

    all_actions = np.concatenate(all_acts, axis=0)
    scen_actions = {s: np.concatenate(v, axis=0) for s, v in scen_acts.items()}
    print(f"Loaded rollout actions from {n_files} npz files")
    print(f"Total rollout timesteps: {all_actions.shape[0]}")
    for s, a in sorted(scen_actions.items()):
        print(f"  Scenario {s}: {a.shape[0]} steps")
    return all_actions, scen_actions


def compute_distributions(actions):
    """计算各动作维度的频率分布。"""
    n = len(actions)
    codebook = np.bincount(actions[:, 0], minlength=CODEBOOK_SIZE).astype(float) / n
    ordering = []
    for s in range(NUM_SLICES):
        cnt = np.bincount(actions[:, 1 + s], minlength=ORDER_SIZE).astype(float) / n
        ordering.append(cnt)
    intra = []
    for s in range(NUM_SLICES):
        cnt = np.bincount(actions[:, 1 + NUM_SLICES + s], minlength=INTRA_SIZE).astype(float) / n
        intra.append(cnt)
    return codebook, ordering, intra


# ── 绘图工具 ────────────────────────────────────────────────────
def bar_plot(ax, values, labels, color, title, xlabel, ylabel="Frequency", ylim=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, alpha=0.82, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    if ylim:
        ax.set_ylim(0, ylim)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 在柱顶标注数值
    for b in bars:
        h = b.get_height()
        if h > 0.005:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.003,
                    f"{h:.2%}", ha="center", va="bottom", fontsize=6.5)


def grouped_bar(ax, scen_values_dict, labels, title, xlabel, ylabel="Frequency",
                width=0.15, ylim=None):
    """按场景分组柱状图。"""
    n_groups = len(labels)
    n_scen   = len(scen_values_dict)
    x = np.arange(n_groups)
    offsets = (np.arange(n_scen) - (n_scen - 1) / 2) * width

    for idx, (scen, vals) in enumerate(sorted(scen_values_dict.items())):
        ax.bar(x + offsets[idx], vals, width=width,
               label=f"s{scen}", color=get_scenario_color(scen),
               alpha=0.82, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.legend(fontsize=7.5, ncol=n_scen, loc="upper right",
              framealpha=0.8, handlelength=1.2, columnspacing=0.6)


# ── 主绘图函数 ──────────────────────────────────────────────────
def plot_main_distributions(all_dist, scen_dist_map, output_dir, out_prefix, title_prefix):
    """
    Figure 1: 总体分布 3 行：
      Row 1: Codebook 分布（11列）
      Row 2: Ordering 分布（每切片一列，5子图）
      Row 3: Intra 分布（每切片一列，5子图）
    """
    all_cb, all_ord, all_intra = all_dist
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

    # --- Row 0: Codebook ---
    ax_cb = fig.add_subplot(gs[0, :])   # 占全行
    bar_plot(ax_cb, all_cb, CODEBOOK_LABELS,
             color="#5C9ED6", title="Codebook (Inter-Slice Resource Mode) Distribution",
             xlabel="Codebook Index")
    # 添加均匀分布参考线
    ax_cb.axhline(1.0 / CODEBOOK_SIZE, color="gray", linestyle="--",
                  linewidth=1.0, label="Uniform baseline")
    ax_cb.legend(fontsize=8, loc="upper right")

    # --- Row 1: Ordering per slice ---
    ord_ymax = max(v.max() for v in all_ord) * 1.25
    for s in range(NUM_SLICES):
        ax = fig.add_subplot(gs[1, s])
        bar_plot(ax, all_ord[s], ORDER_LABELS,
                 color=SCENARIO_COLORS.get(s, "#888"),
                 title=f"Ordering — Slice {s}",
                 xlabel="Priority Score",
                 ylim=ord_ymax)
        ax.axhline(1.0 / ORDER_SIZE, color="gray", linestyle="--", linewidth=0.8)

    # --- Row 2: Intra per slice ---
    intra_ymax = max(v.max() for v in all_intra) * 1.25
    for s in range(NUM_SLICES):
        ax = fig.add_subplot(gs[2, s])
        bar_plot(ax, all_intra[s], INTRA_LABELS,
                 color=SCENARIO_COLORS.get(s, "#888"),
                 title=f"Intra-Slice Mode — Slice {s}",
                 xlabel="Intra Mode",
                 ylim=intra_ymax)
        ax.axhline(1.0 / INTRA_SIZE, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle(
        f"{title_prefix}\n"
        "(IDT-v2 action space: codebook + ordering + intra)",
        fontsize=12, fontweight="bold", y=0.98
    )

    out_png = os.path.join(output_dir, f"{out_prefix}_action_distribution_main.png")
    out_pdf = os.path.join(output_dir, f"{out_prefix}_action_distribution_main.pdf")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    return out_png


def plot_per_scenario(all_dist, scen_dist_map, output_dir, out_prefix, title_prefix):
    """
    Figure 2: 按场景分组对比
      3行 × 3列：codebook（1行跨全幅） + ordering合并 + intra合并
    """
    all_cb, all_ord, all_intra = all_dist

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.5)

    # Row 0: Codebook per scenario
    scen_cb = {s: d[0] for s, d in scen_dist_map.items()}
    grouped_bar(axes[0], scen_cb, CODEBOOK_LABELS,
                title="Codebook Distribution per Scenario",
                xlabel="Codebook Index",
                width=0.14)
    axes[0].axhline(1.0 / CODEBOOK_SIZE, color="gray", linestyle="--",
                    linewidth=1.0, label="Uniform")

    # Row 1: Ordering（5 切片的 priority=最高rank比例 聚合）
    # 展示：各场景各切片被排名第1（score=4）的频率
    top_rank_freq = {}  # {scen: array of shape (5,)}
    for s, (_, ord_d, _) in scen_dist_map.items():
        top_rank_freq[s] = np.array([ord_d[sl][ORDER_SIZE - 1] for sl in range(NUM_SLICES)])

    slice_labels = [f"Slice {s}" for s in range(NUM_SLICES)]
    grouped_bar(axes[1], top_rank_freq, slice_labels,
                title="Ordering — Frequency of Being Assigned Highest Priority (rank=4) per Slice",
                xlabel="Slice Index",
                width=0.14)
    axes[1].axhline(1.0 / NUM_SLICES, color="gray", linestyle="--",
                    linewidth=1.0, label="Uniform (if random)")

    # Row 2: Intra（mode=2激进比例）
    aggressive_freq = {}  # {scen: array of shape (5,)}
    for s, (_, _, intra_d) in scen_dist_map.items():
        aggressive_freq[s] = np.array([intra_d[sl][INTRA_SIZE - 1] for sl in range(NUM_SLICES)])

    grouped_bar(axes[2], aggressive_freq, slice_labels,
                title='Intra Mode — Frequency of "Aggressive" Allocation (mode=2) per Slice',
                xlabel="Slice Index",
                width=0.14)

    fig.suptitle(
        f"{title_prefix} — Breakdown by Scenario",
        fontsize=12, fontweight="bold", y=1.01
    )

    out_png = os.path.join(output_dir, f"{out_prefix}_action_distribution_by_scenario.png")
    out_pdf = os.path.join(output_dir, f"{out_prefix}_action_distribution_by_scenario.pdf")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def plot_compact_summary(all_dist, output_dir, out_prefix, title_prefix):
    """
    Figure 3: 紧凑版 — 3×1 简洁对比（论文用）
    """
    all_cb, all_ord, all_intra = all_dist

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plt.subplots_adjust(wspace=0.35)

    # Codebook
    ax = axes[0]
    x = np.arange(CODEBOOK_SIZE)
    ax.bar(x, all_cb, color="#5C9ED6", alpha=0.85, edgecolor="white")
    ax.axhline(1.0 / CODEBOOK_SIZE, color="gray", linestyle="--", linewidth=1.0,
               label="Uniform")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(CODEBOOK_SIZE)], fontsize=8)
    ax.set_xlabel("Codebook Index", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("(a) Codebook Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ordering — 展示各切片分布的箱线图/热图
    ax = axes[1]
    ord_matrix = np.stack(all_ord)    # (5, 5)
    im = ax.imshow(ord_matrix, aspect="auto", cmap="Blues",
                   vmin=0, vmax=ord_matrix.max())
    ax.set_xticks(range(ORDER_SIZE))
    ax.set_xticklabels([f"score {i}" for i in range(ORDER_SIZE)], fontsize=8)
    ax.set_yticks(range(NUM_SLICES))
    ax.set_yticklabels([f"Slice {s}" for s in range(NUM_SLICES)], fontsize=8)
    ax.set_xlabel("Priority Score", fontsize=9)
    ax.set_title("(b) Ordering Distribution (Heatmap)", fontsize=10, fontweight="bold")
    for r in range(NUM_SLICES):
        for c in range(ORDER_SIZE):
            ax.text(c, r, f"{ord_matrix[r, c]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if ord_matrix[r, c] < 0.5 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Intra — 展示各切片分布热图
    ax = axes[2]
    intra_matrix = np.stack(all_intra)   # (5, 3)
    im2 = ax.imshow(intra_matrix, aspect="auto", cmap="Oranges",
                    vmin=0, vmax=intra_matrix.max())
    ax.set_xticks(range(INTRA_SIZE))
    ax.set_xticklabels(INTRA_LABELS, fontsize=8)
    ax.set_yticks(range(NUM_SLICES))
    ax.set_yticklabels([f"Slice {s}" for s in range(NUM_SLICES)], fontsize=8)
    ax.set_xlabel("Intra Mode", fontsize=9)
    ax.set_title("(c) Intra-Slice Mode Distribution", fontsize=10, fontweight="bold")
    for r in range(NUM_SLICES):
        for c in range(INTRA_SIZE):
            ax.text(c, r, f"{intra_matrix[r, c]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if intra_matrix[r, c] < 0.5 else "white")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{title_prefix}\n"
        "Distribution of codebook, ordering, and intra-slice action modes",
        fontsize=11, fontweight="bold", y=1.02
    )

    out_png = os.path.join(output_dir, f"{out_prefix}_action_distribution_compact.png")
    out_pdf = os.path.join(output_dir, f"{out_prefix}_action_distribution_compact.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


def print_statistics(all_dist, scen_dist_map):
    all_cb, all_ord, all_intra = all_dist
    print("\n=== Action Distribution Statistics ===")
    print(f"\nCodebook (frequency):")
    for i, freq in enumerate(all_cb):
        bar = "█" * int(freq * 50)
        print(f"  Mode {i:2d}: {freq:.3f}  {bar}")
    print(f"  Entropy: {-np.sum(all_cb * np.log(all_cb + 1e-9)):.4f} / {np.log(CODEBOOK_SIZE):.4f} (max)")

    print(f"\nOrdering (mean priority score per slice):")
    for s in range(NUM_SLICES):
        mean_score = np.dot(all_ord[s], np.arange(ORDER_SIZE))
        print(f"  Slice {s}: mean_score={mean_score:.3f}, dist={np.round(all_ord[s], 3)}")

    print(f"\nIntra (mode distribution per slice):")
    for s in range(NUM_SLICES):
        print(f"  Slice {s}: conservative={all_intra[s][0]:.3f}, "
              f"medium={all_intra[s][1]:.3f}, aggressive={all_intra[s][2]:.3f}")

    print(f"\n--- Per-scenario Codebook entropy ---")
    for sc, (scb, _, _) in sorted(scen_dist_map.items()):
        ent = -np.sum(scb * np.log(scb + 1e-9))
        top_mode = np.argmax(scb)
        print(f"  Scenario s{sc}: entropy={ent:.4f}, dominant_mode={top_mode} ({scb[top_mode]:.2%})")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot action distributions from dataset or rollout metric_raw.")
    parser.add_argument("--source", choices=["dataset", "rollout"], default="dataset")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--metric_raw_dir", type=str, default=ROLLOUT_RAW_DIR)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--out_prefix", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载所有动作 ──────────────────────────────────────────
    if args.source == "dataset":
        all_actions, scen_actions = load_all_actions(args.dataset_dir)
        out_prefix = args.out_prefix or "x3_dataset"
        title_prefix = "Dataset Action Distribution (D-B training trajectories)"
    else:
        all_actions, scen_actions = load_actions_from_metric_raw(args.metric_raw_dir, args.scenarios)
        out_prefix = args.out_prefix or "x3_rollout"
        title_prefix = "Rollout Action Distribution (IDT policy)"

    # ── 2. 计算分布 ──────────────────────────────────────────────
    all_dist  = compute_distributions(all_actions)
    scen_dist_map = {s: compute_distributions(a) for s, a in sorted(scen_actions.items())}

    # ── 3. 打印统计 ──────────────────────────────────────────────
    print_statistics(all_dist, scen_dist_map)

    # ── 4. 绘图 ──────────────────────────────────────────────────
    print("\nPlotting...")
    plot_main_distributions(all_dist, scen_dist_map, args.output_dir, out_prefix, title_prefix)
    plot_per_scenario(all_dist, scen_dist_map, args.output_dir, out_prefix, title_prefix)
    plot_compact_summary(all_dist, args.output_dir, out_prefix, title_prefix)

    print("\nDone!")


if __name__ == "__main__":
    main()
