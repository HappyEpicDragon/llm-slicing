"""
敏感性分析绘图脚本（Fig.10/11）。

Fig.10：Context Length 敏感性曲线（D4 实验）
Fig.11：Dataset Size 敏感性曲线（N1 实验）

支持合并为 2×2 大图（plot_combined_sensitivity），节省论文版面。
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional

from src.basic_apis.metrics_utils.plot_style import setup_style, METHOD_STYLES


def load_sensitivity_data(json_root: str, x_values: list, scenario_ids: list = None) -> tuple:
    """
    从标准 metric_json 目录加载多个 x 值对应的结果。

    Args:
        json_root: 每个 x 值对应一个子目录，子目录名为 str(x)，内含 summary.json
                   例如 data/channel_generality/dt_ctx{ctx}/metric_json/
        x_values:  横轴取值列表
        scenario_ids: 要聚合的场景列表（None 表示聚合所有找到的场景）

    Returns:
        (returns_mean, returns_std, viols_mean, viols_std) — 每个数组长度 = len(x_values)
    """
    returns_mean, returns_std = [], []
    viols_mean, viols_std = [], []

    for x in x_values:
        x_dir = str(x)
        # 查找该 x 下所有场景的 summary
        r_vals, v_vals = [], []
        for scen_id in (scenario_ids or [5, 6, 7, 8, 9]):
            summary_path = os.path.join(json_root, x_dir, "metric_json",
                                        f"scenario_{scen_id}", "summary.json")
            if not os.path.exists(summary_path):
                # 也尝试以 x 为目录名的顶级 summary
                summary_path = os.path.join(json_root.replace("{x}", x_dir),
                                            "metric_json", f"scenario_{scen_id}", "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                r_vals.append(data.get('reward_mean', 0.0))
                v_vals.append(data.get('hp_viol_mean', 0.0))

        returns_mean.append(float(np.mean(r_vals)) if r_vals else 0.0)
        returns_std.append(float(np.std(r_vals)) if r_vals else 0.0)
        viols_mean.append(float(np.mean(v_vals)) if v_vals else 0.0)
        viols_std.append(float(np.std(v_vals)) if v_vals else 0.0)

    return (np.array(returns_mean), np.array(returns_std),
            np.array(viols_mean), np.array(viols_std))


def plot_context_sensitivity(
    ctx_lengths: List[int],
    returns_mean: np.ndarray,
    returns_std: np.ndarray,
    viols_mean: np.ndarray,
    viols_std: np.ndarray,
    save_path: str,
    current_ctx: int = 20,
):
    """
    绘制 Context Length 敏感性曲线（Fig.10）。
    双子图：左轴 Avg Reward，右轴 HP Violation Rate。
    """
    setup_style(font_size=9)
    idt_style = METHOD_STYLES['dt']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # 左图：Avg Reward
    ax1.plot(ctx_lengths, returns_mean, color=idt_style['color'], marker='o',
             markersize=4, linewidth=1.2, label='IDT')
    ax1.fill_between(ctx_lengths,
                     returns_mean - returns_std,
                     returns_mean + returns_std,
                     alpha=0.15, color=idt_style['color'])
    if current_ctx in ctx_lengths:
        idx = ctx_lengths.index(current_ctx)
        ax1.axvline(x=current_ctx, color='#888888', linestyle='--', linewidth=0.8,
                    label=f'Default (L={current_ctx})')
        ax1.plot(ctx_lengths[idx], returns_mean[idx], 'o',
                 color=idt_style['color'], markersize=7, markeredgecolor='k', markeredgewidth=0.5)

    ax1.set_xlabel("Context Length $L$", fontsize=9)
    ax1.set_ylabel("Avg Episode Return", fontsize=9)
    ax1.set_title("(a) Context Length vs Return", fontsize=9)
    ax1.legend(fontsize=7)

    # 右图：HP Violation Rate
    ax2.plot(ctx_lengths, viols_mean, color=idt_style['color'], marker='o',
             markersize=4, linewidth=1.2, label='IDT')
    ax2.fill_between(ctx_lengths,
                     viols_mean - viols_std,
                     viols_mean + viols_std,
                     alpha=0.15, color=idt_style['color'])
    if current_ctx in ctx_lengths:
        ax2.axvline(x=current_ctx, color='#888888', linestyle='--', linewidth=0.8,
                    label=f'Default (L={current_ctx})')

    ax2.set_xlabel("Context Length $L$", fontsize=9)
    ax2.set_ylabel("HP Violation Rate", fontsize=9)
    ax2.set_title("(b) Context Length vs HP Violation", fontsize=9)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    print(f"Context sensitivity plot saved to: {save_path}.{{png,pdf}}")
    plt.close(fig)


def plot_dataset_sensitivity(
    ratios: List[float],
    returns_mean: np.ndarray,
    returns_std: np.ndarray,
    viols_mean: np.ndarray,
    viols_std: np.ndarray,
    save_path: str,
):
    """
    绘制 Dataset Size 敏感性曲线（Fig.11）。
    """
    setup_style(font_size=9)
    idt_style = METHOD_STYLES['dt']

    x_pct = [r * 100 for r in ratios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # 左图：Avg Reward
    ax1.plot(x_pct, returns_mean, color=idt_style['color'], marker='D',
             markersize=4, linewidth=1.2, label='IDT')
    ax1.fill_between(x_pct,
                     returns_mean - returns_std,
                     returns_mean + returns_std,
                     alpha=0.15, color=idt_style['color'])
    ax1.set_xlabel("Dataset Size (%)", fontsize=9)
    ax1.set_ylabel("Avg Episode Return", fontsize=9)
    ax1.set_title("(a) Dataset Size vs Return", fontsize=9)
    ax1.set_xticks([10, 30, 50, 100])
    ax1.legend(fontsize=7)

    # 右图：HP Violation Rate
    ax2.plot(x_pct, viols_mean, color=idt_style['color'], marker='D',
             markersize=4, linewidth=1.2, label='IDT')
    ax2.fill_between(x_pct,
                     viols_mean - viols_std,
                     viols_mean + viols_std,
                     alpha=0.15, color=idt_style['color'])
    ax2.set_xlabel("Dataset Size (%)", fontsize=9)
    ax2.set_ylabel("HP Violation Rate", fontsize=9)
    ax2.set_title("(b) Dataset Size vs HP Violation", fontsize=9)
    ax2.set_xticks([10, 30, 50, 100])
    ax2.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    print(f"Dataset sensitivity plot saved to: {save_path}.{{png,pdf}}")
    plt.close(fig)


def plot_combined_sensitivity(
    ctx_lengths: List[int],
    ctx_returns_mean: np.ndarray,
    ctx_returns_std: np.ndarray,
    ctx_viols_mean: np.ndarray,
    ctx_viols_std: np.ndarray,
    ratios: List[float],
    ds_returns_mean: np.ndarray,
    ds_returns_std: np.ndarray,
    ds_viols_mean: np.ndarray,
    ds_viols_std: np.ndarray,
    save_path: str,
    current_ctx: int = 20,
):
    """
    合并 Fig.10 和 Fig.11 为 2×2 大图（节省论文版面）。
    """
    setup_style(font_size=9)
    idt_style = METHOD_STYLES['dt']
    color = idt_style['color']

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    def _plot_line(ax, x, ymean, ystd, xlabel, ylabel, title, vline_x=None):
        ax.plot(x, ymean, color=color, marker='o', markersize=3.5, linewidth=1.2)
        ax.fill_between(x, ymean - ystd, ymean + ystd, alpha=0.15, color=color)
        if vline_x is not None and vline_x in x:
            ax.axvline(x=vline_x, color='#888888', linestyle='--', linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)

    _plot_line(axes[0], ctx_lengths, ctx_returns_mean, ctx_returns_std,
               "Context Length $L$", "Avg Return", "(a) Context vs Return",
               vline_x=current_ctx)
    _plot_line(axes[1], ctx_lengths, ctx_viols_mean, ctx_viols_std,
               "Context Length $L$", "HP Viol. Rate", "(b) Context vs HP Viol.",
               vline_x=current_ctx)

    x_pct = [r * 100 for r in ratios]
    _plot_line(axes[2], x_pct, ds_returns_mean, ds_returns_std,
               "Dataset Size (%)", "Avg Return", "(c) Dataset Size vs Return")
    _plot_line(axes[3], x_pct, ds_viols_mean, ds_viols_std,
               "Dataset Size (%)", "HP Viol. Rate", "(d) Dataset Size vs HP Viol.")
    axes[2].set_xticks([10, 30, 50, 100])
    axes[3].set_xticks([10, 30, 50, 100])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    print(f"Combined sensitivity plot saved to: {save_path}.{{png,pdf}}")
    plt.close(fig)


def plot_sensitivity_from_cfg(cfg=None):
    """从 Hydra cfg 构建参数并绘图（供 channel_generality.py 调用）"""
    if cfg is None:
        print("Warning: cfg is None, skipping plot_sensitivity")
        return

    s_cfg = cfg.get('plot_sensitivity', cfg)

    # Context Length 数据
    ctx_root = str(s_cfg.get('ctx_data_root', 'data/channel_generality'))
    ctx_lengths = list(s_cfg.get('ctx_lengths', [5, 10, 20, 30, 50]))
    current_ctx = int(s_cfg.get('current_ctx', 20))
    ctx_save = str(s_cfg.get('ctx_save_path', 'outputs/figures/channel_generality/sensitivity/context_sensitivity'))

    # Dataset Size 数据
    ds_root = str(s_cfg.get('ds_data_root', 'data/channel_generality'))
    ratios = list(s_cfg.get('dataset_ratios', [0.1, 0.3, 0.5, 1.0]))
    ds_save = str(s_cfg.get('ds_save_path', 'outputs/figures/channel_generality/sensitivity/dataset_sensitivity'))

    combined_save = str(s_cfg.get('combined_save_path', 'outputs/figures/channel_generality/sensitivity/combined_sensitivity'))

    ctx_rm, ctx_rs, ctx_vm, ctx_vs = load_sensitivity_data(
        ctx_root.replace("{x}", "{x}"), ctx_lengths
    )
    ds_rm, ds_rs, ds_vm, ds_vs = load_sensitivity_data(
        ds_root.replace("{x}", "{x}"), [int(r * 100) for r in ratios]
    )

    plot_context_sensitivity(ctx_lengths, ctx_rm, ctx_rs, ctx_vm, ctx_vs,
                             ctx_save, current_ctx)
    plot_dataset_sensitivity(ratios, ds_rm, ds_rs, ds_vm, ds_vs, ds_save)
    plot_combined_sensitivity(
        ctx_lengths, ctx_rm, ctx_rs, ctx_vm, ctx_vs,
        ratios, ds_rm, ds_rs, ds_vm, ds_vs,
        combined_save, current_ctx
    )
