# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ============================================================
# 工具函数
# ============================================================

def _read_progress_csv(csv_path: str) -> dict:
    """读取 Ray Tune 的 progress.csv，返回 {列名: [float, ...]} 字典。"""
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                val = row[col]
                try:
                    data[col].append(float(val))
                except (ValueError, TypeError):
                    data[col].append(float('nan'))
    return data


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """对序列做中心式移动平均，首尾用 pad 填充以保持长度不变。"""
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _moving_std(arr: np.ndarray, window: int) -> np.ndarray:
    """对序列做局部标准差（用于 shaded band），长度与 arr 相同。"""
    result = np.zeros_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = np.std(arr[lo:hi])
    return result


# ============================================================
# 主绘图函数
# ============================================================

def plot_finetune_entropy(cfg):
    """
    绘制 Fine-tuned PPO 训练过程中的 Policy Entropy 曲线，
    用于说明 Negative Transfer 现象。

    图形内容：
      - 主曲线：inter_slice_sched entropy（爆炸的那条，Negative Transfer 信号）
      - 对比曲线：intra_slice_sched entropy（相对稳定，形成对比）
      - 垂直虚线标注 "collapse zone" 起始迭代
      - 移动平均 + shaded confidence band
    """
    entropy_cfg = cfg.plot_finetune_entropy
    csv_path    = entropy_cfg.progress_csv_path
    save_dir    = entropy_cfg.save_dir
    smooth_win  = int(entropy_cfg.get("smooth_window", 20))
    collapse_iter = int(entropy_cfg.get("collapse_iter", 300))
    dpi         = int(entropy_cfg.get("dpi", 300))

    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. 读取数据
    # ----------------------------------------------------------
    print(f"[plot_finetune_entropy] Reading: {csv_path}")
    data = _read_progress_csv(csv_path)

    inter_key = "info/learner/inter_slice_sched/learner_stats/entropy"
    intra_key = "info/learner/intra_slice_sched/learner_stats/entropy"

    for key in [inter_key, intra_key]:
        if key not in data:
            raise KeyError(f"Column not found in progress.csv: '{key}'")

    inter_raw = np.array(data[inter_key])
    intra_raw = np.array(data[intra_key])
    iters     = np.arange(len(inter_raw))

    # 过滤 NaN（Ray Tune 偶尔会有空行）
    valid_mask  = ~(np.isnan(inter_raw) | np.isnan(intra_raw))
    iters       = iters[valid_mask]
    inter_raw   = inter_raw[valid_mask]
    intra_raw   = intra_raw[valid_mask]

    # ----------------------------------------------------------
    # 2. 平滑处理
    # ----------------------------------------------------------
    inter_smooth = _moving_average(inter_raw, smooth_win)
    intra_smooth = _moving_average(intra_raw, smooth_win)
    inter_std    = _moving_std(inter_raw, smooth_win)
    intra_std    = _moving_std(intra_raw, smooth_win)

    # ----------------------------------------------------------
    # 3. 样式定义（与项目 TWC 风格保持一致）
    # ----------------------------------------------------------
    COLOR_INTER = '#d62728'   # 砖红，主角（崩溃曲线）
    COLOR_INTRA = '#1f77b4'   # 经典蓝，对比曲线
    COLOR_COLLAPSE = '#555555'

    style_inter = dict(color=COLOR_INTER, linewidth=2.0, zorder=3,
                       label='Inter-slice Scheduler Entropy')
    style_intra = dict(color=COLOR_INTRA, linewidth=2.0, zorder=3,
                       label='Intra-slice Scheduler Entropy')

    # ----------------------------------------------------------
    # 4. 绘图
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # --- 原始数据（半透明细线，展示真实噪声） ---
    ax.plot(iters, inter_raw, color=COLOR_INTER, linewidth=0.5, alpha=0.25, zorder=1)
    ax.plot(iters, intra_raw, color=COLOR_INTRA, linewidth=0.5, alpha=0.25, zorder=1)

    # --- 移动平均主曲线 ---
    ax.plot(iters, inter_smooth, **style_inter)
    ax.plot(iters, intra_smooth, **style_intra)

    # --- Shaded band（± 1 局部标准差） ---
    ax.fill_between(iters,
                    inter_smooth - inter_std,
                    inter_smooth + inter_std,
                    color=COLOR_INTER, alpha=0.12, zorder=2)
    ax.fill_between(iters,
                    intra_smooth - intra_std,
                    intra_smooth + intra_std,
                    color=COLOR_INTRA, alpha=0.12, zorder=2)

    # --- Collapse zone 标注 ---
    ax.axvline(x=collapse_iter, color=COLOR_COLLAPSE,
               linestyle='--', linewidth=1.4, zorder=4)
    ax.text(collapse_iter + 8,
            ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1.0,
            'Policy Collapse Zone',
            color=COLOR_COLLAPSE, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # --- 背景着色（collapse zone 之后） ---
    ax.axvspan(collapse_iter, iters[-1],
               color=COLOR_COLLAPSE, alpha=0.05, zorder=0)

    # ----------------------------------------------------------
    # 5. 坐标轴与样式（TWC 学术风格）
    # ----------------------------------------------------------
    ax.set_xlabel('Fine-tuning Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Policy Entropy', fontsize=12, fontweight='bold')
    ax.set_xlim(iters[0], iters[-1])
    ax.set_ylim(bottom=0)

    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # --- 图例（手动构建，加入原始数据说明） ---
    legend_handles = [
        Line2D([0], [0], color=COLOR_INTER, linewidth=2.0,
               label='Inter-slice Scheduler (Moving Avg.)'),
        Line2D([0], [0], color=COLOR_INTRA, linewidth=2.0,
               label='Intra-slice Scheduler (Moving Avg.)'),
        mpatches.Patch(facecolor=COLOR_INTER, alpha=0.25,
                       label='Inter-slice Raw'),
        mpatches.Patch(facecolor=COLOR_INTRA, alpha=0.25,
                       label='Intra-slice Raw'),
        Line2D([0], [0], color=COLOR_COLLAPSE, linewidth=1.4,
               linestyle='--', label=f'Collapse Onset (iter={collapse_iter})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              frameon=True, framealpha=0.9, fontsize=9,
              edgecolor='#cccccc')

    plt.tight_layout()

    # ----------------------------------------------------------
    # 6. 保存
    # ----------------------------------------------------------
    save_path_png = os.path.join(save_dir, 'finetune_policy_entropy.png')
    save_path_pdf = os.path.join(save_dir, 'finetune_policy_entropy.pdf')
    fig.savefig(save_path_png, dpi=dpi, bbox_inches='tight')
    fig.savefig(save_path_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"[plot_finetune_entropy] Saved → {save_path_png}")
    print(f"[plot_finetune_entropy] Saved → {save_path_pdf}")
