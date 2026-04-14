"""
Transfer PPO Adaptation Curve 绘图脚本（Fig.9）。

绘制：不同目标场景（S_C, S_E）的 Transfer PPO fine-tuning 曲线
      + IDT zero-shot 性能水平参考线（含 shaded band）。
双 X 轴：底轴 Online Adaptation Steps，顶轴 Online Interaction Samples。
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional

from src.basic_apis.metrics_utils.plot_style import setup_style, METHOD_STYLES, SCENARIO_LABELS


def load_transfer_data(csv_path: str) -> dict:
    """
    从 test_ppo_baseline_finetune 生成的 CSV 读取适配曲线数据。
    返回: {ckp_index: {"mean_return": float, "hp_violation_rate": float}}
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        idx = int(row.get('ckp_index', -1))
        # 只保留真实 checkpoint；Best_Model 常用 -1，不属于在线适配曲线点
        if idx < 0:
            continue
        result[idx] = {
            "mean_return": float(row.get('mean_return', 0.0)),
            "hp_violation_rate": float(row.get('hp_violation_rate', 0.0)),
            "nhp_violation_rate": float(row.get('nhp_violation_rate', 0.0)),
        }
    return result


def load_idt_baseline(json_path: str, metric: str) -> tuple:
    """
    从 DT test 的标准 JSON 读取 IDT zero-shot 基线（与 metric 对齐）。
    返回: (mean_return, std_return)
    """
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, 'r') as f:
        data = json.load(f)
    metric_key_map = {
        "mean_return": ("reward_mean", "reward_std"),
        "hp_violation_rate": ("hp_viol_mean", "hp_viol_std"),
        "nhp_violation_rate": ("nhp_viol_mean", "nhp_viol_std"),
    }
    mean_key, std_key = metric_key_map.get(metric, ("reward_mean", "reward_std"))
    return float(data.get(mean_key, 0.0)), float(data.get(std_key, 0.0))


def plot_transfer_curve(
    transfer_data: dict,
    idt_baselines: dict,
    save_path: str,
    steps_per_ckp: int = 100,
    samples_per_step: int = 500,
    metric: str = "mean_return",
    title: str = "Transfer PPO Adaptation Curve",
):
    """
    绘制 Transfer PPO 适配曲线（Fig.9）。

    Args:
        transfer_data: {scenario_label: {ckp_index: {"mean_return": ..., "hp_violation_rate": ...}}}
                       例如 {"$S_C$": {0: {...}, 1: {...}}, "$S_E$": {...}}
        idt_baselines: {scenario_label: (mean, std)}
                       IDT zero-shot 性能（水平参考线）
        save_path:     输出文件路径（不含后缀）
        steps_per_ckp: 每个 checkpoint 对应的 fine-tuning 步数
        samples_per_step: 每步对应的在线交互样本数（用于顶轴）
        metric:        绘制的指标，"mean_return" 或 "hp_violation_rate"
        title:         图表标题
    """
    setup_style(font_size=9)

    scenario_colors = {
        '$S_C$': '#e6194b',
        '$S_E$': '#3cb44b',
        'Scenario 7': '#e6194b',
        'Scenario 9': '#3cb44b',
    }
    default_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for i, (scen_label, curve_data) in enumerate(transfer_data.items()):
        sorted_items = sorted(curve_data.items())
        ckp_indices = [k for k, _ in sorted_items]
        values = [v[metric] for _, v in sorted_items]

        x_steps = [k * steps_per_ckp for k in ckp_indices]
        color = scenario_colors.get(scen_label, default_colors[i % len(default_colors)])

        ax.plot(x_steps, values,
                color=color, marker='o', markersize=3, linewidth=1.2,
                label=f'Transfer PPO → {scen_label}')

        # IDT zero-shot 参考线
        if scen_label in idt_baselines:
            idt_mean, idt_std = idt_baselines[scen_label]
            if idt_mean is not None:
                ax.axhline(y=idt_mean, color=color, linestyle=':', linewidth=1.0,
                           alpha=0.8)
                if idt_std is not None and idt_std > 0:
                    ax.axhspan(idt_mean - idt_std, idt_mean + idt_std,
                               alpha=0.08, color=color)

    # IDT 标注
    ax.annotate(
        'IDT: Zero online samples required',
        xy=(0.02, 0.95), xycoords='axes fraction',
        fontsize=7, style='italic', ha='left', va='top',
        color='#555555',
    )

    ylabel_map = {
        "mean_return": "Episode Return",
        "hp_violation_rate": "HP Violation Rate",
        "nhp_violation_rate": "NHP Violation Rate",
    }
    ax.set_xlabel("Online Adaptation Steps", fontsize=9)
    ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='lower right')

    # 顶轴：Online Interaction Samples
    ax_top = ax.twiny()
    ax_top.set_xlim(
        ax.get_xlim()[0] * samples_per_step,
        ax.get_xlim()[1] * samples_per_step
    )
    ax_top.set_xlabel("Online Interaction Samples", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    print(f"Transfer curve saved to: {save_path}.{{png,pdf}}")
    plt.close(fig)


def plot_transfer_curve_from_cfg(cfg=None):
    """从 Hydra cfg 构建参数并绘图"""
    if cfg is None:
        print("Warning: cfg is None, skipping plot_transfer_curve")
        return

    tc_cfg = cfg.get('plot_transfer_curve', cfg)

    # 加载各场景的适配曲线数据
    data_root = str(tc_cfg.get('data_root', 'data/channel_generality/transfer_ppo'))
    idt_root = str(tc_cfg.get('idt_root', 'data/channel_generality/dt'))
    save_path = str(tc_cfg.get('save_path', 'plots/transfer_curve'))
    metric = str(tc_cfg.get('metric', 'mean_return'))
    target_scenarios = list(tc_cfg.get('target_scenarios', [7, 9]))
    steps_per_ckp = int(tc_cfg.get('steps_per_ckp', 100))
    samples_per_step = int(tc_cfg.get('samples_per_step', 500))

    transfer_data = {}
    idt_baselines = {}

    for scen_id in target_scenarios:
        scen_label = SCENARIO_LABELS.get(scen_id, f'Scenario {scen_id}')
        csv_path = os.path.join(data_root, f"scenario_{scen_id}", "finetune_adaptation_curve.csv")

        if os.path.exists(csv_path):
            transfer_data[scen_label] = load_transfer_data(csv_path)
        else:
            print(f"Warning: {csv_path} not found, skipping scenario {scen_id}")

        idt_json = os.path.join(idt_root, "metric_json", f"scenario_{scen_id}", "summary.json")
        idt_mean, idt_std = load_idt_baseline(idt_json, metric)
        if idt_mean is not None:
            idt_baselines[scen_label] = (idt_mean, idt_std)

    if not transfer_data:
        print("No transfer data found, skipping plot")
        return

    plot_transfer_curve(
        transfer_data=transfer_data,
        idt_baselines=idt_baselines,
        save_path=save_path,
        steps_per_ckp=steps_per_ckp,
        samples_per_step=samples_per_step,
        metric=metric,
    )
