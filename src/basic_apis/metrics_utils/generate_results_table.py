"""
综合结果表生成脚本（Tab.III / Tab.IV）。

自动扫描标准 JSON 目录，聚合所有方法在所有测试场景上的 mean ± std，
生成 LaTeX tabular 格式（符合 IEEE TWC 要求，最优值自动加粗）。

用法：
    python main.py name=channel_generality mode=generate_results_table

    # 或直接调用函数
    python -c "
    from src.basic_apis.metrics_utils.generate_results_table import generate_latex_table
    generate_latex_table(...)
    "
"""
import os
import json
import numpy as np
from omegaconf import DictConfig
from typing import Optional

from src.basic_apis.metrics_utils.plot_style import METHOD_ORDER, METHOD_STYLES, SCENARIO_LABELS


# ==============================================================================
# 数据加载
# ==============================================================================

def load_method_results(
    data_root: str,
    method_key: str,
    scenario_ids: list,
    metric: str = "hp_viol",
) -> dict:
    """
    从标准 JSON 目录加载单个方法在所有场景上的结果。

    返回:
        {scenario_id: (mean, std)}
    """
    metric_file_map = {
        "hp_viol":  "hp_violations.json",
        "nhp_viol": "nhp_violations.json",
        "hp_dist":  "hp_distance.json",
        "nhp_dist": "nhp_distance.json",
        "reward":   "episode_rewards.json",
    }
    fname = metric_file_map.get(metric, f"{metric}.json")
    method_dir = os.path.join(data_root, method_key, "metric_json")

    results = {}
    for scen_id in scenario_ids:
        scen_dir = os.path.join(method_dir, f"scenario_{scen_id}")

        # 汇总所有 seed 的数据
        seed_means = []
        for seed_name in sorted(os.listdir(scen_dir)) if os.path.exists(scen_dir) else []:
            seed_path = os.path.join(scen_dir, seed_name, fname)
            if os.path.exists(seed_path):
                with open(seed_path) as f:
                    data = json.load(f)
                if "mean" in data:
                    seed_means.append(float(data["mean"]))

        # 也尝试直接读 summary.json（场景级聚合）
        if not seed_means:
            summary_path = os.path.join(scen_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    data = json.load(f)
                mean_key = f"{metric.replace('_viol', '_viol_')}_mean".replace("reward_", "reward_")
                # 适配不同命名方式
                for key in [f"{metric}_mean", "hp_viol_mean", "nhp_viol_mean",
                            "hp_dist_mean", "nhp_dist_mean", "reward_mean",
                            f"{metric.split('_')[0]}_viol_mean"]:
                    if key in data:
                        results[scen_id] = (float(data[key]), float(data.get(key.replace('_mean', '_std'), 0.0)))
                        break
                continue

        if seed_means:
            results[scen_id] = (float(np.mean(seed_means)), float(np.std(seed_means)))
        else:
            results[scen_id] = (float('nan'), float('nan'))

    return results


def load_all_methods(data_root: str, method_keys: list, scenario_ids: list,
                     metric: str) -> dict:
    """
    加载所有方法在所有场景上的指定 metric 结果。

    Returns:
        {method_key: {scenario_id: (mean, std)}}
    """
    all_results = {}
    for method_key in method_keys:
        all_results[method_key] = load_method_results(data_root, method_key,
                                                       scenario_ids, metric)
    return all_results


# ==============================================================================
# LaTeX 表格生成
# ==============================================================================

def _fmt_cell(mean: float, std: float, bold: bool = False, lower_better: bool = True) -> str:
    """格式化单元格：mean ± std，最优值加粗"""
    if np.isnan(mean):
        cell = "—"
    else:
        cell = f"{mean:.3f}{{\\tiny$\\pm${std:.3f}}}"
    if bold:
        cell = f"\\textbf{{{cell}}}"
    return cell


def generate_latex_table(
    results_dict: dict,
    scenario_ids: list,
    metric: str = "hp_viol",
    output_path: Optional[str] = None,
    caption: str = "Comprehensive Performance Comparison",
    label: str = "tab:main_results",
    lower_better: bool = True,
) -> str:
    """
    生成 LaTeX tabular 代码。

    Args:
        results_dict: {method_key: {scenario_id: (mean, std)}}
        scenario_ids: 列对应的场景 ID 列表
        metric:       指标名（用于列标题）
        output_path:  输出 .tex 文件路径（None 表示不保存）
        caption:      表格标题
        label:        LaTeX label
        lower_better: True 表示越低越好（加粗最小值）

    Returns:
        LaTeX 代码字符串
    """
    method_keys = [k for k in METHOD_ORDER if k in results_dict]
    scen_labels = [SCENARIO_LABELS.get(s, f"$S_{{{s}}}$") for s in scenario_ids]

    n_cols = len(scenario_ids) + 2  # Method + N scenarios + Avg
    col_spec = "l" + "r" * (len(scenario_ids) + 1)

    metric_display = {
        "hp_viol":  "HP Viol. $\\downarrow$",
        "nhp_viol": "NHP Viol. $\\downarrow$",
        "hp_dist":  "HP Dist. $\\downarrow$",
        "nhp_dist": "NHP Dist. $\\downarrow$",
        "reward":   "Episode Return $\\uparrow$",
    }.get(metric, metric)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # 表头
    header_cols = ["Method"] + scen_labels + ["Avg"]
    lines.append(" & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")

    # 对每个场景找最优值（用于加粗）
    best_per_scen = {}
    for scen_id in scenario_ids:
        vals = []
        for mk in method_keys:
            v, _ = results_dict[mk].get(scen_id, (float('nan'), 0.0))
            if not np.isnan(v):
                vals.append(v)
        if vals:
            best_per_scen[scen_id] = min(vals) if lower_better else max(vals)
        else:
            best_per_scen[scen_id] = float('nan')

    # 数据行
    for mk in method_keys:
        label_str = METHOD_STYLES.get(mk, {}).get('label', mk)
        row_means = []
        cells = [label_str]

        for scen_id in scenario_ids:
            mean, std = results_dict[mk].get(scen_id, (float('nan'), 0.0))
            is_best = not np.isnan(mean) and not np.isnan(best_per_scen.get(scen_id, float('nan')))
            if is_best:
                is_best = abs(mean - best_per_scen[scen_id]) < 1e-9
            cells.append(_fmt_cell(mean, std, bold=is_best, lower_better=lower_better))
            row_means.append(mean)

        valid = [v for v in row_means if not np.isnan(v)]
        avg = float(np.mean(valid)) if valid else float('nan')
        cells.append(f"{avg:.3f}" if not np.isnan(avg) else "—")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_code = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                    exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table saved to: {output_path}")

    return latex_code


# ==============================================================================
# 主入口
# ==============================================================================

def generate_results_table(cfg: DictConfig, path_context):
    """综合结果表生成主入口（channel_generality.py 调用）"""
    table_cfg = cfg.get('generate_results_table', cfg)

    data_root = str(table_cfg.get('data_root', 'data/channel_generality'))
    save_dir = str(table_cfg.get('save_dir', 'outputs/latex_tables'))
    scenario_ids = list(table_cfg.get('scenario_ids', [5, 6, 7, 8, 9]))
    method_keys = list(table_cfg.get('method_keys', METHOD_ORDER))

    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("hp_viol",  True,  "Tab.III HP Violation"),
        ("nhp_viol", True,  "Tab.III NHP Violation"),
        ("hp_dist",  True,  "Tab.III HP Distance"),
        ("nhp_dist", True,  "Tab.III NHP Distance"),
        # reward 各算法不具可比性（reward 函数定义不同），不纳入对比表
    ]

    for metric, lower_better, desc in metrics:
        print(f"\nGenerating table: {desc} ...")
        results = load_all_methods(data_root, method_keys, scenario_ids, metric)

        latex = generate_latex_table(
            results_dict=results,
            scenario_ids=scenario_ids,
            metric=metric,
            output_path=os.path.join(save_dir, f"table_{metric}.tex"),
            caption=f"Performance Comparison — {metric.replace('_', ' ').upper()} "
                    f"(mean $\\pm$ std over 5 seeds, 5 test scenarios)",
            label=f"tab:{metric}_results",
            lower_better=lower_better,
        )
        print(latex[:500] + "\n...")

    # 消融表（Tab.IV）
    ablation_methods = ['dt_fixed_rr', 'dt', 'dt_baseline']
    ablation_methods += [k for k in method_keys if 'ablation' in k or 'raw_action' in k
                         or 'reward_only' in k or 'no_rescue' in k or 'fixed_order' in k
                         or 'fixed_rr' in k]

    if len(ablation_methods) > 1:
        print("\nGenerating ablation table ...")
        ablation_results = load_all_methods(data_root, ablation_methods, scenario_ids, "hp_viol")
        generate_latex_table(
            results_dict=ablation_results,
            scenario_ids=scenario_ids,
            metric="hp_viol",
            output_path=os.path.join(save_dir, "table_ablation.tex"),
            caption="Ablation Study — HP Violation Rate",
            label="tab:ablation",
            lower_better=True,
        )

    print(f"\nAll tables saved to: {save_dir}")
