# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import json
#
#
# # ==========================================
# # 1. 配置
# # ==========================================
# class PlotConfig:
#     SCENARIOS = ['S5', 'S6', 'S7', 'S8', 'S9']
#     SCENARIO_IDS = [5, 6, 7, 8, 9]
#
#     MODELS_DISPLAY = ['PPO', 'PPO-HA', 'DT-Base', 'GRRM-DT (Ours)']
#     MODELS_KEYS = ['ppo_baseline', 'ppo_ha_weighted', 'dt_baseline', 'dt']
#
#     COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#00008B']
#     # 纹理配置
#     HATCHES = ['//', '..', '\\\\', '']
#
#     JSON_DIR = "./data/channel_generality"
#     SAVE_DIR = "./plots/bar_charts_final"
#
#     METRIC_MAPPING = {
#         'distance': 'normalized_distance_fulfill_cumsum',
#         'violation': 'normalized_violations_per_episode_cumsum'
#     }
#
#
# def load_real_data():
#     """ 加载真实数据 """
#     results = {}
#     for met_key, met_filename in PlotConfig.METRIC_MAPPING.items():
#         results[met_key] = {}
#         for typ in ['hp', 'nhp']:
#             matrix = np.zeros((len(PlotConfig.SCENARIO_IDS), len(PlotConfig.MODELS_KEYS)))
#             for m_idx, model_key in enumerate(PlotConfig.MODELS_KEYS):
#                 for s_row_idx, s_id in enumerate(PlotConfig.SCENARIO_IDS):
#                     filepath = os.path.join(
#                         PlotConfig.JSON_DIR, model_key, "metric_json",
#                         f"scenario_{s_id}", f"{met_filename}.json"
#                     )
#                     val = 0
#                     if os.path.exists(filepath):
#                         try:
#                             with open(filepath, 'r') as f:
#                                 data = json.load(f)
#                                 episode_values = data.get(typ, [])
#                                 if episode_values:
#                                     val = np.sum(episode_values)
#                         except Exception:
#                             pass
#                     matrix[s_row_idx, m_idx] = val
#             results[met_key][typ] = matrix
#     return results
#
#
# # ==========================================
# # 2. 核心绘图函数 (Annotate版)
# # ==========================================
# def plot_grouped_bar_chart(data_matrix, ax, title, y_label, show_legend=False):
#     n_groups = len(PlotConfig.SCENARIOS)
#     n_bars = len(PlotConfig.MODELS_DISPLAY)
#     bar_width = 0.2
#     index = np.arange(n_groups)
#
#     # 判断是否为负值图 (Distance)
#     is_negative_plot = np.sum(data_matrix) < -1e-5
#
#     # 遍历每个模型画柱子
#     for i in range(n_bars):
#         model_data = data_matrix[:, i]
#         color = PlotConfig.COLORS[i]
#         label = PlotConfig.MODELS_DISPLAY[i]
#         hatch = PlotConfig.HATCHES[i]
#
#         x_pos = index + (i - n_bars / 2 + 0.5) * bar_width
#
#         # 绘图
#         bars = ax.bar(x_pos, model_data, bar_width,
#                       label=label, color=color, alpha=0.85,
#                       edgecolor='black', linewidth=0.7, hatch=hatch)
#
#     # === 设置 Y 轴 Symlog ===
#     # 调整阈值，让 0 附近的柱子也能稍微显示出来一点
#     ax.set_yscale('symlog', linthresh=0.5)
#     ax.grid(True, which="major", ls="--", alpha=0.3)
#     ax.axhline(0, color='black', linewidth=0.8)
#
#     # === View Limit 微调 ===
#     # 仅为了防止 0 贴边，不需要留出 huge headroom，因为文字紧贴柱子了
#     data_min = np.min(data_matrix)
#     data_max = np.max(data_matrix)
#
#     if is_negative_plot:
#         # 负值图：强制顶部为 0
#         current_bottom = ax.get_ylim()[0]
#         # 稍微往下扩一点点，防止最长的柱子文字出界
#         ax.set_ylim(bottom=current_bottom * 1.15, top=0)
#     else:
#         # 正值图：强制底部为 0
#         current_top = ax.get_ylim()[1]
#         ax.set_ylim(bottom=0, top=current_top * 1.15)
#
#     # === 重新遍历添加标签 (使用 annotate) ===
#     # 必须在 set_yscale 之后进行，但 annotate 使用的是点坐标，所以不受 log 扭曲影响
#     for i in range(n_bars):
#         model_data = data_matrix[:, i]
#         x_pos = index + (i - n_bars / 2 + 0.5) * bar_width
#
#         for x, value in zip(x_pos, model_data):
#
#             # --- 1. 修复 -0 和 格式化 ---
#             if abs(value) < 1e-4:
#                 text_str = "0"
#             else:
#                 # 先转成两位小数
#                 text_str = f"{value:.2f}"
#                 # 修复 -0.00 的情况
#                 if text_str == "-0.00" or text_str == "-0":
#                     text_str = "0"
#                 else:
#                     # 去除末尾的 0 和 .
#                     text_str = text_str.rstrip('0').rstrip('.')
#
#             # --- 2. 颜色与字体 (取消加粗) ---
#             font_w = 'normal'
#             txt_col = 'black'
#
#             # --- 3. 核心定位 (使用 offset points) ---
#             # xy 是数据坐标(柱子顶端)，xytext 是屏幕偏移量
#
#             if is_negative_plot:
#                 # [Distance 图]: 即使是 0，也往下偏移
#                 # xytext=(0, -3) 表示向下偏移 3 个点
#                 offset_points = (0, -3)
#                 va_align = 'top'
#
#                 # 特殊处理：如果是 0，坐标定在 y=0
#                 target_y = 0 if abs(value) < 1e-4 else value
#
#             else:
#                 # [Violation 图]: 往上偏移
#                 # xytext=(0, 3) 表示向上偏移 3 个点
#                 offset_points = (0, 3)
#                 va_align = 'bottom'
#
#                 target_y = 0 if abs(value) < 1e-4 else value
#
#             ax.annotate(text_str,
#                         xy=(x, target_y),
#                         xytext=offset_points,
#                         textcoords="offset points",  # 关键！基于屏幕像素的偏移
#                         ha='center',
#                         va=va_align,
#                         fontsize=6,
#                         fontweight=font_w,
#                         color=txt_col)
#
#     # 设置坐标轴标签
#     ax.set_title(title, fontweight='bold', fontsize=12)
#     ax.set_ylabel(y_label, fontweight='bold')
#     ax.set_xticks(index)
#     ax.set_xticklabels(PlotConfig.SCENARIOS, fontweight='bold')
#
#     if show_legend:
#         ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
#                   ncol=4, fontsize=10, frameon=False)
#
#
# def plot_hist():
#     final_data = load_real_data()
#     # final_data = load_mock_data() # 测试用
#
#     sns.set_context("paper", font_scale=1.2)
#     sns.set_style("whitegrid")
#
#     fig, axes = plt.subplots(2, 2, figsize=(14, 11))
#     plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9)
#
#     # Row 1: Distance (负值)
#     plot_grouped_bar_chart(final_data['distance']['hp'], axes[0, 0],
#                            "Cumulative Distance (HP)", "Total Distance (Symlog)", show_legend=True)
#
#     plot_grouped_bar_chart(final_data['distance']['nhp'], axes[0, 1],
#                            "Cumulative Distance (NHP)", "Total Distance (Symlog)")
#
#     # Row 2: Violation (正值)
#     plot_grouped_bar_chart(final_data['violation']['hp'], axes[1, 0],
#                            "Cumulative Violations (HP)", "Total Count (Symlog)")
#
#     plot_grouped_bar_chart(final_data['violation']['nhp'], axes[1, 1],
#                            "Cumulative Violations (NHP)", "Total Count (Symlog)")
#
#     os.makedirs(PlotConfig.SAVE_DIR, exist_ok=True)
#     save_path = os.path.join(PlotConfig.SAVE_DIR, "final_perfect_fit_chart.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Chart saved to: {save_path}")
#
#
# # Mock Data for logic testing
# def load_mock_data():
#     results = {}
#     dist_base = -np.array([
#         [100.123, 20.5, 50.001, 0],
#         [500, 100, 200, -0.0001],  # 测试 -0
#         [2000, 500, 800, 0],
#         [8000, 1500, 2000, 0],
#         [50000, 10000, 12000, 0]
#     ])
#     results['distance'] = {'hp': dist_base, 'nhp': dist_base * 0.8}
#     vio_base = np.abs(dist_base) / 10
#     results['violation'] = {'hp': vio_base, 'nhp': vio_base * 0.5}
#     return results
#
#
# if __name__ == "__main__":
#     plot_hist()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

try:
    from src.basic_apis.metrics_utils.plot_style import METHOD_STYLES, setup_style
    _HAS_STYLE = True
except ImportError:
    _HAS_STYLE = False


# ==========================================
# 1. 配置（v2，与增补实验 Baseline 体系同步）
# ==========================================
class PlotConfig:
    SCENARIOS = ['A', 'B', 'C', 'D', 'E']
    SCENARIO_IDS = [5, 6, 7, 8, 9]
    N_SEEDS = 5

    # Fig.6 默认主对比方法
    MODELS_KEYS = [
        'ppo_multi',           # Multi-Scen. PPO [42]
        'ppo_lagrangian',      # Lagrangian PPO [20]
        'ppo_ha_weighted',     # PPO Discrete (Teacher)
        'dt_baseline',         # DT Baseline
        'dt',                  # Proposed IDT
    ]

    MODELS_DISPLAY = [
        'Multi-Scen. PPO [42]',
        'Lagrangian PPO [20]',
        'PPO Discrete (Teacher)',
        'DT-Baseline',
        'Proposed IDT',
    ]

    COLORS = [
        '#d62728',   # 红：Multi-Scen. PPO
        '#e377c2',   # 粉：Lagrangian PPO
        '#ff7f0e',   # 橙：PPO Discrete
        '#2ca02c',   # 绿：DT Baseline
        '#00008B',   # 深蓝：IDT
    ]

    HATCHES = ['//', '||', '\\\\', '--', '']

    JSON_DIR = "./data/channel_generality"
    SAVE_DIR = "./plots/bar_charts_final"

    # 指标文件名（新格式：seed 级目录下）
    METRIC_FILES = {
        'hp_dist':  'hp_distance.json',      # {"hp_distance": [...], "mean": float}
        'nhp_dist': 'nhp_distance.json',     # {"nhp_distance": [...], "mean": float}
        'hp_viol':  'hp_violations.json',    # {"hp_violations": [...], "mean": float}
        'nhp_viol': 'nhp_violations.json',   # {"nhp_violations": [...], "mean": float}
        'reward':   'episode_rewards.json',  # {"rewards": [...], "mean": float}
    }

    # 字段名映射
    METRIC_VALUE_KEY = {
        'hp_dist':  'mean',
        'nhp_dist': 'mean',
        'hp_viol':  'mean',
        'nhp_viol': 'mean',
        'reward':   'mean',
    }


# Fig.6 专用视觉编码（优先级高于 plot_style 中的通用设置）
FIG6_STYLE_OVERRIDES = {
    'ppo_multi':        {'color': '#d62728', 'hatch': '//',  'label': 'Multi-Scen. PPO [42]'},
    'ppo_lagrangian':   {'color': '#e377c2', 'hatch': '||',  'label': 'Lagrangian PPO [20]'},
    'ppo_ha_weighted':  {'color': '#ff7f0e', 'hatch': '\\\\', 'label': 'PPO-Discrete (Teacher)'},
    'dt_baseline':      {'color': '#2ca02c', 'hatch': '--',  'label': 'DT-Baseline'},
    'dt':               {'color': '#00008B', 'hatch': '',    'label': 'Proposed IDT'},
}


# ==========================================
# 2. 数据加载（新格式：seed_{N}/xxx.json）
# ==========================================
def load_real_data(data_root=None, model_keys=None, n_seeds=None):
    """
    从标准 seed-level JSON 加载数据，返回 mean + std 矩阵。

    Returns:
        results: dict[metric_key] = {
            'mean': ndarray [n_scenarios, n_models],
            'std':  ndarray [n_scenarios, n_models],
        }
    """
    if data_root is None:
        data_root = PlotConfig.JSON_DIR
    if model_keys is None:
        model_keys = PlotConfig.MODELS_KEYS
    if n_seeds is None:
        n_seeds = PlotConfig.N_SEEDS

    n_s = len(PlotConfig.SCENARIO_IDS)
    n_m = len(model_keys)
    results = {}

    for met_key, met_file in PlotConfig.METRIC_FILES.items():
        val_key = PlotConfig.METRIC_VALUE_KEY[met_key]
        mean_mat = np.full((n_s, n_m), np.nan)
        std_mat = np.zeros((n_s, n_m))

        for m_idx, model_key in enumerate(model_keys):
            for s_row_idx, s_id in enumerate(PlotConfig.SCENARIO_IDS):
                seed_vals = []
                for seed in range(n_seeds):
                    filepath = os.path.join(
                        data_root, model_key, "metric_json",
                        f"scenario_{s_id}", f"seed_{seed}", met_file
                    )
                    if os.path.exists(filepath):
                        try:
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                            seed_vals.append(float(data.get(val_key, 0.0)))
                        except Exception:
                            pass

                if seed_vals:
                    mean_mat[s_row_idx, m_idx] = float(np.mean(seed_vals))
                    std_mat[s_row_idx, m_idx] = float(np.std(seed_vals))
                else:
                    mean_mat[s_row_idx, m_idx] = 0.0

        results[met_key] = {'mean': mean_mat, 'std': std_mat}

    return results


# ==========================================
# 3. 核心绘图函数（支持 error bar + 8 方法）
# ==========================================
def plot_grouped_bar_chart(mean_matrix, std_matrix, ax, title, y_label,
                           show_legend=False, model_keys=None):
    if model_keys is None:
        model_keys = PlotConfig.MODELS_KEYS

    n_groups = len(PlotConfig.SCENARIOS)
    n_bars = len(model_keys)
    bar_width = 0.085
    index = np.arange(n_groups)

    linthresh = 0.5

    for i, model_key in enumerate(model_keys):
        # 优先使用 Fig.6 专用样式，其次使用 plot_style 通用样式
        if model_key in FIG6_STYLE_OVERRIDES:
            style = FIG6_STYLE_OVERRIDES[model_key]
            color = style.get('color', '#333333')
            hatch = style.get('hatch', '')
            label = style.get('label', model_key)
        elif _HAS_STYLE and model_key in METHOD_STYLES:
            style = METHOD_STYLES[model_key]
            color = style.get('color', '#333333')
            hatch = style.get('hatch', '')
            label = style.get('label', model_key)
        else:
            try:
                color = PlotConfig.COLORS[i]
                hatch = PlotConfig.HATCHES[i]
                label = PlotConfig.MODELS_DISPLAY[i] if i < len(PlotConfig.MODELS_DISPLAY) else model_key
            except IndexError:
                color, hatch, label = '#333333', '', model_key

        x_pos = index + (i - n_bars / 2 + 0.5) * bar_width
        raw_data = mean_matrix[:, i]
        err_data = std_matrix[:, i]

        # 将 NaN 归零（无数据时柱子高度为 0）
        plot_raw = np.nan_to_num(raw_data, nan=0.0)
        plot_err = np.nan_to_num(err_data, nan=0.0)

        ax.bar(x_pos, plot_raw, bar_width,
               label=label, color=color, alpha=0.85,
               edgecolor='black', linewidth=0.5, hatch=hatch,
               yerr=plot_err, capsize=2,
               error_kw={"elinewidth": 1.0, "ecolor": "black", "capthick": 1.0})

        # 0 值柱在视觉上会“消失”，这里显式画一个短 stub 并标注 0。
        # 保持数据值不变（仍为 0），只做视觉可见性增强。
        finite_mask = np.isfinite(raw_data)
        nonzero_mask = finite_mask & (np.abs(raw_data) > 1e-12)
        zero_mask = finite_mask & ~nonzero_mask
        if np.any(zero_mask):
            nonzero_vals = raw_data[nonzero_mask]
            # 优先沿该序列主方向显示（distance 子图通常为负，violation 通常为正）
            direction = -1.0 if (nonzero_vals.size > 0 and float(np.mean(nonzero_vals)) < 0.0) else 1.0
            y_stub = direction * linthresh * 0.22
            x_zero = x_pos[zero_mask]

            ax.vlines(x_zero, 0.0, y_stub, colors=color, linewidth=1.4, alpha=0.30, zorder=4)
            ax.hlines(
                [y_stub] * len(x_zero),
                x_zero - bar_width * 0.20,
                x_zero + bar_width * 0.20,
                colors='black',
                linewidth=0.8,
                zorder=4
            )
            for xz in x_zero:
                ax.annotate(
                    "0",
                    xy=(xz, y_stub),
                    xytext=(0, 2 if direction > 0 else -2),
                    textcoords="offset points",
                    ha='center',
                    va='bottom' if direction > 0 else 'top',
                    fontsize=6,
                    color='black',
                    zorder=5
                )

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=10)
    ax.set_xticks(index)
    ax.set_xticklabels(PlotConfig.SCENARIOS, fontweight='bold', fontsize=9)
    ax.set_xlabel("Scenario", fontweight='bold', fontsize=10)
    ax.set_yscale('symlog', linthresh=linthresh)
    ax.grid(True, which="major", ls="--", alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)

    if show_legend:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=3, fontsize=8, frameon=False)


# ==========================================
# 4. 主绘图入口
# ==========================================
def plot_hist(cfg=None):
    """
    绘制分组柱状图（Fig.5）：HP Distance、NHP Distance、HP Violation、NHP Violation（2×2）。

    Args:
        cfg: Hydra/OmegaConf 配置对象。支持以下字段：
             - plot_hist.data_root / plot_hist.json_data_dir: 数据根目录
             - plot_hist.output_dir / plot_hist.save_dir: 输出目录
             - plot_hist.model_keys: 方法列表（覆盖默认8个）
             - plot_hist.n_seeds: seed 数量（默认 5）
    """
    data_root = PlotConfig.JSON_DIR
    output_dir = PlotConfig.SAVE_DIR
    model_keys = list(PlotConfig.MODELS_KEYS)
    n_seeds = PlotConfig.N_SEEDS

    if cfg is not None:
        hist_cfg = cfg.get('plot_hist', cfg)
        # 兼容两套命名：data_root / json_data_dir
        if hasattr(hist_cfg, 'data_root') and hist_cfg.data_root:
            data_root = str(hist_cfg.data_root)
        elif hasattr(hist_cfg, 'json_data_dir') and hist_cfg.json_data_dir:
            data_root = str(hist_cfg.json_data_dir)
        # 兼容两套命名：output_dir / save_dir
        if hasattr(hist_cfg, 'output_dir') and hist_cfg.output_dir:
            output_dir = str(hist_cfg.output_dir)
        elif hasattr(hist_cfg, 'save_dir') and hist_cfg.save_dir:
            output_dir = str(hist_cfg.save_dir)
        if hasattr(hist_cfg, 'model_keys') and hist_cfg.model_keys:
            model_keys = list(hist_cfg.model_keys)
        if hasattr(hist_cfg, 'n_seeds') and hist_cfg.n_seeds:
            n_seeds = int(hist_cfg.n_seeds)

    if _HAS_STYLE:
        setup_style()

    data = load_real_data(data_root=data_root, model_keys=model_keys, n_seeds=n_seeds)

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")

    # 2×2 布局：Row1 = Distance, Row2 = Violation Count
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)

    # Row 1: Distance（HP / NHP）
    plot_grouped_bar_chart(
        data['hp_dist']['mean'], data['hp_dist']['std'],
        axes[0, 0], "(a) HP Distance", "Mean HP Intent Distance (per step)",
        show_legend=True, model_keys=model_keys
    )
    plot_grouped_bar_chart(
        data['nhp_dist']['mean'], data['nhp_dist']['std'],
        axes[0, 1], "(b) NHP Distance", "Mean NHP Intent Distance (per step)",
        model_keys=model_keys
    )

    # Row 2: Violation Count（HP / NHP）
    plot_grouped_bar_chart(
        data['hp_viol']['mean'], data['hp_viol']['std'],
        axes[1, 0], "(c) HP Violations", "Mean HP Violation Count (per episode)",
        model_keys=model_keys
    )
    plot_grouped_bar_chart(
        data['nhp_viol']['mean'], data['nhp_viol']['std'],
        axes[1, 1], "(d) NHP Violations", "Mean NHP Violation Count (per episode)",
        model_keys=model_keys
    )

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        save_path = os.path.join(output_dir, f"final_bar_chart_v2.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_hist()