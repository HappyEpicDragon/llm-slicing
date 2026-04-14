import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import json
import os
import math
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# def plot_hist(cfg):
#     """
#     绘制柱状图：展示不同 Checkpoint 下 HP 和 NHP 的平均指标值
#     顺序：DT (0,0) -> CKP_0...CKP_49 -> Best_Model
#     """
#     hist_cfg = cfg.plot_hist
#     save_dir = hist_cfg.save_dir
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 1. 定义数据源目录
#     base_dir = hist_cfg.json_data_dir
#
#     # 2. 定义 Checkpoint 的扫描顺序
#     # (1) DT (手动添加，作为基准)
#     # (2) CKP_0 到 CKP_49 (按数字顺序尝试读取)
#     # (3) Best_Model (最后)
#     folder_sequence = ['DT']
#     # 生成 CKP_0 到 CKP_49 的列表
#     folder_sequence.extend([f'CKP_{i}' for i in range(50)])
#     folder_sequence.append('Best_Model')
#
#     # 引用 MetricPlotter 中的配色 (保持风格一致)
#     colors = ['#1f77b4', '#d62728']  # 0: Blue (NHP/Base), 1: Red (HP/High)
#
#     for metric in hist_cfg.metrics:
#         print(f"Processing Hist Metric: {metric} ...")
#
#         # 用于绘图的数据容器
#         valid_labels = []  # X轴标签
#         hp_means = []  # HP 高度
#         nhp_means = []  # NHP 高度
#
#         for folder_name in folder_sequence:
#             # --- 特殊情况：DT ---
#             if folder_name == 'DT':
#                 valid_labels.append('DT')
#                 hp_means.append(0.0)
#                 nhp_means.append(0.0)
#                 continue
#
#             # --- 常规情况：读取文件 ---
#             file_path = os.path.join(base_dir, folder_name, f'{metric}.json')
#
#             # 如果文件存在，则读取并计算均值；如果中间某些CKP不存在，则跳过
#             if os.path.exists(file_path):
#                 try:
#                     data = load_json_to_dict(file_path)
#
#                     # 计算平均值 (Mean)
#                     val_hp = np.mean(data['hp']) if data['hp'] else 0.0
#                     val_nhp = np.mean(data['nhp']) if data['nhp'] else 0.0
#
#                     # 简化 X 轴标签：CKP_14 -> 14, Best_Model -> Best
#                     if folder_name.startswith('CKP_'):
#                         short_label = folder_name.split('_')[1]
#                     elif folder_name == 'Best_Model':
#                         short_label = 'Best'
#                     else:
#                         short_label = folder_name
#
#                     valid_labels.append(short_label)
#                     hp_means.append(val_hp)
#                     nhp_means.append(val_nhp)
#                 except Exception as e:
#                     print(f"[Warn] Error loading {file_path}: {e}")
#             # else: 文件不存在则直接跳过，不占位
#
#         # --- 开始绘图 ---
#         if not valid_labels:
#             print(f"[Warn] No valid data found for {metric}")
#             continue
#
#         fig, ax = plt.subplots(figsize=(12, 6))  # 宽一点，适应较多的柱子
#
#         x = np.arange(len(valid_labels))
#         width = 0.35  # 柱子宽度
#
#         # 绘制柱子
#         # NHP 放左边 (或底层逻辑)，HP 放右边
#         rects1 = ax.bar(x - width / 2, nhp_means, width, label='NHP', color=colors[0], alpha=0.8)
#         rects2 = ax.bar(x + width / 2, hp_means, width, label='HP', color=colors[1], alpha=0.9)
#
#         # --- 样式美化 (复用 MetricPlotter 的 Tufte 风格) ---
#         ax.set_ylabel(f'Mean {metric}', fontsize=12, fontweight='bold')
#         ax.set_title(f'Performance over Checkpoints: {metric}', fontsize=14, pad=15)
#
#         # 设置 X 轴
#         ax.set_xticks(x)
#         ax.set_xticklabels(valid_labels, rotation=45 if len(valid_labels) > 15 else 0)
#         ax.set_xlabel('Checkpoints', fontsize=12, fontweight='bold')
#
#         # 图例
#         ax.legend(loc='upper left', frameon=False, fontsize=11)
#
#         # 网格与边框
#         ax.grid(True, axis='y', color='#dddddd', linestyle='--', alpha=0.8, zorder=0)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_linewidth(1.5)
#         ax.spines['bottom'].set_linewidth(1.5)
#
#         # 加一条 0 线
#         ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
#
#         # 保存
#         plt.tight_layout()
#         save_path = os.path.join(save_dir, f'hist_{metric}.png')
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         print(f"Hist saved to {save_path}")

def plot_hist(cfg):
    """
    修复版演进图：
    1. 修正角色：DT 是 Proposed (Target), PPO 是 Baseline (Fine-tuning)
    2. 修复布局：使用固定坐标的 inset_axes 防止图片畸变
    """
    hist_cfg = cfg.plot_hist
    save_dir = hist_cfg.save_dir
    base_dir = hist_cfg.json_data_dir
    os.makedirs(save_dir, exist_ok=True)

    # ==========================
    # 1. 配色与样式定义 (TWC风格)
    # ==========================
    # Proposed (DT) 使用醒目的红色
    style_dt = {
        'color': '#d62728',  # 砖红色
        'linestyle': '--',
        'linewidth': 2.0,
        'label': 'Proposed (Decision Transformer)'
    }

    # Baseline (PPO) 使用蓝色
    style_ppo = {
        'color': '#1f77b4',  # 经典蓝
        'linestyle': '-',
        'linewidth': 2.0,
        'label': 'Baseline (PPO Fine-tuning)'
    }

    for metric in hist_cfg.metrics:
        print(f"Processing Fixed Evolution Plot: {metric} ...")

        # --- 数据容器 ---
        ckp_indices = []
        ppo_mean, ppo_std = [], []
        dt_mean, dt_std = 0.0, 0.0

        # A. 读取 Proposed (DT) - 作为基准线
        dt_file = os.path.join(base_dir, 'DT', f'{metric}.json')
        if os.path.exists(dt_file):
            d = load_json_to_dict(dt_file)
            # 这里我们要取 HP 和 NHP 分别画
            # 为了简化逻辑，这里先读取数据，绘图时分上下两图
            dt_hp = np.mean(d['hp']) if d['hp'] else 0.0
            dt_nhp = np.mean(d['nhp']) if d['nhp'] else 0.0
        else:
            print("[Warn] DT file not found, skipping...")
            continue

        # B. 读取 Baseline (PPO) - 随 Checkpoint 变化
        ppo_hp_series, ppo_nhp_series = [], []
        ppo_hp_std_series, ppo_nhp_std_series = [], []

        for i in range(50):
            fpath = os.path.join(base_dir, f'CKP_{i}', f'{metric}.json')
            if os.path.exists(fpath):
                try:
                    d = load_json_to_dict(fpath)
                    ckp_indices.append(i)

                    # HP
                    ppo_hp_series.append(np.mean(d['hp']) if d['hp'] else 0.0)
                    ppo_hp_std_series.append(np.std(d['hp']) if d['hp'] else 0.0)
                    # NHP
                    ppo_nhp_series.append(np.mean(d['nhp']) if d['nhp'] else 0.0)
                    ppo_nhp_std_series.append(np.std(d['nhp']) if d['nhp'] else 0.0)
                except:
                    pass

        if not ckp_indices: continue

        # 转 Numpy
        ppo_hp_series = np.array(ppo_hp_series)
        ppo_hp_std_series = np.array(ppo_hp_std_series)
        ppo_nhp_series = np.array(ppo_nhp_series)
        ppo_nhp_std_series = np.array(ppo_nhp_std_series)

        # ==========================
        # 2. 绘图 (上下子图)
        # ==========================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0.15) # 调整子图间距

        # --- 上子图: HP (High Priority) ---
        # 1. 画 DT (Target Line)
        ax1.axhline(dt_hp, **style_dt)
        # 2. 画 PPO (Curve)
        ax1.plot(ckp_indices, ppo_hp_series, **style_ppo)
        # 阴影
        ax1.fill_between(ckp_indices,
                         ppo_hp_series - ppo_hp_std_series,
                         ppo_hp_series + ppo_hp_std_series,
                         color=style_ppo['color'], alpha=0.15)

        ax1.set_ylabel(f"HP {metric}", fontweight='bold', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # --- 下子图: NHP (Non-High Priority) ---
        # 1. 画 DT
        ax2.axhline(dt_nhp, **style_dt)
        # 2. 画 PPO
        ax2.plot(ckp_indices, ppo_nhp_series, **style_ppo)
        ax2.fill_between(ckp_indices,
                         ppo_nhp_series - ppo_nhp_std_series,
                         ppo_nhp_series + ppo_nhp_std_series,
                         color=style_ppo['color'], alpha=0.15)

        ax2.set_ylabel(f"NHP {metric}", fontweight='bold', fontsize=12)
        ax2.set_xlabel("Fine-tuning Checkpoints (Training Progress)", fontweight='bold', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # ==========================
        # 3. 添加放大镜 (Inset) - 专治"看起来像横线"
        # ==========================
        # 我们在 NHP 图 (ax2) 中添加放大镜，只放大 PPO 的波动区域

        # 自动计算 PPO 的波动范围
        y_min = np.min(ppo_nhp_series)
        y_max = np.max(ppo_nhp_series)
        y_range = y_max - y_min

        # 如果波动太小，强行给一点 buffer，防止上下限一样
        if y_range < 0.001:
            padding = 0.01
        else:
            padding = y_range * 0.5

        view_ylim = (y_min - padding, y_max + padding)

        # 创建 Inset Axes (固定位置：宽40%，高35%，位于右侧中间)
        # loc='center right' 配合 bbox_to_anchor 微调位置
        axins = inset_axes(ax2, width="40%", height="35%", loc='center right', borderpad=1)

        # 在小图里只画 PPO (为了看清楚它的趋势)
        axins.plot(ckp_indices, ppo_nhp_series, color=style_ppo['color'], linewidth=1.5)
        axins.fill_between(ckp_indices,
                           ppo_nhp_series - ppo_nhp_std_series,
                           ppo_nhp_series + ppo_nhp_std_series,
                           color=style_ppo['color'], alpha=0.1)

        # 设置小图视野
        axins.set_xlim(0, 49)
        axins.set_ylim(view_ylim)

        # 小图样式
        axins.set_title("Zoom: PPO Trend", fontsize=9)
        axins.grid(True, linestyle=':', alpha=0.5)
        axins.tick_params(labelsize=8)

        # 画连线 (Mark Inset) - 连接大图和小图
        # loc1, loc2 控制连线的角，2=左上, 4=右下
        mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle=':')

        # ==========================
        # 4. 图例与保存
        # ==========================
        # 收集图例
        handles, labels = ax2.get_legend_handles_labels()
        # 放在最下面
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=11)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15) # 留出底部给图例

        save_path = os.path.join(save_dir, f'evolution_fixed_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Fixed plot saved to {save_path}")


def plot(cfg):
    plot_cfg = cfg
    scenario_list = plot_cfg.get('plot_scenario_list', None)

    for metric in plot_cfg.metrics:
        # 实例化绘图器，初始化时增加分辨率
        plotter = MetricPlotter(figsize=(10, 6))

        for baseline in plot_cfg.baselines:
            # --- 数据读取逻辑 (保持不变) ---
            if scenario_list is not None and len(scenario_list) > 0:
                merged_data = {'hp': [], 'nhp': []}
                for sc_idx in scenario_list:
                    file_path = os.path.join(
                        plot_cfg.json_data_base_dir, baseline,
                        'metric_json', f'scenario_{sc_idx}', f'{metric}.json'
                    )
                    try:
                        part_data = load_json_to_dict(file_path)
                        merged_data['hp'].extend(part_data['hp'])
                        merged_data['nhp'].extend(part_data['nhp'])
                    except FileNotFoundError:
                        print(f"[Warn] Skip: {file_path}")
                data = merged_data
            else:
                file_path = os.path.join(plot_cfg.json_data_base_dir, baseline, 'metric_json', f'{metric}.json')
                data = load_json_to_dict(file_path)
            # ---------------------------

            plotter.add_baseline(data, baseline)

        plotter.set_plot(
            xlabel='Channel Data Episode',
            ylabel=metric,  # 这里建议在外部通过字典映射成更好看的英文名
            # 图例放在下面，自适应列数
            legend_loc='outside_bottom'
        )
        save_path = os.path.join(plot_cfg.save_dir, f'{metric}.png')
        plotter.save(save_path)


# class MetricPlotter:
#     """
#     Metric绘图管理类 - 论文优化版
#     """
#
#     def __init__(self, enable_hp=True, figsize=(10, 6)):
#         self.enable_hp = enable_hp
#         self.fig, self.ax = plt.subplots(figsize=figsize)
#         self.baselines = []
#
#         # 优化后的配色：更深沉、对比度更高的颜色，适合论文阅读
#         self.colors = [
#             '#1f77b4',  # 经典蓝
#             '#d62728',  # 砖红
#             '#2ca02c',  # 深绿
#             '#ff7f0e',  # 橙色
#             '#9467bd',  # 紫色
#             '#8c564b',  # 棕色
#         ]
#
#         # 标记样式
#         self.markers = ['o', 's', '^', 'D', 'v', '*']
#
#         self.current_idx = 0
#
#     def add_baseline(self, data_dict, label_name):
#         """
#         添加baseline数据，应用视觉分层逻辑
#         """
#         color = self.colors[self.current_idx % len(self.colors)]
#         marker = self.markers[self.current_idx % len(self.markers)]
#
#         # 准备数据
#         data_hp = np.array(data_dict['hp'])
#         data_nhp = np.array(data_dict['nhp'])
#
#         # 统一长度索引
#         data_len = len(data_hp)
#         x = np.arange(data_len)
#
#         # 1. 稀疏化标记 (Sparse Markers)
#         # 逻辑：总长度的 8% ~ 10% 作为一个间隔，保证整张图只有 10-12 个标记点
#         mark_step = max(1, int(data_len / 10))
#
#         if self.enable_hp:
#             # --- 绘制 NHP (背景层) ---
#             # 策略：细线、虚线、高透明度、Z轴靠后
#             self.ax.plot(x, np.cumsum(data_nhp),
#                          marker=marker, markevery=mark_step, markersize=5,
#                          linewidth=1, linestyle='--', alpha=0.5,  # Alpha 0.5 让它变淡
#                          color=color, label=f'{label_name} (NHP)',
#                          zorder=2)  # zorder 越小越靠后
#
#             # --- 绘制 HP (前景层/强调层) ---
#             # 策略：粗线、实线、低透明度、Z轴靠前
#             self.ax.plot(x, np.cumsum(data_hp),
#                          marker=marker, markevery=mark_step, markersize=7,
#                          linewidth=1, linestyle='-', alpha=0.9,  # Alpha 0.9 突出显示
#                          color=color, label=f'{label_name} (HP)',
#                          zorder=3)  # zorder 大于 NHP，保证压在上面
#         else:
#             # 仅绘制 NHP 模式
#             self.ax.plot(x, data_nhp, marker=marker, markevery=mark_step,
#                          linewidth=2.5, color=color, label=label_name)
#
#         self.current_idx += 1
#
#     def set_plot(self, xlabel='Time Step', ylabel='Metric Value', title=None,
#                  legend_loc='best', show_target=True, grid=True):
#
#         # 1. 网格优化：使用灰色虚线，并置于最底层
#         if grid:
#             self.ax.grid(True, which='major', color='#dddddd', linestyle='--', alpha=0.8, zorder=0)
#
#         # 2. 零刻度线
#         if show_target:
#             self.ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=1)
#
#         # 3. 坐标轴美化
#         self.ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', labelpad=10)
#         self.ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
#         self.ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
#
#         # 移除顶部和右侧的边框（Tufte 风格，减少墨水比）
#         self.ax.spines['top'].set_visible(False)
#         self.ax.spines['right'].set_visible(False)
#         self.ax.spines['left'].set_linewidth(1.5)
#         self.ax.spines['bottom'].set_linewidth(1.5)
#
#         if title:
#             self.ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
#
#         # 4. 图例优化 (Legend)
#         if legend_loc == 'outside_bottom':
#             # 自动计算列数：如果有 4 个 baseline * 2 = 8 条线，分 4 列展示比较好看
#             num_lines = len(self.ax.get_legend_handles_labels()[1])
#             ncols = min(4, math.ceil(num_lines / 2))  # 尝试控制行数不超过2-3行
#
#             self.ax.legend(
#                 loc='upper center',
#                 bbox_to_anchor=(0.5, -0.15),  # 放在x轴下方
#                 ncol=ncols,
#                 fontsize=12,
#                 frameon=False,  # 去掉图例边框，更简洁
#                 columnspacing=1.5
#             )
#         else:
#             self.ax.legend(loc=legend_loc, fontsize=11, framealpha=0.9)
#
#     def save(self, filename='metric_plot.png', dpi=300):
#         # 使用 bbox_inches='tight' 确保外置图例不会被裁剪
#         plt.tight_layout()
#         self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
#         print(f"图表已保存为 '{filename}'")
#
#     def close(self):
#         plt.close(self.fig)

class MetricPlotter:
    """
    Metric绘图管理类 - 论文优化版 (Moving Average 风格)
    """

    def __init__(self, enable_hp=True, figsize=(10, 6)):
        self.enable_hp = enable_hp
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # 优化后的配色：更适合学术发表
        self.colors = [
            '#e41a1c',  # 红色 (Proposed/Strong)
            '#377eb8',  # 蓝色 (Baseline/Weak)
            '#4daf4a',  # 绿色
            '#984ea3',  # 紫色
            '#ff7f00',  # 橙色
            '#a65628',  # 棕色
        ]

        # 标记样式 (虽然滑动平均图通常不加标记，但在稀疏点加一点可以区分曲线)
        self.markers = ['o', 's', '^', 'D', 'v', '*']

        self.current_idx = 0

    def _smooth_data(self, data, window_size=5):
        """
        内部辅助函数：计算滑动平均
        window_size: 窗口大小，数据越抖，窗口应该设得越大 (建议 5-20)
        """
        series = pd.Series(data)
        # min_periods=1 保证开头数据不会变成 NaN
        return series.rolling(window=window_size, min_periods=1).mean().values

    def add_baseline(self, data_dict, label_name):
        """
        添加baseline数据 - 改为 Moving Average 风格
        """
        color = self.colors[self.current_idx % len(self.colors)]
        marker = self.markers[self.current_idx % len(self.markers)]

        # 准备数据 (去掉 cumsum!)
        raw_hp = np.array(data_dict['hp'])
        raw_nhp = np.array(data_dict['nhp'])

        raw_hp = np.cumsum(raw_hp)
        raw_nhp = np.cumsum(raw_nhp)

        # 定义滑动窗口大小 (根据你的数据长度调整，通常 10-20 效果最好)
        # 如果你的 Episode 很少（比如只有 20 个），窗口设小一点（比如 3）
        data_len = len(raw_hp)
        window = max(3, int(data_len * 0.1))

        # 计算平滑曲线
        smooth_hp = self._smooth_data(raw_hp, window_size=window)
        smooth_nhp = self._smooth_data(raw_nhp, window_size=window)

        x = np.arange(data_len)

        # 稀疏标记：只在 10% 的点上画标记，避免拥挤
        mark_step = max(1, int(data_len / 8))

        if self.enable_hp:
            # === 1. 绘制 NHP (虚线，背景层) ===
            # 画原始数据的阴影范围 (可选，如果太乱可以注释掉 ax.fill_between)
            # self.ax.fill_between(x, raw_nhp, alpha=0.1, color=color)

            self.ax.plot(x, smooth_nhp,
                         linestyle='--', linewidth=1.5, alpha=0.6,
                         marker=marker, markevery=mark_step, markersize=5,
                         color=color, label=f'{label_name} (NHP)',
                         zorder=2)

            # === 2. 绘制 HP (实线，强调层) ===
            # 关键：画出原始数据的浅色阴影，体现“方差”或“瞬时波动”
            # 如果原始数据 raw_hp 波动很大，这个阴影会很有信息量
            self.ax.plot(x, raw_hp, color=color, alpha=0.15, linewidth=1, zorder=1)

            # 画平滑后的主曲线
            self.ax.plot(x, smooth_hp,
                         linestyle='-', linewidth=2.5, alpha=1.0,
                         marker=marker, markevery=mark_step, markersize=7,
                         color=color, label=f'{label_name} (HP)',
                         zorder=3)
        else:
            # 仅绘制 NHP 模式
            self.ax.plot(x, raw_nhp, color=color, alpha=0.15, linewidth=1)
            self.ax.plot(x, smooth_nhp, linestyle='-', linewidth=2.5,
                         marker=marker, markevery=mark_step,
                         color=color, label=label_name)

        self.current_idx += 1

    def set_plot(self, xlabel='Episode', ylabel='Metric Value', title=None,
                 legend_loc='outside_bottom', show_target=True, grid=True):

        # ... (以下部分基本保持不变，保留你的 Tufte 风格) ...

        # 1. 网格优化
        if grid:
            self.ax.grid(True, which='major', color='#dddddd', linestyle='--', alpha=0.8, zorder=0)

        # 2. 零刻度线 (Target Line)
        if show_target:
            self.ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=1)

        # 3. 坐标轴美化
        self.ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', labelpad=10)
        self.ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
        self.ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)

        # 移除顶部和右侧边框
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_linewidth(1.5)
        self.ax.spines['bottom'].set_linewidth(1.5)

        if title:
            self.ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

        # 4. 图例优化
        if legend_loc == 'outside_bottom':
            num_lines = len(self.ax.get_legend_handles_labels()[1])
            ncols = min(4, math.ceil(num_lines / 2))
            self.ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=ncols,
                fontsize=12,
                frameon=False,
                columnspacing=1.5
            )
        else:
            self.ax.legend(loc=legend_loc, fontsize=11, framealpha=0.9)

    def save(self, filename='metric_plot.png', dpi=300):
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to '{filename}'")

    def close(self):
        plt.close(self.fig)


# load_json_to_dict 函数保持不变
def load_json_to_dict(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data