"""
全局绘图样式配置文件（按「画图指南」Part D 规范）。

所有绘图脚本应在文件顶部调用：
    from src.basic_apis.metrics_utils.plot_style import METHOD_STYLES, setup_style, get_style
    setup_style()
"""
import matplotlib.pyplot as plt

# ─── 方法样式字典 ──────────────────────────────────────────────────────────────
# 键名与 data/channel_generality/<key>/ 目录名一致
METHOD_STYLES = {
    'ppo_baseline': {
        'color': '#d62728', 'ls': '-',   'marker': 'o', 'hatch': '//',
        'label': 'PPO-MLP [42]',
    },
    'ppo_lagrangian': {
        'color': '#e377c2', 'ls': '-',   'marker': '^', 'hatch': '||',
        'label': 'Lagrangian PPO [20]',
    },
    'transfer_ppo': {
        'color': '#8c564b', 'ls': '-',   'marker': 'D', 'hatch': '..',
        'label': 'Transfer PPO [25]',
    },
    'ppo_multi': {
        'color': '#bcbd22', 'ls': '--',  'marker': '+', 'hatch': '--',
        'label': 'Multi-Scen. PPO',
    },
    'ppo_ha_weighted': {
        'color': '#ff7f0e', 'ls': '--',  'marker': 's', 'hatch': '\\\\',
        'label': 'PPO Discrete (Teacher)',
    },
    'dt_baseline': {
        'color': '#2ca02c', 'ls': '-.',  'marker': 'v', 'hatch': '--',
        'label': 'DT Baseline',
    },
    'dt': {
        'color': '#00008B', 'ls': ':',   'marker': '*', 'hatch': '',
        'label': 'IDT (Full Action)',
    },
    'dt_fixed_rr': {
        'color': '#1f77b4', 'ls': '-',   'marker': '*', 'hatch': '',
        'label': 'IDT + Fixed RR',
    },
    'dt_no_rescue': {
        'color': '#d62728', 'ls': '--', 'marker': 'x', 'hatch': '',
        'label': 'IDT w/o Rescue',
    },
    'dt_fixed_order': {
        'color': '#2ca02c', 'ls': '-.', 'marker': '^', 'hatch': '',
        'label': 'IDT Fixed Order',
    },
    'dt_reward_only': {
        'color': '#9467bd', 'ls': '-', 'marker': 'D', 'hatch': '',
        'label': 'IDT Reward-only',
    },
    # 参考线（理想上界）
    '_ideal': {
        'color': '#1f77b4', 'ls': '--',  'marker': None, 'hatch': '',
        'label': 'Ideal Bound',
    },
}

# 默认方法顺序（用于表格/柱状图统一排序）
METHOD_ORDER = [
    "ppo_baseline",
    "ppo_lagrangian",
    "transfer_ppo",
    "transfer_ppo_0shot",
    "ppo_ha_weighted",
    "dt_baseline",
    "dt",
    "dt_fixed_rr",
    # 以下主要用于消融/敏感性
    "dt_no_rescue",
    "dt_fixed_order",
    "dt_reward_only",
    "dt_ctx5",
    "dt_ctx10",
    "dt_ctx20",
    "dt_ctx30",
    "dt_ctx50",
    "dt_ds10pct",
    "dt_ds30pct",
    "dt_ds50pct",
    "dt_ds100pct",
]

# 场景标签映射
SCENARIO_LABELS = {
    0: "$S_0$",
    1: "$S_1$",
    2: "$S_2$",
    3: "$S_3$",
    4: "$S_4$",
    5: "$S_A$",
    6: "$S_B$",
    7: "$S_C$",
    8: "$S_D$",
    9: "$S_E$",
}


def get_style(method_key: str) -> dict:
    """返回方法对应的样式字典，未知方法返回默认灰色样式。"""
    return METHOD_STYLES.get(method_key, {
        'color': '#7f7f7f', 'ls': '-', 'marker': 'o', 'hatch': '',
        'label': method_key,
    })


def setup_style(font_size=None):
    """应用全局 matplotlib 样式（IEEE 论文风格）。"""
    try:
        import seaborn as sns
        sns.set_context("paper", font_scale=1.3)
        sns.set_style("whitegrid")
    except ImportError:
        pass

    plt.rcParams.update({
        # 字体
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':   'stix',
        # 分辨率
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.format':     'pdf',
        # 线条
        'lines.linewidth':    2.0,
        'lines.markersize':   6,
        # 轴
        'axes.labelweight':   'bold',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.alpha':         0.35,
        'grid.linestyle':     '--',
        # 图例
        'legend.frameon':     False,
        'legend.fontsize':    9,
        # 字号（单栏 3.5 inch 场景）
        'axes.labelsize':     10,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'axes.titlesize':     12,
    })
    if font_size is not None:
        plt.rcParams.update({
            'axes.labelsize': font_size,
            'xtick.labelsize': max(font_size - 2, 6),
            'ytick.labelsize': max(font_size - 2, 6),
            'axes.titlesize': font_size + 2,
            'legend.fontsize': max(font_size - 1, 6),
        })
