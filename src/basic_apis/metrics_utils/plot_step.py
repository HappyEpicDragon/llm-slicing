import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import deque

from src.basic_apis.metrics_utils.plot_style import get_style, setup_style


# ==========================================
# 1. 全局配置
# ==========================================
class Config:
    # --- 核心开关 ---
    # 选项: "distance" (归一化距离) 或 "violation" (归一化违约数)
    TARGET_METRIC = "distance"

    # 路径配置
    data_base_dir = "./data/channel_generality"
    save_json_dir = "./data/metric_json_step_level"
    save_plot_dir = "./plots/step_level_plots"

    # 算法与场景
    baselines = ["ppo_baseline", "ppo_ha_weighted", "dt_baseline", "dt"]
    test_scenario_list = [9]
    test_episode_begin = 60
    cross_scenario_skip_episode = 100
    episodes_per_scenario = 20

    # 导入工具函数
    from src.basic_apis.ppo.utils import calculate_slice_ue_obs, intent_drift_calc


# ==========================================
# 2. 核心计算逻辑 (Distance & Violation)
# ==========================================

# 通用入口：根据 Config 选择计算哪种指标
def calc_step_metric(data_metrics, priority=False):
    if Config.TARGET_METRIC == "distance":
        return calc_step_level_distance(data_metrics, priority)
    elif Config.TARGET_METRIC == "violation":
        return calc_step_level_violation(data_metrics, priority)
    else:
        raise ValueError(f"Unknown metric: {Config.TARGET_METRIC}")


def calc_step_level_distance(data_metrics, priority=False, max_number_ues_slice=5):
    """ 计算 Step 级别的 Distance """
    TOTAL_STEPS = data_metrics["slice_ue_assoc"].shape[0]
    intent_drift = get_intent_drift_local(data_metrics)

    metric_per_step = np.zeros(TOTAL_STEPS)
    active_slices_per_step = np.zeros(TOTAL_STEPS)

    for step_idx in range(TOTAL_STEPS):
        intent_array = []
        active_count = 0
        num_slices = data_metrics["slice_ue_assoc"][step_idx].shape[0]

        for slice_idx in range(num_slices):
            # 活跃检查
            is_active = (data_metrics["basestation_slice_assoc"][step_idx][0, slice_idx] != 0)
            if not is_active: continue
            # 优先级检查
            if priority:
                slice_prio = data_metrics["slice_req"][step_idx][f"slice_{slice_idx}"]["priority"]
                if slice_prio == 0: continue

            active_count += 1
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][slice_idx].nonzero()[0]

            # 计算 Drift
            _, intent_drift_slice = Config.calculate_slice_ue_obs(
                max_number_ues_slice, intent_drift[step_idx], slice_idx, slice_ues, data_metrics["slice_req"][step_idx],
            )
            # 过滤非违约项 (只保留负数)
            intent_drift_slice = np.delete(
                intent_drift_slice, np.logical_or(np.isclose(intent_drift_slice, -2), intent_drift_slice >= 0)
            )
            min_intent = np.min(intent_drift_slice) if intent_drift_slice.shape[0] > 0 else 0
            intent_array.append(min_intent)

        metric_per_step[step_idx] = np.sum(intent_array) if intent_array else 0
        active_slices_per_step[step_idx] = active_count

    return metric_per_step, active_slices_per_step


def calc_step_level_violation(data_metrics, priority=False, max_number_ues_slice=5):
    """ 计算 Step 级别的 Violation Count """
    TOTAL_STEPS = data_metrics["slice_ue_assoc"].shape[0]
    intent_drift = get_intent_drift_local(data_metrics)

    metric_per_step = np.zeros(TOTAL_STEPS)
    active_slices_per_step = np.zeros(TOTAL_STEPS)

    for step_idx in range(TOTAL_STEPS):
        step_violations = 0
        active_count = 0
        num_slices = data_metrics["slice_ue_assoc"][step_idx].shape[0]

        for slice_idx in range(num_slices):
            # 活跃检查
            is_active = (data_metrics["basestation_slice_assoc"][step_idx][0, slice_idx] != 0)
            if not is_active: continue
            # 优先级检查
            if priority:
                slice_prio = data_metrics["slice_req"][step_idx][f"slice_{slice_idx}"]["priority"]
                if slice_prio == 0: continue

            active_count += 1
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][slice_idx].nonzero()[0]

            # 计算 Drift
            _, intent_drift_slice = Config.calculate_slice_ue_obs(
                max_number_ues_slice, intent_drift[step_idx], slice_idx, slice_ues, data_metrics["slice_req"][step_idx],
            )

            # 判断违约: 只要有一个指标 < 0 即视为该切片违约
            # 注意: -2 代表无效值，需要排除
            valid_drifts = intent_drift_slice[~np.isclose(intent_drift_slice, -2)]
            is_violated = np.any(valid_drifts < 0)

            if is_violated:
                step_violations += 1

        metric_per_step[step_idx] = step_violations
        active_slices_per_step[step_idx] = active_count

    return metric_per_step, active_slices_per_step


def get_intent_drift_local(data_metrics, max_number_ues_slice=5, intent_overfulfillment_rate=0.2):
    TOTAL_STEPS = data_metrics["slice_ue_assoc"].shape[0]
    last_unformatted_obs = deque(maxlen=10)
    number_slices = data_metrics["slice_ue_assoc"].shape[1]
    number_ues_slice = int(data_metrics["slice_ue_assoc"].shape[2] / number_slices)
    intent_drift = np.zeros((TOTAL_STEPS, number_slices, number_ues_slice, 3))

    for step_idx in range(TOTAL_STEPS):
        dict_info = {
            "pkt_effective_thr": data_metrics["pkt_effective_thr"][step_idx],
            "slice_req": data_metrics["slice_req"][step_idx],
            "buffer_occupancies": data_metrics["buffer_occupancies"][step_idx],
            "buffer_latencies": data_metrics["buffer_latencies"][step_idx],
            "slice_ue_assoc": data_metrics["slice_ue_assoc"][step_idx],
            "dropped_pkts": data_metrics["dropped_pkts"][step_idx],
        }
        last_unformatted_obs.appendleft(dict_info)
        intent_drift[step_idx, :, :, :] = Config.intent_drift_calc(
            last_unformatted_obs, max_number_ues_slice, intent_overfulfillment_rate, True,
        )
    return intent_drift


def process_data():
    os.makedirs(Config.save_json_dir, exist_ok=True)
    metric_name = Config.TARGET_METRIC
    print(f">>> 开始处理 Step 级别数据: {metric_name} ...")

    for baseline in Config.baselines:
        all_steps_nhp, all_steps_hp = [], []

        for scenario_idx in Config.test_scenario_list:
            start_ep = Config.test_episode_begin + Config.cross_scenario_skip_episode * scenario_idx
            end_ep = start_ep + Config.episodes_per_scenario

            for episode_idx in tqdm(range(start_ep, end_ep), desc=f"[{baseline}] Scen {scenario_idx}"):
                raw_path = os.path.join(Config.data_base_dir, baseline, "metric_raw", f"scenario_{scenario_idx}",
                                        f"ep_{episode_idx}.npz")
                if not os.path.exists(raw_path):
                    raw_path = os.path.join(Config.data_base_dir, baseline, "metric_raw", baseline,
                                            f"ep_{episode_idx}.npz")
                    if not os.path.exists(raw_path): continue

                data = np.load(raw_path, allow_pickle=True)
                data_metrics = {k: data[k] for k in data.files}

                # === NHP 计算 ===
                val_nhp, act_nhp = calc_step_metric(data_metrics, priority=False)
                # 归一化 (Metric / Active Slices)
                norm_nhp = np.divide(val_nhp, act_nhp, out=np.zeros_like(val_nhp), where=act_nhp != 0)

                # === HP 计算 ===
                val_hp, act_hp = calc_step_metric(data_metrics, priority=True)
                norm_hp = np.divide(val_hp, act_hp, out=np.zeros_like(val_hp), where=act_hp != 0)

                all_steps_nhp.extend(norm_nhp)
                all_steps_hp.extend(norm_hp)

        # 保存文件名带上 metric 名字，防止覆盖
        save_path = os.path.join(Config.save_json_dir, f"{baseline}_step_{metric_name}.json")
        with open(save_path, 'w') as f:
            json.dump({"nhp": all_steps_nhp, "hp": all_steps_hp}, f)
        print(f"Saved {baseline} {metric_name} data: {len(all_steps_nhp)} steps")


# ==========================================
# 3. 绘图器 (适配 Label)
# ==========================================
class StepMetricPlotterSubplots:
    def __init__(self, rolling_window=500):
        setup_style()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0.08)
        self.rolling_window = int(rolling_window)

    def add_line(self, data, label, is_hp=True):
        """
        绘制一条时序曲线，支持单次运行和多 seed 跨 seed 统计。

        Args:
            data: 单个数组（向后兼容）或 list of arrays（每个 seed 一条曲线）。
                  多 seed 时计算跨 seed mean ± std confidence band。
            label: 方法标识符（在 colors/linestyles/label_map 中查找）。
            is_hp: True 绘制上子图（HP），False 绘制下子图（NHP）。
        """
        ax = self.ax1 if is_hp else self.ax2
        sty = get_style(label)
        color = sty.get("color", "black")
        ls = sty.get("ls", "-")
        display_label = sty.get("label", label)
        window = max(1, self.rolling_window)

        # ── 判断是否为多 seed 模式 ──────────────────────────────────────
        is_multi_seed = isinstance(data, (list, tuple)) and len(data) > 0 and \
                        hasattr(data[0], '__len__')

        if is_multi_seed:
            # 多 seed：data = [array_seed0, array_seed1, ...]
            # 1. 各 seed 独立 rolling 平滑
            min_len = min(len(s) for s in data)
            smoothed = np.array([
                pd.Series(np.asarray(s)[:min_len])
                .rolling(window=window, min_periods=1).mean().values
                for s in data
            ])  # [n_seeds, T]
            mean_curve = smoothed.mean(axis=0)
            std_curve = smoothed.std(axis=0)
            x = np.arange(min_len)
        else:
            # 单次运行（向后兼容）
            series = pd.Series(np.asarray(data))
            smooth_mean = series.rolling(window=window, min_periods=1).mean().values
            smooth_std = series.rolling(window=window, min_periods=1).std().values
            mean_curve = smooth_mean
            std_curve = smooth_std * 0.5   # 原行为：±0.5σ 内 rolling std
            x = np.arange(len(data))

        alpha_fill = 0.15 if is_hp else 0.1
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=alpha_fill, linewidth=0)

        lw = 2.0 if is_hp else 2.5
        marker = sty.get("marker", None)
        markevery = max(len(x) // 30, 1) if marker is not None else None
        ax.plot(
            x, mean_curve,
            color=color, linestyle=ls, linewidth=lw, label=display_label,
            marker=marker, markevery=markevery, markersize=4, alpha=0.95
        )

    def save(self, path):
        # 动态设置 Y 轴标签
        if Config.TARGET_METRIC == "violation":
            y_label = "Norm. Violation Ratio"
            # 违约数通常是正数，范围 [0, 1]
            # 我们可以稍微限制一下 ylim 让图好看点，或者自动适应
            # self.ax1.set_ylim(-0.05, 1.05)
        else:
            y_label = "Norm. Distance"

        # 上子图
        self.ax1.set_ylabel(f"HP {y_label}", fontsize=12, fontweight='bold')
        self.ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        self.ax1.grid(True, which='major', linestyle='--', alpha=0.4)
        self.ax1.tick_params(labelbottom=False)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)

        # 下子图
        self.ax2.set_ylabel(f"NHP {y_label}", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Simulation Steps", fontsize=14, fontweight='bold')
        self.ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        self.ax2.grid(True, which='major', linestyle='--', alpha=0.4)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)

        # 图例
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.fig.legend(by_label.values(), by_label.keys(),
                        loc='lower center', bbox_to_anchor=(0.5, 0.02),
                        ncol=4, frameon=False, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {path}")


# ==========================================
# 4. 数据加载（新格式：metric_raw/scenario_*/ep_seed*.npz）
# ==========================================

def _load_multi_seed_npz(data_root, method_key, scenario_ids, metric_field,
                          n_seeds=5):
    """
    从 metric_raw/scenario_{id}/ep_seed{N}.npz 加载所有 seed 的 step 级别数据。

    metric_field: npz 中的数组字段名，如 'step_hp_dist', 'step_nhp_dist'。
    若为 'step_hp_viol' / 'step_nhp_viol'，则从 step_*_dist 派生（dist<0 -> 1）。

    返回：
        dict: {scenario_id: list[ndarray]}  每个 scenario 对应 n_seeds 个数组
              若无数据则为空列表
    """
    derive_viol = metric_field in ("step_hp_viol", "step_nhp_viol")

    # Important: check NHP before HP because "nhp" also contains substring "hp".
    # Otherwise step_nhp_* would be mistakenly mapped to step_hp_* and collapse to zeros.
    if "nhp" in metric_field:
        load_field = "step_nhp_dist" if derive_viol else "step_nhp_dist"
    elif "hp" in metric_field:
        load_field = "step_hp_dist" if derive_viol else "step_hp_dist"
    else:
        load_field = metric_field

    result = {}
    for s_id in scenario_ids:
        raw_dir = os.path.join(data_root, method_key, "metric_raw", f"scenario_{s_id}")
        seed_arrays = []
        for seed in range(n_seeds):
            npz_path = os.path.join(raw_dir, f"ep_seed{seed}.npz")
            if os.path.exists(npz_path):
                try:
                    npz = np.load(npz_path)
                    if load_field in npz:
                        arr = npz[load_field].ravel()
                        if derive_viol:
                            arr = (arr < 0).astype(np.float64)
                        seed_arrays.append(arr)
                except Exception as e:
                    print(f"  Warning: Could not load {npz_path}: {e}")
        result[s_id] = seed_arrays
    return result


# ==========================================
# 5. 执行入口
# ==========================================
def plot_step(cfg=None):
    """
    绘制 step-level 时序图（Fig.3/4/12）。

    优先从新格式 metric_raw/scenario_*/ep_seed*.npz 加载多 seed 数据，
    生成 mean ± std confidence band。若 npz 文件不存在，退回到旧 JSON 格式（向后兼容）。

    Args:
        cfg: Hydra/OmegaConf 配置对象。支持以下字段：
             - plot_step.data_root:     数据根目录
             - plot_step.output_dir:    输出目录
             - plot_step.model_keys:    参与绘图的方法列表
             - plot_step.target_metric: 'distance' 或 'violation'（默认 violation）
             - plot_step.scenario_ids:  绘图所用场景列表（默认 [5,6,7,8,9]）
             - plot_step.n_seeds:       seed 数量（默认 5）
             cfg 为 None 时沿用内部硬编码（向后兼容）。
    """
    data_root = Config.data_base_dir
    output_dir = Config.save_plot_dir
    model_keys = list(Config.baselines)
    target_metric = Config.TARGET_METRIC
    scenario_ids = list(Config.test_scenario_list)
    n_seeds = 5
    rolling_window = 500

    if cfg is not None:
        step_cfg = cfg.get('plot_step', cfg)
        if hasattr(step_cfg, 'data_root') and step_cfg.data_root:
            data_root = str(step_cfg.data_root)
        if hasattr(step_cfg, 'output_dir') and step_cfg.output_dir:
            output_dir = str(step_cfg.output_dir)
        if hasattr(step_cfg, 'model_keys') and step_cfg.model_keys:
            model_keys = list(step_cfg.model_keys)
        if hasattr(step_cfg, 'target_metric') and step_cfg.target_metric:
            target_metric = str(step_cfg.target_metric)
        if hasattr(step_cfg, 'scenario_ids') and step_cfg.scenario_ids:
            scenario_ids = list(step_cfg.scenario_ids)
        if hasattr(step_cfg, 'n_seeds') and step_cfg.n_seeds:
            n_seeds = int(step_cfg.n_seeds)
        if hasattr(step_cfg, 'rolling_window') and step_cfg.rolling_window:
            rolling_window = int(step_cfg.rolling_window)

    Config.TARGET_METRIC = target_metric
    plotter = StepMetricPlotterSubplots(rolling_window=rolling_window)

    draw_order = model_keys

    # npz 字段映射（violation 从 step-level distance 派生：dist<0 -> 1，满足 Fig.4/5 时序需求）
    if target_metric == 'violation':
        npz_field_hp = 'step_hp_viol'
        npz_field_nhp = 'step_nhp_viol'
    elif target_metric == 'distance':
        npz_field_hp = 'step_hp_dist'
        npz_field_nhp = 'step_nhp_dist'
    else:
        npz_field_hp = 'step_rewards'
        npz_field_nhp = 'step_rewards'

    for baseline in draw_order:
        # --- 优先尝试新格式（multi-seed npz）---
        seed_data = _load_multi_seed_npz(data_root, baseline, scenario_ids,
                                          npz_field_hp, n_seeds=n_seeds)
        hp_arrays = [arr for s_id in scenario_ids for arr in seed_data.get(s_id, [])]

        seed_data_nhp = _load_multi_seed_npz(data_root, baseline, scenario_ids,
                                              npz_field_nhp, n_seeds=n_seeds)
        nhp_arrays = [arr for s_id in scenario_ids for arr in seed_data_nhp.get(s_id, [])]

        if hp_arrays:
            plotter.add_line(hp_arrays, baseline, is_hp=True)
            plotter.add_line(nhp_arrays if nhp_arrays else hp_arrays, baseline, is_hp=False)
        else:
            # --- 退回旧 JSON 格式（向后兼容）---
            json_dir = os.path.join(Config.save_json_dir)
            json_path = os.path.join(json_dir, f"{baseline}_step_{target_metric}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if data.get('hp'):
                    plotter.add_line(data['hp'], baseline, is_hp=True)
                if data.get('nhp'):
                    plotter.add_line(data['nhp'], baseline, is_hp=False)
            else:
                print(f"Warning: No data found for '{baseline}' (npz or json), skipping.")

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        plot_filename = f"optimized_subplots_{target_metric}.{ext}"
        plotter.save(os.path.join(output_dir, plot_filename))


if __name__ == "__main__":
    # --- 你可以在这里切换画什么 ---

    # 画 Violation
    Config.TARGET_METRIC = "violation"
    plot_step()

    # 如果想画 Distance，解开下面两行
    # Config.TARGET_METRIC = "distance"
    # plot_step()