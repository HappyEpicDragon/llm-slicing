import os
from omegaconf import DictConfig
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

# 引用你的环境和Agent
from src.basic_apis.ppo.ppo_oneshot.global_slicing_env import GlobalSlicingEnv  # 注意文件名
# from src.basic_apis.ppo.ppo_oneshot.agent_oneshot import CustomFeatureExtractor
from src.basic_apis.ppo.ppo_oneshot.agent_oneshot import CustomFeatureExtractor
from pathlib import Path
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy # 导入标准策略

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from src.basic_apis.ppo.utils import calculate_slice_ue_obs # 复用这个计算核心
from collections import deque


class PeriodicMetricsCallback(BaseCallback):
    """
    [精准版] 累积计算核心指标。
    逻辑：逐步累加 info 中的 drift 和 violation，并在 check_freq (Episode结束) 时打印总结。
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self._reset_stats()

        self.RED = "\033[91m"
        self.GREEN = "\033[92m"
        self.YELLOW = "\033[93m"
        self.CYAN = "\033[96m"
        self.RESET = "\033[0m"

    def _reset_stats(self):
        self.stats = {
            "HP": {"viol": 0, "dist": 0.0, "count": 0},
            "NHP": {"viol": 0, "dist": 0.0, "count": 0}
        }
        self.current_step_in_period = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        # 1. 累加逻辑
        self._accumulate(info)
        self.current_step_in_period += 1

        # 2. 周期打印 (Episode Summary)
        if self.current_step_in_period >= self.check_freq:
            self._print_summary()

            # 记录到 TensorBoard
            self.logger.record("metrics/hp_violations_total", self.stats["HP"]["viol"])
            self.logger.record("metrics/hp_distance_total", self.stats["HP"]["dist"])
            self.logger.record("metrics/nhp_violations_total", self.stats["NHP"]["viol"])
            self.logger.record("metrics/nhp_distance_total", self.stats["NHP"]["dist"])

            self._reset_stats()

        return True

    def _accumulate(self, info):
        # 动态扫描所有切片
        slice_indices = set()
        for k in info.keys():
            if k.startswith("meta/slice_") and k.endswith("_priority"):
                slice_indices.add(int(k.split('_')[1]))

        for s_idx in slice_indices:
            # 跳过未激活切片
            if info.get(f"meta/slice_{s_idx}_active", 0) == 0:
                continue

            # 获取优先级 (精准)
            prio = info.get(f"meta/slice_{s_idx}_priority", 0)
            cat = "HP" if prio > 0 else "NHP"

            # 遍历指标
            for metric in ['thr', 'rel', 'lat']:
                # 跳过无 SLA 要求的指标
                if info.get(f"meta/slice_{s_idx}_{metric}_req", 0) == 0:
                    continue

                # 提取数据
                drift = info.get(f"drift/slice_{s_idx}_{metric}", 0.0)

                # 统计
                self.stats[cat]["count"] += 1
                if drift < 0:
                    self.stats[cat]["viol"] += 1
                    self.stats[cat]["dist"] += drift  # 累加负值 (Distance)

    def _print_summary(self):
        hp = self.stats["HP"]
        nhp = self.stats["NHP"]

        hp_rate = hp["viol"] / hp["count"] if hp["count"] > 0 else 0.0
        nhp_rate = nhp["viol"] / nhp["count"] if nhp["count"] > 0 else 0.0

        print(
            f"\n{self.CYAN}{'=' * 35} EPISODE SUMMARY (Last {self.current_step_in_period} Steps) {'=' * 35}{self.RESET}")

        # 打印 HP
        c_hp = self.GREEN if hp["viol"] == 0 else self.RED
        print(f"HIGH Priority (HP):")
        print(f"  > Violations : {c_hp}{hp['viol']}{self.RESET} / {hp['count']} ({hp_rate:.2%})")
        print(f"  > Distance   : {c_hp}{hp['dist']:.4f}{self.RESET} (Sum of negative drifts)")

        # 打印 NHP
        c_nhp = self.YELLOW if nhp["viol"] > 0 else self.GREEN
        print(f"Normal Priority (NHP):")
        print(f"  > Violations : {c_nhp}{nhp['viol']}{self.RESET} / {nhp['count']} ({nhp_rate:.2%})")
        print(f"  > Distance   : {c_nhp}{nhp['dist']:.4f}{self.RESET}")
        print(f"{self.CYAN}{'=' * 100}{self.RESET}\n")


class ConsoleLogCallback(BaseCallback):
    """
    [全维度监控版]
    1. 实时打印三种 Reward 组件 (Base/Shaping/Guidance)。
    2. 实时打印三种 SLA 维度状态 (Throughput/Latency/Reliability)。
    """

    def __init__(self, verbose=0, print_freq=200):
        super().__init__(verbose)
        self.print_freq = print_freq

        # 简单的内部累加器，用于在表格上方显示 "Current Episode Cumulative Stats"
        self.ep_stats = {"HP_viol": 0, "NHP_viol": 0}

        # ANSI 颜色
        self.RED = "\033[91m"
        self.GREEN = "\033[92m"
        self.YELLOW = "\033[93m"
        self.GRAY = "\033[90m"
        self.CYAN = "\033[96m"
        self.RESET = "\033[0m"

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        # 1. TensorBoard 记录 (保持详细记录)
        # 记录三种 Reward 组件
        if "reward/sla_base" in info:
            self.logger.record("reward/sla_base", info["reward/sla_base"])
        if "reward/shaping" in info:
            self.logger.record("reward/shaping", info["reward/shaping"])
        if "reward/guidance" in info:
            self.logger.record("reward/guidance", info["reward/guidance"])

        # 🔥 新增: 记录总 Reward (PPO 已自动记录为 rollout/ep_rew_mean)
        # 但我们可以手动记录每步的即时总 Reward
        total_reward = info.get("reward/sla_base", 0.0) + \
                       info.get("reward/shaping", 0.0) + \
                       info.get("reward/guidance", 0.0)
        self.logger.record("reward/total_step", total_reward)

        # 2. 简单的累加逻辑 (用于显示当前 Episode 违约总数)
        if self.n_calls % 1000 == 1:  # 假设 1000 步一个 Episode，重置
            self.ep_stats = {"HP_viol": 0, "NHP_viol": 0}
        self._quick_accumulate(info)

        # 3. 控制台打印
        if self.n_calls % self.print_freq == 0:
            self._print_console_report(info)

        return True

    def _quick_accumulate(self, info):
        """快速统计当前帧的违约数"""
        for k in info.keys():
            if k.startswith("drift/slice_"):
                # key格式: drift/slice_{s_idx}_{metric}
                try:
                    parts = k.split('_')
                    # parts: ['drift/slice', '0', 'thr']
                    s_idx = int(parts[1])
                    metric = parts[2]

                    # 检查是否激活
                    if info.get(f"meta/slice_{s_idx}_active", 0) == 0: continue
                    # 检查是否有 SLA 要求
                    if info.get(f"meta/slice_{s_idx}_{metric}_req", 0) == 0: continue

                    drift = info[k]
                    if drift < 0:
                        prio = info.get(f"meta/slice_{s_idx}_priority", 0)
                        if prio > 0:
                            self.ep_stats['HP_viol'] += 1
                        else:
                            self.ep_stats['NHP_viol'] += 1
                except:
                    continue

    def _print_console_report(self, info):
        total_steps = self.num_timesteps

        # --- 1. 获取三种 Reward ---
        base_rew = info.get("reward/sla_base", 0.0)
        shap_rew = info.get("reward/shaping", 0.0)
        guid_rew = info.get("reward/guidance", 0.0)
        total_rew = base_rew + shap_rew + guid_rew

        p_scale = info.get("power_scale_factor", 0.0)

        # --- 2. 打印头部 ---
        print(f"\n{'=' * 95}")
        print(f"🚀 [Step {total_steps}] Snapshot | Ep Progress: {total_steps % 1000}/1000")
        print(f"{'=' * 95}")

        # 累计违约数
        hp_v = self.ep_stats['HP_viol']
        nhp_v = self.ep_stats['NHP_viol']
        c_hp = self.RED if hp_v > 0 else self.GREEN
        c_nhp = self.YELLOW if nhp_v > 0 else self.GREEN

        print(f"📊 Ep Violations: HP {c_hp}{hp_v}{self.RESET} | NHP {c_nhp}{nhp_v}{self.RESET}")
        print(
            f"💰 Rewards     : Base {base_rew:+.3f} | Shape {shap_rew:+.3f} | Guide {guid_rew:+.3f} | {self.CYAN}Sum {total_rew:+.3f}{self.RESET}")
        print(f"🔋 Power Scale : {self._color_scale(p_scale)}")
        print(f"{'-' * 95}")

        # --- 3. 打印表格头 ---
        # 调整列宽以容纳三列指标
        # Thr: Throughput, Lat: Latency, Rel: Reliability (Packet Loss)
        print(f"{'Slice':<6} | {'Thr (Drift)':<22} | {'Lat (Drift)':<22} | {'Rel (Drift)':<22}")
        print(f"{'-' * 95}")

        # --- 4. 遍历打印切片行 ---
        # 动态获取切片索引
        slice_indices = set()
        for key in info.keys():
            if key.startswith("drift/slice_"):
                parts = key.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    slice_indices.add(int(parts[1]))

        for s_idx in sorted(list(slice_indices)):
            # 检查是否激活
            if info.get(f"meta/slice_{s_idx}_active", 0) == 0:
                # print(f"{self.GRAY}S-{s_idx:<4} | {'INACTIVE':<75}{self.RESET}") # 可选：隐藏非激活切片以节省空间
                continue

            # 优先级标记
            prio = info.get(f"meta/slice_{s_idx}_priority", 0)
            prio_mark = f"{self.RED}*{self.RESET}" if prio > 0 else " "

            row_str = f"S-{s_idx:<4}{prio_mark}| "

            # 遍历三个指标
            for metric in ['thr', 'lat', 'rel']:
                row_str += self._format_metric_cell(info, s_idx, metric)

            print(row_str)
        print(f"{'=' * 95}\n")

    def _format_metric_cell(self, info, s_idx, metric):
        """辅助函数：格式化单元格字符串"""
        # 检查是否有 SLA 要求
        is_req = info.get(f"meta/slice_{s_idx}_{metric}_req", 0)

        width = 22  # 列宽

        if not is_req:
            # 无要求显示灰色横杠
            content = "---"
            return f"{self.GRAY}{content:<{width}}{self.RESET} | "
        else:
            # 获取 Drift 值
            val = info.get(f"drift/slice_{s_idx}_{metric}", 0.0)

            # 判断状态颜色
            if val < 0:
                # 违约 (红色)
                color = self.RED
                status = "Viol"
            else:
                # 达标 (绿色)
                color = self.GREEN
                status = "OK"

            # 格式化数值 string
            val_str = f"{val:+.4f}"
            content = f"{val_str} ({status})"

            # 组合
            return f"{color}{content:<{width}}{self.RESET} | "

    def _color_scale(self, val):
        if val > 0.95: return f"{self.GREEN}{val:.4f}{self.RESET}"
        if val > 0.50: return f"{self.YELLOW}{val:.4f}{self.RESET}"
        return f"{self.RED}{val:.4f}{self.RESET}"


# class DiagnosisCallback(BaseCallback):
#     """
#     深层诊断回调（适配 One-Shot No-Power Action Space）
#     """
#
#     def __init__(self, verbose=0, check_freq=1000):
#         super().__init__(verbose)
#         self.check_freq = check_freq
#
#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq != 0:
#             return True
#
#         # 获取环境
#         env = self.training_env.envs[0].unwrapped
#
#         # 获取上一帧数据
#         raw_obs = env.last_raw_obs
#         if raw_obs is None:
#             return True
#
#         # 获取动作
#         actions = self.locals.get("actions")
#         if actions is not None:
#             action = actions[0]
#         else:
#             return True
#
#         print(f"\n{'#' * 30} DIAGNOSIS REPORT (Step {self.num_timesteps}) {'#' * 30}")
#
#         # ================= 1. 供需关系分析 =================
#         buffer_occ = raw_obs.get("buffer_occupancies", np.zeros(env.max_users))
#         # 优先使用 bits (新版 Env), 兼容 pkts
#         incoming = raw_obs.get("pkt_incoming_bits", raw_obs.get("pkt_incoming", np.zeros(env.max_users)))
#
#         # 判定活跃用户
#         active_users_idx = np.where((buffer_occ > 1e-9) | (incoming > 1e-9))[0]
#
#         print(f"\n[1. Demand vs Supply]")
#         print(f"  - Active Users Count: {len(active_users_idx)} / {env.max_users}")
#         if len(active_users_idx) > 0:
#             print(f"  - Active Users IDs: {active_users_idx}")
#             # print(f"  - Their Buffer Occ: {buffer_occ[active_users_idx]}")
#         else:
#             print(f"  - SYSTEM IDLE (No Traffic)")
#
#         # ================= 2. 动作有效性分析 (关键修改) =================
#         # [修改] 动作不再包含功率，是一维数组 [User_RBG0, User_RBG1, ... User_RBG26]
#         user_choices = action
#
#         # 统计：分配给了哪些用户 (0=Wait, 1..N=User)
#         allocated_users = user_choices[user_choices > 0] - 1
#
#         # 核心诊断：计算"无效分配"
#         wasted_rbs = 0
#         for u_idx in allocated_users:
#             if u_idx not in active_users_idx:
#                 wasted_rbs += 1
#
#         wasted_ratio = (wasted_rbs / env.num_rbgs) * 100
#
#         print(f"\n[2. Action Efficiency]")
#         print(f"  - RBGs Allocated to Users: {len(allocated_users)} / {env.num_rbgs}")
#         print(f"  - Ghost Allocations (Wasted): {wasted_rbs} RBGs ({wasted_ratio:.1f}%)")
#
#         if wasted_rbs > 0:
#             print(f"    ⚠️ ALERT: Model is allocating resources to empty buffers!")
#         else:
#             print(f"    ✅ GREAT: No wasted resources.")
#
#         # ================= 3. 物理层质量分析 =================
#         csi = raw_obs.get("target_cell_power", np.zeros((env.max_users, env.num_rbgs)))
#         if csi.ndim > 2: csi = np.squeeze(csi)
#
#         if len(allocated_users) > 0:
#             # 采样检查第 0 个 RBG (如果有分配)
#             sample_rb_idx = 0
#             u_id = user_choices[sample_rb_idx]
#             if u_id > 0:
#                 u_idx = u_id - 1
#
#                 # 处理 CSI 维度可能的不一致 [U, R] 或 [R, U]
#                 if csi.shape[0] == env.max_users:
#                     u_csi = csi[u_idx, sample_rb_idx]
#                 else:
#                     u_csi = csi[sample_rb_idx, u_idx]
#
#                 print(f"\n[3. Physics Check (Sample RB 0)]")
#                 print(f"  - Allocated User: {u_idx}")
#                 print(f"  - Raw CSI (Channel Gain): {u_csi:.4e}")
#                 # 估算 SNR
#                 est_snr_db = 10 * np.log10(u_csi / 1e-14 + 1e-30)
#                 print(f"  - Est. SNR: {est_snr_db:.2f} dB")
#
#         print(f"{'#' * 80}\n")
#         return True


class DiagnosisCallback(BaseCallback):
    """
    深层诊断回调（修正版：模拟 Agent 视角）
    """

    def __init__(self, verbose=0, check_freq=1000):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        env = self.training_env.envs[0].unwrapped
        raw_obs = env.last_raw_obs
        if raw_obs is None: return True

        print(f"\n{'#' * 30} DIAGNOSIS REPORT (Step {self.num_timesteps}) {'#' * 30}")

        # =========================================================
        # 🕵️ CSI 物理层深度体检 (Agent View)
        # =========================================================
        # 1. 获取原始数据
        csi_raw = raw_obs.get("target_cell_power")
        if csi_raw.ndim > 2: csi_raw = np.squeeze(csi_raw)

        # 维度对齐 [Users, PRBs]
        if csi_raw.shape[0] == env.num_phys_rbs and csi_raw.shape[1] == env.max_users:
            csi_raw = csi_raw.T

        print(f"\n[1. Raw Physics Data]")
        print(f"  - Raw Shape: {csi_raw.shape} (Expected [25, 135])")

        # =========================================================
        # 2. 模拟 _get_observation 的处理逻辑
        # =========================================================
        # 聚合: 135 -> 27
        rbs_per_rbg = env.num_phys_rbs // env.num_rbgs
        try:
            csi_rbg = csi_raw.reshape(env.max_users, env.num_rbgs, rbs_per_rbg).mean(axis=2)
            print(f"  - Aggregated Shape: {csi_rbg.shape} (Expected [25, 27]) -> ✅ OK")
        except Exception as e:
            print(f"  - Aggregation Failed: {e}")
            return True

        # 转 dBm
        csi_dbm = 10 * np.log10(csi_rbg + 1e-33)

        # 统计 dBm 分布
        min_dbm = np.min(csi_dbm)
        max_dbm = np.max(csi_dbm)
        mean_dbm = np.mean(csi_dbm)

        print(f"\n[2. Aggregated CSI Distribution (dBm)]")
        print(f"  - Min : {min_dbm:.2f} dBm")
        print(f"  - Max : {max_dbm:.2f} dBm")
        print(f"  - Mean: {mean_dbm:.2f} dBm")

        # =========================================================
        # 3. 验证新的归一化标准 [-150, -30]
        # =========================================================
        # 新标准：Lower -150, Upper -30
        assumed_min = -150
        assumed_max = -30

        # 归一化计算
        csi_norm = np.clip((csi_dbm - assumed_min) / (assumed_max - assumed_min) * 2.0 - 1.0, -1.0, 1.0)

        # 检查饱和度
        clipped_low = np.sum(csi_norm == -1.0)
        clipped_high = np.sum(csi_norm == 1.0)
        total = csi_norm.size

        print(f"\n[3. Agent Input Check (Normalized)]")
        print(f"  - Assumption: [{assumed_min}, {assumed_max}] dBm")
        print(f"  - Final Range: [{np.min(csi_norm):.3f}, {np.max(csi_norm):.3f}] (Ideal: [-1, 1])")

        if clipped_low > 0:
            # 如果是因为 -300dBm (死区) 导致的截断，这是正常的
            # 我们只需要检查有没有大量介于 -150 和 -140 之间的有效信号被误杀
            valid_weak_signals = np.sum((csi_dbm > -300) & (csi_dbm < assumed_min))
            print(f"  ℹ️ Low Clipping: {clipped_low}/{total} (Dead zones mapped to -1.0 is OK)")
            if valid_weak_signals > 0:
                print(f"     ⚠️ Warning: {valid_weak_signals} valid weak signals were clipped.")

        if clipped_high > 0:
            print(f"  ⚠️ HIGH SATURATION: {clipped_high}/{total} values are saturated at +1.0!")
            print(f"     Action: Increase assumed_max (current: {assumed_max})")

        if clipped_high == 0:
            print(f"  ✅ High-end Dynamic Range: OK")

        print(f"{'#' * 80}\n")
        return True

def make_env(cfg, path_manager, rank=0, seed=None):
    def _init():
        env = GlobalSlicingEnv(cfg.env_settings, np.random.default_rng(seed + rank), path_manager)
        env = Monitor(env, filename=None)
        return env

    return _init


def train(cfg: DictConfig, path_manager):
    env_config = cfg.environment
    seed = env_config.train_rl.algorithm.seed

    mode = cfg.env_updates.mode
    env_config.env_settings.mode = mode
    scenario_mode = cfg.env_updates.scenario_mode
    env_config.env_settings.scenario_mode = scenario_mode

    env_config.env_settings.model_name = cfg.env_updates.model_name
    env_config.env_settings[scenario_mode][mode].active_scenario_list = cfg.env_updates[scenario_mode][mode].active_scenario_list

    # 创建环境
    env = DummyVecEnv([make_env(env_config, path_manager, rank=0, seed=seed)])

    eval_env_config = env_config.copy()
    eval_env_config.env_settings.mode = 'evaluating'
    eval_env_config.env_settings[scenario_mode]['evaluating'].active_scenario_list = cfg.env_updates[scenario_mode][
        'evaluating'].active_scenario_list


    eval_env = DummyVecEnv([make_env(eval_env_config, path_manager, rank=0, seed=seed + 1000)])



    # === 使用 MaskablePPO ===
    # 注意：MaskablePPO 默认使用 MaskableActorCriticPolicy
    # 提取配置文件中的 feature_extractor 部分
    fe_config = env_config.train_rl.network.feature_extractor
    net_arch = env_config.train_rl.network.net_arch
    from omegaconf import OmegaConf
    net_arch = OmegaConf.to_container(net_arch, resolve=True)

    rl_cfg = env_config.train_rl

    # 构建简单的字典结构，去除 hydra 的复杂结构
    net_config_dict = {
        "global_encoder": list(fe_config.global_encoder.hidden_dims),
        "csi_encoder": list(fe_config.csi_encoder.hidden_dims),
        "slice_alloc_encoder": list(fe_config.slice_alloc_encoder.hidden_dims),
        "user_alloc_encoder": list(fe_config.user_alloc_encoder.hidden_dims),
        "intent_drift_encoder": list(fe_config.intent_drift_encoder.hidden_dims),
        "sla_encoder": list(fe_config.sla_encoder.hidden_dims),
        "topology_encoder": list(fe_config.topology_encoder.hidden_dims),
    }
    import inspect
    print(f"DEBUG: Loaded Class Location: {CustomFeatureExtractor}")
    print(f"DEBUG: Init Signature: {inspect.signature(CustomFeatureExtractor.__init__)}")
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env=env,
        policy_kwargs={
            "features_extractor_class": CustomFeatureExtractor,
            # 在这里传入 net_config
            "features_extractor_kwargs": dict(
                features_dim=fe_config.features_dim,
                net_config=net_config_dict
            ),
            "net_arch": net_arch, # dict(pi=[ 512, 512 ], vf=[ 512, 512 ]),
            "activation_fn": torch.nn.ReLU,
        },
        learning_rate=env_config.train_rl.ppo.learning_rate,
        n_steps=env_config.train_rl.ppo.n_steps,
        batch_size=env_config.train_rl.ppo.batch_size,
        gamma=env_config.train_rl.ppo.gamma,
        verbose=1,
        tensorboard_log=env_config.train_rl.logging.tensorboard_log,
        device=env_config.train_rl.algorithm.device,
        seed=seed,
        ent_coef=0.05
    )

    callbacks = []
    if env_config.train_rl.callbacks.checkpoint.enabled:
        callbacks.append(CheckpointCallback(
            save_freq=env_config.train_rl.callbacks.checkpoint.save_freq,
            save_path=env_config.train_rl.callbacks.checkpoint.save_path,
            name_prefix="ppo_one_shot"
        ))

    # Checkpoint 回调 (保持不变)
    if env_config.train_rl.callbacks.checkpoint.enabled:
        callbacks.append(CheckpointCallback(
            save_freq=env_config.train_rl.callbacks.checkpoint.save_freq,
            save_path=env_config.train_rl.callbacks.checkpoint.save_path,
            name_prefix="ppo_one_shot"
        ))

    if env_config.train_rl.callbacks.eval.enabled:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=env_config.train_rl.saving.best_model.save_path,
            log_path=env_config.train_rl.logging.log_path,
            eval_freq=env_config.train_rl.evaluation.eval_freq,
            n_eval_episodes=env_config.train_rl.evaluation.n_eval_episodes,
            deterministic=env_config.train_rl.evaluation.deterministic,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)

    callbacks.append(ConsoleLogCallback())
    # callbacks.append(DiagnosisCallback(check_freq=100))
    callbacks.append(PeriodicMetricsCallback(check_freq=1000))

    print("开始训练 Global One-Shot PPO with Action Masking...")
    model.learn(
        total_timesteps=env_config.train_rl.training.total_timesteps,
        callback=callbacks,
        progress_bar = True
    )

    model.save(str(Path(env_config.train_rl.saving.save_path) / "final_model"))
    env.close()
    eval_env.close()