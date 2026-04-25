import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional
from copy import deepcopy
from hydra.utils import get_class

from src.basic_apis.network_slicing_business.network_slicing_business_executor import ComponentConfig, ComponentClasses, \
    ComponentFactory, NetworkSlicingBusinessExecutor
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.ppo.utils import intent_drift_calc


class EnvironmentState:
    """管理环境状态和episode逻辑"""

    def __init__(self, config):
        self.config = config
        self.episode_interval_start = self.config.episode_interval_start
        self.episode_interval_end = self.config.episode_interval_end
        self.episode_cross_scenario_skip = self.config.episode_cross_scenario_skip
        self.active_scenario_list = self.config.active_scenario_list
        self.current_scenario_idx = 0
        self.initial_episode_number = self.episode_interval_start + self.episode_cross_scenario_skip * \
                                      self.active_scenario_list[self.current_scenario_idx]
        self.max_number_episodes = self.episode_interval_end + self.episode_cross_scenario_skip * \
                                   self.active_scenario_list[self.current_scenario_idx]

        self.step_number = 0  # rb id
        self.channel_timestep = 0
        self.episode_number = self.initial_episode_number
        self.total_steps = 0

    def reset(self, initial_episode: int = -1):
        if initial_episode != -1:
            self.config.initial_episode_number = initial_episode
            self.episode_number = initial_episode
            self.channel_timestep = 0
        elif self.step_number == self.config.max_number_steps:
            self._advance_channel_timestep()
        self.step_number = 0

    def step(self) -> bool:
        self.step_number += 1
        self.total_steps += 1
        return self.is_episode_done()

    def is_episode_done(self) -> bool:
        return self.step_number >= self.config.max_number_steps

    def _advance_channel_timestep(self):
        self.channel_timestep += 1
        if self.channel_timestep >= self.config.max_channel_timesteps:
            self.channel_timestep = 0
            self._advance_channel_episode()

    def _advance_channel_episode(self):
        if self.episode_number < (self.max_number_episodes - 1):
            self.episode_number += 1
        elif self.episode_number == (self.max_number_episodes - 1):
            if self.current_scenario_idx == len(self.active_scenario_list) - 1:
                self.current_scenario_idx = 0
            else:
                self.current_scenario_idx += 1
            self.initial_episode_number = self.active_scenario_list[
                                              self.current_scenario_idx] * self.episode_cross_scenario_skip + self.episode_interval_start
            self.episode_number = self.initial_episode_number
            self.max_number_episodes = self.episode_interval_end + self.episode_cross_scenario_skip * \
                                       self.active_scenario_list[self.current_scenario_idx]
        else:
            raise Exception(f"Episode number error: {self.episode_number}")

    def current_channel_episode(self):
        return self.episode_number


class PSJRAEnv(gym.Env):
    """Progressive Sequential Joint Resource Allocation Environment"""

    def __init__(self, env_config, np_random, path_context):
        super().__init__()
        self.env_config = env_config
        self.mode = self.env_config.mode
        self.scenario_mode = self.env_config.scenario_mode
        self.state_config = self.env_config.env_state_config

        # 配置 State
        self.state_config.episode_interval_start = self.env_config[self.scenario_mode][self.mode].init_scenario_episode
        self.state_config.episode_interval_end = self.env_config[self.scenario_mode][self.mode].max_scenario_episodes
        self.state_config.episode_cross_scenario_skip = self.env_config[self.scenario_mode].episode_cross_scenario_skip
        self.state_config.active_scenario_list = self.env_config[self.scenario_mode][self.mode].active_scenario_list

        self.env_state = EnvironmentState(self.state_config)
        self.np_random = np_random
        self.component_config = ComponentConfig(self.env_config.components)

        # 工厂模式创建组件
        association_class = get_class(self.env_config.components.association[self.scenario_mode].class_path)
        traffic_class = get_class(self.env_config.components.traffic.class_path)
        mobility_class = get_class(self.env_config.components.mobility.class_path)
        channel_class = get_class(self.env_config.components.channel[self.scenario_mode].class_path)

        component_classes = ComponentClasses(ChannelClass=channel_class, AssociationClass=association_class,
                                             TrafficClass=traffic_class, MobilityClass=mobility_class)
        self.component_factory = ComponentFactory(component_config=self.component_config,
                                                  component_classes=component_classes, np_random=self.np_random,
                                                  path_context=path_context)

        self.business_executor = None
        self._create_scenario()

        # 环境参数
        self.max_number_users = self.env_config.components.ues.max_number_ues
        self.max_number_rbs = self.env_config.components.basestations.num_available_rbs[0]
        self.max_number_slices = self.env_config.components.slices.max_number_slices
        self.number_power_levels = 10
        self.P_total = self.env_config.components.basestations.total_power
        self.bandwidth_total = self.component_config.basestation_config.bandwidths[0]
        self.num_rbs = self.component_config.basestation_config.num_available_rbs[0]

        # 功率等级 (Log分布)
        self.P_total = 100.0  # W (基站总功率)
        self.num_rbs = 135  # 资源块数
        self.avg_power_per_rb = self.P_total / self.num_rbs  # ≈ 0.74W

        # 10 级功率：围绕 0.74W 设计
        self.power_levels = np.array([
            0.05,  # Level 0: 极低功率（6.7% 平均值，紧急节能）
            0.15,  # Level 1: 很低功率（20% 平均值）
            0.30,  # Level 2: 低功率（40% 平均值）
            0.50,  # Level 3: 中低功率（67% 平均值）
            0.74,  # Level 4: 🎯 平均功率（100% 平均值）
            1.00,  # Level 5: 中等功率（135% 平均值）
            1.50,  # Level 6: 中高功率（200% 平均值）
            2.00,  # Level 7: 高功率（270% 平均值）
            2.50,  # Level 8: 很高功率（338% 平均值）
            3.50,  # Level 9: 峰值功率（473% 平均值）
        ])

        # 验证功率配置
        print(f"\n{'=' * 80}")
        print(f"[POWER CONFIG] 功率等级配置")
        print(f"{'=' * 80}")
        print(f"  总功率预算: {self.P_total}W")
        print(f"  资源块数量: {self.num_rbs} RB")
        print(f"  平均功率/RB: {self.avg_power_per_rb:.3f}W")
        print(f"  功率等级 (W): {self.power_levels}")
        print(f"  Level 4 (平均): {self.power_levels[4]:.3f}W")
        print(
            f"  Level 9 (峰值): {self.power_levels[9]:.3f}W (占总功率 {self.power_levels[9] / self.P_total * 100:.1f}%)")
        print(f"{'=' * 80}\n")

        # 动作空间: [User ID, Power Level]
        self.action_space = spaces.MultiDiscrete([self.max_number_users + 1, self.number_power_levels])

        # 观测空间
        self.observation_space = self._build_observation_space()

        # 状态变量
        self.t = 0
        self.remaining_power = 100
        self.prev_intent_drift = None
        self.last_raw_obs = {}
        self.episode_count = 0
        self.total_steps = 0

        # RBG 设置
        self.rbg_size = 5

        # 奖励函数注册
        self.reward_type = getattr(env_config, 'reward_type', 'sla_aware')
        self._immediate_reward_func = self._get_reward_function()
        print(f"[Env Init] Reward Type: {self.reward_type}, RBG Size: {self.rbg_size}")
        self.episode_count = 0

        # [新增] 用于存储当前 Episode 内每一步的 Reward
        self.current_ep_reward_trace = []
        self.episode_accumulated_reward = 0.0  # [NEW] 记账本
        # 在 __init__ 的最后添加
        self.action_history = {'users': [], 'powers': []}
        self.current_ep_reward_trace = []
        self.episode_accumulated_reward = 0.0
        self._last_channel_episode = self.env_state.episode_number
        self.episode_immediate_rewards = []
        self.tti_duration = 1e-3

    def _get_reward_function(self):
        reward_functions = {
            'sparse': self._compute_immediate_reward_sparse,
            'mean': self._compute_immediate_reward_mean,
            'positive_sum': self._compute_immediate_reward_positive_sum,
            'weighted': self._compute_immediate_reward_weighted,
            'sla_aware': self._compute_immediate_reward_sla_aware,
        }
        return reward_functions.get(self.reward_type, reward_functions['sparse'])

    def _build_observation_space(self):
        return spaces.Dict({
            'timestep': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'remaining_rbs': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_rb_id': spaces.Box(low=0, high=self.max_number_rbs, shape=(1,), dtype=np.int32),
            'remaining_power': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'csi_current_rb': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_number_users,), dtype=np.float32),
            'slice_allocated_power': spaces.Box(low=0.0, high=self.P_total, shape=(1, self.max_number_slices),
                                                dtype=np.float32),
            'slice_allocated_rbs': spaces.Box(low=0.0, high=self.max_number_rbs, shape=(1, self.max_number_slices),
                                              dtype=np.float32),
            'user_allocated_rbs': spaces.Box(low=0.0, high=self.max_number_rbs, shape=(1, self.max_number_users),
                                             dtype=np.float32),
            'user_allocated_power': spaces.Box(low=0.0, high=self.P_total, shape=(1, self.max_number_users),
                                               dtype=np.float32),
            'intent_drift': spaces.Box(low=-2, high=1, shape=(self.max_number_slices, 5, 3), dtype=np.float32),
            'sla_target': spaces.Box(low=0.0, high=np.inf, shape=(self.max_number_slices, 3), dtype=np.float32),
            'active_sla': spaces.Box(low=0.0, high=1.0, shape=(self.max_number_slices, 3), dtype=np.float32),
            'slice_priority': spaces.Box(low=0.0, high=1.0, shape=(self.max_number_slices,), dtype=np.float32),
            'user_to_slice': spaces.Box(low=0, high=1, shape=(self.max_number_slices, self.max_number_users),
                                        dtype=np.int32),
            'active_users': spaces.Box(low=0, high=1, shape=(1, self.max_number_users), dtype=np.int8),
            'active_slices': spaces.Box(low=0, high=1, shape=(1, self.max_number_slices), dtype=np.int8),
        })

    # def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict, Dict]:
    #     if not seed:
    #         seed = 42
    #     else:
    #         seed = int(seed)
    #     super().reset(seed=seed)
    #     np.random.seed(seed)
    #
    #     self._initialize_state()
    #     self.episode_count += 1
    #     self.episode_immediate_rewards = []
    #     self.prev_intent_drift = None
    #
    #     # 配置同步检查
    #     if self.env_state.config.max_number_steps != self.max_number_rbs:
    #         print(
    #             f"⚠️ Config Mismatch Fixed: max_number_steps {self.env_state.config.max_number_steps} -> {self.max_number_rbs}")
    #         self.env_state.config.max_number_steps = self.max_number_rbs
    #
    #     self.env_state.reset()
    #     self.business_executor.update_associations(self.env_state.channel_timestep, self.env_state.episode_number)
    #     # [新增] 清空 trace
    #     self.current_ep_reward_trace = []
    #     self.episode_accumulated_reward = 0.0  # [NEW] 记账本
    #
    #     return self._get_obs({}), self._get_info()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict, Dict]:
        """
        重置环境（正确版本）
        注意：不清空 Buffer！Buffer 应该跨 Episode 持续存在
        """
        if not seed:
            seed = 42
        else:
            seed = int(seed)
        super().reset(seed=seed)
        np.random.seed(seed)

        # ==================== 重置 Episode 级别的状态 ====================
        self._initialize_state()
        self.episode_count += 1
        self.episode_immediate_rewards = []
        self.prev_intent_drift = None

        # 配置同步检查
        if self.env_state.config.max_number_steps != self.max_number_rbs:
            print(
                f"⚠️ Config Mismatch Fixed: max_number_steps {self.env_state.config.max_number_steps} -> {self.max_number_rbs}")
            self.env_state.config.max_number_steps = self.max_number_rbs

        self.env_state.reset()
        self.business_executor.update_associations(self.env_state.channel_timestep, self.env_state.episode_number)

        # ==================== ✅ 只在场景切换时清空 Buffer ====================
        # 检查是否切换了 channel_episode (场景切换)
        if hasattr(self, '_last_channel_episode'):
            current_channel_episode = self.env_state.episode_number
            if self.env_state.episode_number != self._last_channel_episode:
                print(f"[RESET] Switching Channel Episode -> CLEARING BUFFERS")
                # 强制清空 UEs 状态
                if hasattr(self.business_executor.components.ues, 'buffer_state'):
                    self.business_executor.components.ues.buffer_state.fill(0)

        # 记录当前 channel_episode
        self._last_channel_episode = self.env_state.episode_number

        # ==================== 清空 Episode 级别的累积变量 ====================
        self.current_ep_reward_trace = []
        self.episode_accumulated_reward = 0.0

        return self._get_obs({}), self._get_info()

    def step(self, action):
        """RBG Step Function (极端延迟检测版)"""

        # ✅ 在循环开始前添加
        if not hasattr(self, '_audit_printed'):
            self._audit_printed = False

        user_id, power_level = action
        power_value = self.power_levels[power_level]

        if not hasattr(self, 'action_history'):
            self.action_history = {'users': [], 'powers': []}

        self.action_history['users'].append(int(user_id))
        self.action_history['powers'].append(int(power_level))

        # 动作探针
        if self.total_steps % 100 == 0 and self.total_steps > 0:
            print(
                f"[ACTION SAMPLE] Step {self.total_steps}: User={user_id}, Power={power_level}, PowerVal={power_value:.3f}W")

        actual_user_idx = -1
        if user_id > 0:
            actual_user_idx = int(user_id) - 1

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        last_raw_obs = None

        # ==================== ✅ 新增：提前检测极端延迟 ====================
        if hasattr(self, 'last_raw_obs') and self.last_raw_obs:
            avg_latency_steps = np.mean(self.last_raw_obs.get("buffer_latencies", [0.0]))
            avg_latency_sec = avg_latency_steps * self.tti_duration

            if avg_latency_sec > 15.0:  # 15 秒
                print(f"[EARLY TERMINATE] Ep {self.episode_count}: "
                      f"AvgLatency={avg_latency_sec * 1000:.0f}ms ({avg_latency_steps:.1f} steps) > 15s")
                terminated = True
                final_obs = self._get_obs(self.last_raw_obs)
                final_reward = -100.0

                # 清理
                self.current_ep_reward_trace.append(final_reward)
                self.action_history = {'users': [], 'powers': []}
                self.current_ep_reward_trace = []
                self.episode_accumulated_reward = 0.0
                return final_obs, final_reward, terminated, truncated, info

        # === RBG Loop (5 TTI) ===
        for rbg_idx in range(self.rbg_size):
            if self.env_state.step_number >= self.max_number_rbs:
                terminated = True
                break

            # 功率扣除
            if actual_user_idx != -1:
                self.remaining_power -= power_value
                if self.remaining_power < 0:
                    self.remaining_power = 0

            # 物理环境 Step
            raw_obs_data = self.business_executor.step(
                action, self.env_state.step_number, self.env_state.channel_timestep, self.env_state.episode_number
            )

            # ... (State probe print) ...
            # [FIX] Print correct units in state probe
            if (self.episode_count % 20 == 0) and (self.env_state.step_number % 10 == 0):
                lat_steps = np.mean(raw_obs_data['buffer_latencies'])
                lat_ms = lat_steps * self.tti_duration * 1000
                print(f"[STATE] Ep {self.episode_count} Step {self.env_state.step_number} "
                      f"AvgBuffer={np.mean(raw_obs_data['buffer_occupancies']):.3f} "
                      f"AvgLatency={lat_ms:.1f}ms ({lat_steps:.1f} steps)")

            # 计算辅助数据
            intent_drift = intent_drift_calc([raw_obs_data], 5, 0.2)
            raw_obs_data["intent_drift"] = intent_drift

            priority_array = np.zeros(self.max_number_slices, dtype=np.float32)
            slice_req = raw_obs_data.get("slice_req", {})
            for idx in range(self.max_number_slices):
                req = slice_req.get(f'slice_{idx}', {})
                if req:
                    priority_array[idx] = req.get('priority', 0)
            raw_obs_data["slice_priority"] = priority_array

            self.last_raw_obs = raw_obs_data
            last_raw_obs = raw_obs_data

            if self.env_config.mode == 'testing' and self.env_state.step_number + 1 == self.max_number_rbs:
                self.business_executor.step_metric(raw_obs_data)

            # 推进环境
            terminated_env = self.env_state.step()
            self.t = self.env_state.step_number
            self.total_steps = self.env_state.total_steps

            # ==================== ✅ 修复：RBG 内部延迟检测 ====================
            avg_latency_steps = np.mean(raw_obs_data["buffer_latencies"])
            avg_latency_sec = avg_latency_steps * self.tti_duration

            if avg_latency_sec > 15.0:
                if np.random.rand() < 0.1:
                    print(f"[RBG TERMINATE] Ep {self.episode_count} Step {self.t}: "
                          f"Latency={avg_latency_sec * 1000:.0f}ms > 15s")
                terminated_env = True

            # 计算奖励
            prev_drift = self.prev_intent_drift if self.prev_intent_drift is not None else np.zeros_like(intent_drift)

            immediate_reward = self._compute_immediate_reward_sla_aware(
                prev_drift, intent_drift, action, raw_obs_data["slice_ue_assoc"],
                priority_array, raw_obs_data["basestation_slice_assoc"][0], slice_req, raw_obs_data
            )

            self.prev_intent_drift = intent_drift.copy()
            total_reward += immediate_reward

            if terminated_env:
                terminated = True
                break

        # === Loop End ===
        observation = self._get_obs(last_raw_obs)
        final_reward = total_reward / self.rbg_size

        # 测试保存逻辑
        if self.env_config.mode == 'testing' and terminated:
            next_ts = self.env_state.channel_timestep + 1
            if next_ts >= self.state_config.max_channel_timesteps:
                curr_ep = self.env_state.current_channel_episode()
                self.business_executor.save_metric(self.env_config.model_name, f"ep_{curr_ep}.npz")
                if hasattr(self.business_executor, 'reset_metric'):
                    self.business_executor.reset_metric()

        self.current_ep_reward_trace.append(final_reward)

        # Episode 结束统计
        if terminated:
            user_counts = np.bincount(self.action_history['users'], minlength=self.max_number_users + 1)
            power_counts = np.bincount(self.action_history['powers'], minlength=self.number_power_levels)

            if len(self.current_ep_reward_trace) > 1:
                process_rewards = self.current_ep_reward_trace[:-1]
                terminal_reward = self.current_ep_reward_trace[-1]
                avg_process = np.mean(process_rewards)
                total_process = np.sum(process_rewards)
            else:
                avg_process, total_process, terminal_reward = 0, 0, self.current_ep_reward_trace[0]

            # 详细打印（每 5 个 Episode）
            if self.episode_count % 5 == 0:
                print(f"\n{'=' * 80}")
                print(f"📊 [EPISODE {self.episode_count} COMPLETE REPORT]")
                print(f"{'=' * 80}")

                print(f"💰 REWARD BREAKDOWN:")
                print(
                    f"  ├─ Process (Steps 0-{len(process_rewards) - 1}): Avg={avg_process:.2f}, Total={total_process:.2f}")
                print(f"  ├─ Terminal (Step {len(process_rewards)}): {terminal_reward:.2f}")
                print(f"  └─ Episode Total: {total_process + terminal_reward:.2f}")

                print(f"\n🎯 ACTION STATISTICS:")
                top_5_users = np.argsort(user_counts)[-5:][::-1]
                print(
                    f"  ├─ User 0 (Wait): {user_counts[0]} times ({user_counts[0] / len(self.action_history['users']) * 100:.1f}%)")
                print(f"  ├─ Top 5 Active Users: {top_5_users}")
                print(f"  └─ Power Distribution: {power_counts}")
                print(f"{'=' * 80}\n")

            # ✅ 清理标记
            if hasattr(self, '_audit_printed'):
                delattr(self, '_audit_printed')

            # 清空历史
            self.action_history = {'users': [], 'powers': []}
            self.current_ep_reward_trace = []
            self.episode_accumulated_reward = 0.0

        # 在 RBG Loop 结束后，terminated 判断前添加
        if terminated and self.episode_count % 10 == 0:
            print(f"\n[SLICE 4 DIAGNOSIS] Episode {self.episode_count}")
            slice_4_users = (last_raw_obs["slice_ue_assoc"][4] == 1).nonzero()[0]
            if len(slice_4_users) > 0:
                lat_steps = last_raw_obs['buffer_latencies'][slice_4_users]
                lat_ms = lat_steps * self.tti_duration * 1000
                print(f"  Users: {slice_4_users}")
                print(f"  Buffers: {last_raw_obs['buffer_occupancies'][slice_4_users]}")
                print(f"  Latencies (ms): {lat_ms}")  # Corrected

        return observation, final_reward, terminated, truncated, info


    def action_masks(self) -> np.ndarray:
        user_mask = self._compute_user_mask()
        power_mask = self._compute_power_mask()
        return np.concatenate([user_mask, power_mask])

    # def _compute_user_mask(self) -> np.ndarray:
    #     ue_assoc = self.components.basestations.ue_assoc[0]
    #     return np.insert(ue_assoc, 0, True).astype(bool)

    def _compute_user_mask(self) -> np.ndarray:
        """
        [Data-Only Fix] 计算用户动作掩码
        核心修改：移除 ue_assoc (连接状态) 的硬性过滤。
        原因：ue_assoc 变量可能滞后或不准，导致有数据且信道良好的用户被误屏蔽。
        我们只过滤“空Buffer”用户，连接性问题交给物理层和Reward去处理。
        """
        # 1. 确保获取最新组件
        if self.business_executor is not None:
            target_components = self.business_executor.components
        else:
            target_components = self.components

        # 2. 获取 Buffer 状态 (这是唯一真理)
        has_data_mask = np.zeros(self.max_number_users, dtype=bool)
        try:
            if hasattr(target_components.ues, 'buffer_state'):
                # 向量化
                buffer_occupancy = np.sum(target_components.ues.buffer_state, axis=1)
                has_data_mask = buffer_occupancy > 0.001
            elif hasattr(target_components.ues, 'buffers'):
                # 对象
                occupancies = np.array([b.get_buffer_occupancy() for b in target_components.ues.buffers])
                has_data_mask = occupancies > 0.001
        except Exception as e:
            print(f"Mask Error: {e}")
            # 兜底：全 True
            has_data_mask = np.ones(self.max_number_users, dtype=bool)

        # 3. 构建 Mask (只看有没有数据！)
        # 不再与 ue_assoc 做 AND 运算
        full_mask = np.zeros(self.max_number_users + 1, dtype=bool)
        full_mask[1:] = has_data_mask

        # 4. 处理 User 0 (Wait)
        if np.any(has_data_mask):
            full_mask[0] = False  # 有活干就不许停
        else:
            full_mask[0] = True  # 没活干才休息

        # 5. 安全兜底 (防止全 False 死锁)
        if not np.any(full_mask):
            full_mask[0] = True

        return full_mask

    def _compute_power_mask(self) -> np.ndarray:
        power_mask = (self.power_levels <= self.remaining_power)
        if not np.any(power_mask): power_mask[0] = True
        if self.remaining_power < 0.2 * self.P_total:
            power_mask[:] = False
            power_mask[:3] = True
        return power_mask.astype(bool)

    def _initialize_state(self) -> None:
        self.t = 0
        self.remaining_power = 100
        self.prev_reward = None

    def _get_obs(self, raw_obs_data) -> Dict:
        if not raw_obs_data: return self._get_default_obs()

        # 计算前瞻 CSI
        current_step, max_rbs = self.t, self.max_number_rbs
        csi_obs = np.zeros(self.max_number_users, dtype=np.float32)
        full_channel_data = getattr(self.business_executor.components.channel, 'cached_channel_data', None)

        if full_channel_data is None and "target_cell_power" in raw_obs_data:
            full_channel_data = raw_obs_data["target_cell_power"]

        if full_channel_data is not None:
            try:
                squeezed = np.squeeze(full_channel_data)
                if squeezed.ndim == 1: squeezed = squeezed[np.newaxis, :]

                if current_step < max_rbs:
                    end = min(current_step + self.rbg_size, max_rbs)
                    if squeezed.shape[0] > squeezed.shape[1]:  # [Time, User]
                        csi_obs = np.mean(squeezed[current_step:end, :], axis=0)
                    else:  # [User, Time]
                        csi_obs = np.mean(squeezed[:, current_step:end], axis=1)
            except Exception:
                pass

        if csi_obs.shape != (self.max_number_users,):
            csi_obs = np.zeros(self.max_number_users, dtype=np.float32)

        # 提取统计
        rb_alloc = raw_obs_data["sched_decision"]
        slice_assoc = raw_obs_data["slice_ue_assoc"]
        user_pwr = np.sum(rb_alloc, axis=1)
        user_rbs = np.sum(rb_alloc > 0, axis=1)

        sla_t, sla_a = self._extract_sla_info(raw_obs_data["slice_req"])

        obs = {
            'timestep': np.array([self.t / self.max_number_rbs], dtype=np.float32),
            'remaining_rbs': np.array([(self.max_number_rbs - self.t) / self.max_number_rbs], dtype=np.float32),
            'current_rb_id': np.array([self.t], dtype=np.int32),
            'remaining_power': np.array([self.remaining_power / self.P_total], dtype=np.float32),
            'csi_current_rb': csi_obs.astype(np.float32),
            'slice_allocated_power': (slice_assoc @ user_pwr.T).T.reshape(1, -1).astype(np.float32),
            'slice_allocated_rbs': (slice_assoc @ user_rbs.T).T.reshape(1, -1).astype(np.float32),
            'user_allocated_rbs': user_rbs.reshape(1, -1).astype(np.float32),
            'user_allocated_power': user_pwr.reshape(1, -1).astype(np.float32),
            'intent_drift': raw_obs_data["intent_drift"].astype(np.float32),
            'sla_target': sla_t.astype(np.float32),
            'active_sla': sla_a.astype(np.float32),
            'slice_priority': raw_obs_data["slice_priority"].astype(np.float32),
            'user_to_slice': raw_obs_data["slice_ue_assoc"].astype(np.int32),
            'active_users': raw_obs_data["basestation_ue_assoc"].reshape(1, -1).astype(np.int8),
            'active_slices': raw_obs_data["basestation_slice_assoc"].reshape(1, -1).astype(np.int8),
        }
        self._validate_observation_shapes(obs)
        return obs

    def _get_info(self) -> Dict:
        """
        生成额外信息字典
        """
        return {}

    def _validate_observation_shapes(self, observation: Dict):
        """
        验证观测形状是否与观测空间匹配 (调试用，防止维度错误)
        """
        for key, value in observation.items():
            # 忽略不在空间定义中的临时键
            if key not in self.observation_space.spaces:
                continue

            expected_shape = self.observation_space.spaces[key].shape
            actual_shape = value.shape

            if expected_shape != actual_shape:
                # 打印详细错误帮助定位
                raise ValueError(
                    f"观测 '{key}' 形状不匹配!\n"
                    f"  期望: {expected_shape}\n"
                    f"  实际: {actual_shape}\n"
                    f"  值: {value}"
                )

    def _create_scenario(self):
        self.components = self.component_factory.create_scenario_components(self.env_state.episode_number,
                                                                            self.env_state.channel_timestep)
        self.business_executor = NetworkSlicingBusinessExecutor(self.components, self.component_config)

    def _compute_reward(self, intent_drift, slice_req, slice_ue_assoc, basestation_slice_assoc, slice_priorities,
                        priority_flag=True):
        # Global Reward (Baseline logic)
        n_slices = intent_drift.shape[0]
        active_slices = basestation_slice_assoc.nonzero()[0]
        active_observations = np.zeros(n_slices)
        metrics_map = {"throughput": 0, "reliability": 1, "latency": 2}

        for slice_idx in active_slices:
            slice_ues = slice_ue_assoc[slice_idx].nonzero()[0]
            if slice_ues.shape[0] == 0:
                active_observations[slice_idx] = 1
                continue

            slice_config = slice_req.get(f"slice_{slice_idx}", {})
            if not slice_config or 'parameters' not in slice_config:
                active_observations[slice_idx] = 1
                continue

            configured_metrics = [metrics_map[p["name"]] for p in slice_config["parameters"].values()]
            slice_metrics = [np.mean(intent_drift[slice_idx, :slice_ues.shape[0], m]) for m in configured_metrics]
            active_observations[slice_idx] = np.min(slice_metrics)

        if np.isclose(np.sum(active_observations < 0), 0):
            return np.mean(active_observations)
        elif not np.isclose(np.sum((slice_priorities * active_observations) < 0), 0) and priority_flag:
            negative_idx = (active_observations * slice_priorities < 0).nonzero()[0]
            return np.mean(active_observations[negative_idx]) - 1
        else:
            negative_idx = (active_observations < 0).nonzero()[0]
            return np.mean(active_observations[negative_idx])

    def _extract_sla_info(self, slice_req):
        num_slices, num_sla_types = self.max_number_slices, 3
        sla_target = np.zeros((num_slices, num_sla_types))
        active_sla = np.zeros((num_slices, num_sla_types), dtype=int)
        sla_map = {'throughput': 0, 'reliability': 1, 'latency': 2}

        for slice_idx in range(num_slices):
            req = slice_req.get(f'slice_{slice_idx}', {})
            if req and 'parameters' in req:
                for p in req['parameters'].values():
                    name = p.get('name', '').lower()
                    if name in sla_map:
                        idx = sla_map[name]
                        sla_target[slice_idx, idx] = p['value']
                        active_sla[slice_idx, idx] = 1
        return sla_target, active_sla

    def _get_default_obs(self) -> Dict:
        return {
            'timestep': np.array([0.0], dtype=np.float32),
            'remaining_rbs': np.array([1.0], dtype=np.float32),
            'current_rb_id': np.array([0], dtype=np.int32),
            'remaining_power': np.array([1.0], dtype=np.float32),
            'csi_current_rb': np.zeros(self.max_number_users, dtype=np.float32),
            'slice_allocated_power': np.zeros((1, self.max_number_slices), dtype=np.float32),
            'slice_allocated_rbs': np.zeros((1, self.max_number_slices), dtype=np.float32),
            'user_allocated_rbs': np.zeros((1, self.max_number_users), dtype=np.float32),
            'user_allocated_power': np.zeros((1, self.max_number_users), dtype=np.float32),
            'intent_drift': np.zeros((self.max_number_slices, 5, 3), dtype=np.float32),
            'sla_target': np.zeros((self.max_number_slices, 3), dtype=np.float32),
            'active_sla': np.zeros((self.max_number_slices, 3), dtype=np.float32),
            'slice_priority': np.zeros(self.max_number_slices, dtype=np.float32),
            'user_to_slice': np.zeros((self.max_number_slices, self.max_number_users), dtype=np.int32),
            'active_users': np.zeros((1, self.max_number_users), dtype=np.int8),
            'active_slices': np.zeros((1, self.max_number_slices), dtype=np.int8),
        }

    def _get_user_slice(self, user_idx, slice_ue_assoc):
        for i in range(slice_ue_assoc.shape[0]):
            if slice_ue_assoc[i, user_idx] == 1: return i
        return None

    def _calculate_instant_capacity(self, user_idx, power_val, raw_obs_data):
        rb_idx = self.t - 1
        if rb_idx < 0 or rb_idx >= self.max_number_rbs: return 0.0

        csi = raw_obs_data.get("target_cell_power")
        if csi is None: return 0.0

        gain = 0.0
        try:
            if csi.shape == (self.max_number_users, self.max_number_rbs):
                gain = csi[user_idx, rb_idx]
            elif csi.shape == (self.max_number_rbs, self.max_number_users):
                gain = csi[rb_idx, user_idx]
            elif csi.ndim == 1:
                gain = csi[user_idx]
        except:
            return 0.0

        bw = self.component_config.basestation_config.bandwidths[0]
        n_rbs = self.component_config.basestation_config.num_available_rbs[0]
        noise = getattr(self.components.basestations, 'noise_power', 1e-14)

        sinr = (power_val * gain) / noise
        return (bw / n_rbs * np.log2(1.0 + sinr)) / 1e6

    # --- Reward Functions ---

    def _compute_immediate_reward_sla_aware(
            self,
            old_drift,
            new_drift,
            action,
            slice_ue_assoc,
            slice_priority,
            bs_slice_assoc,
            slice_req,
            raw_obs_data=None,
    ):
        """
        SLA-Aware Reward (稳健版)

        核心修复：
        1. 所有惩罚均有安全上限
        2. 极端延迟立即终止（修复终止逻辑）
        3. 使用分段线性惩罚（移除指数）
        4. 增加数值溢出保护
        """
        # ==================== 🎯 安全参数配置 ====================
        EMPTY_BUFFER_THRESHOLD = 0.02
        PHY_SCORE_SCALE = 0.2

        # 功率激励
        POWER_INCENTIVE_MAP = {
            0: -2.0, 1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5,
            5: 1.5, 6: 2.0, 7: 2.5, 8: 2.0, 9: 1.5,
        }

        # ✅ 延迟惩罚配置（安全版）
        MAX_LATENCY_PENALTY_PER_STEP = 30.0  # ✅ 每步最大惩罚（硬上限）
        EXTREME_LATENCY_THRESHOLD = 10.0

        CLAWBACK_RATE = 0.10
        MAX_CLAWBACK = 3.0
        TERMINAL_PENALTY_FACTOR = 1.0
        MAX_AUDIT_PENALTY = 10.0
        PRIORITY_GAIN = 1.0

        # 训练阶段
        WARMUP_EPISODES = 300
        TRANSITION_EPISODES = 600

        if self.episode_count < WARMUP_EPISODES:
            latency_tolerance_factor = 2.0
            drift_violation_threshold = -0.8
        elif self.episode_count < TRANSITION_EPISODES:
            latency_tolerance_factor = 1.2
            drift_violation_threshold = -0.5
        else:
            latency_tolerance_factor = 1.0
            drift_violation_threshold = 0.0

        user_id, power_level = action

        # 无效动作惩罚
        if user_id == 0:
            return -1.0

        actual_user_idx = int(user_id) - 1

        if self.components.basestations.ue_assoc[0, actual_user_idx] == 0:
            return -2.0

        # 物理层得分
        power_val = self.power_levels[power_level]
        curr_obs = raw_obs_data if raw_obs_data else self.last_raw_obs
        occ = curr_obs["buffer_occupancies"][actual_user_idx]

        phy_capacity_mbps = self._calculate_instant_capacity(actual_user_idx, power_val, curr_obs)

        if occ < 0.001:
            efficiency_factor = 0.0
        elif occ < 0.05:
            efficiency_factor = 0.5
        else:
            efficiency_factor = 1.0

        # 切片权重
        user_slice = self._get_user_slice(actual_user_idx, slice_ue_assoc)
        slice_weight = 1.0

        if user_slice is not None:
            slice_weight += slice_priority[user_slice] * PRIORITY_GAIN
            s_drift = old_drift[user_slice]
            valid_mask = s_drift > -1.5
            if np.any(valid_mask) and np.min(s_drift[valid_mask]) < 0:
                slice_weight *= 1.5

        # 过程奖励
        base_reward = np.log2(phy_capacity_mbps + 1.0) * efficiency_factor * PHY_SCORE_SCALE * slice_weight

        power_incentive = POWER_INCENTIVE_MAP.get(power_level, 0.0)
        if occ > 0.05:
            base_reward += power_incentive
        else:
            if power_level < 3:
                base_reward += power_incentive * 1.5

        step_reward = base_reward

        # ==================== 🔥 核心修复：单位转换 ====================
        extreme_latency_detected = False

        if user_slice is not None:
            slice_users = (slice_ue_assoc[user_slice] == 1).nonzero()[0]
            if len(slice_users) > 0:
                # [FIX] Convert steps to seconds
                avg_latency_steps = np.mean(curr_obs["buffer_latencies"][slice_users])
                avg_latency_sec = avg_latency_steps * self.tti_duration

                avg_buffer = np.mean(curr_obs["buffer_occupancies"][slice_users])
                sla_penalty = 0.0

                # Target is in ms (e.g., 100ms) -> convert to seconds
                target_latency_ms = slice_req.get(f"slice_{user_slice}", {}).get("ues", {}).get("buffer_latency", 100)
                target_latency_seconds = (target_latency_ms / 1000.0) * latency_tolerance_factor

                if avg_latency_sec > target_latency_seconds:
                    ratio = avg_latency_sec / target_latency_seconds

                    # ✅ 分段线性惩罚（避免指数爆炸）
                    if ratio <= 2.0:
                        sla_penalty = -3.0 * (ratio - 1.0)
                    elif ratio <= 5.0:
                        sla_penalty = -6.0 - 3.0 * (ratio - 2.0)
                    elif ratio <= 10.0:
                        sla_penalty = -15.0 - 2.0 * (ratio - 5.0)
                    else:
                        sla_penalty = -25.0 - 1.0 * min(ratio - 10.0, 5.0)

                    sla_penalty = max(sla_penalty, -MAX_LATENCY_PENALTY_PER_STEP)

                    # ✅ 极端延迟检测
                    if avg_latency_sec > EXTREME_LATENCY_THRESHOLD:
                        extreme_latency_detected = True
                        if np.random.rand() < 0.01:
                            print(
                                f"[EXTREME LAT] Slice {user_slice}: {avg_latency_sec * 1000:.0f}ms > {EXTREME_LATENCY_THRESHOLD}s!")

                    # 调试打印（0.5% 概率）
                    if np.random.rand() < 0.005:
                        print(f"[LAT] Slice {user_slice}: "
                              f"Actual={avg_latency_sec * 1000:.1f}ms, "
                              f"Target={target_latency_seconds * 1000:.1f}ms, "
                              f"Ratio={ratio:.1f}x, "
                              f"Penalty={sla_penalty:.2f}")

                # Buffer 惩罚
                if avg_buffer > 0.7:
                    buffer_penalty = -2.0 * (avg_buffer - 0.7)
                    sla_penalty += max(buffer_penalty, -5.0)

                step_reward += sla_penalty

        # ✅ 数值安全检查（防止 NaN 和 Inf）
        if not np.isfinite(step_reward):
            print(f"[NAN/INF DETECTED] step_reward={step_reward}, 强制设为 -30.0")
            step_reward = -30.0

        # ==================== 终端审计 ====================
        is_final_step = (self.env_state.step_number >= self.max_number_rbs)
        final_reward = step_reward

        if is_final_step:
            active_slices_indices = bs_slice_assoc.nonzero()[0]
            violation_found = False
            violation_details = []
            audit_penalty = 0.0

            for s_idx in active_slices_indices:
                s_drift = new_drift[s_idx]
                valid_mask = s_drift > -1.5
                if not np.any(valid_mask):
                    continue

                min_drift = np.min(s_drift[valid_mask])
                slice_users = (slice_ue_assoc[s_idx] == 1).nonzero()[0]
                slice_users = (slice_ue_assoc[s_idx] == 1).nonzero()[0]
                avg_buff = np.mean(curr_obs["buffer_occupancies"][slice_users]) if len(slice_users) > 0 else 0.0

                # [FIX] Unit conversion
                avg_latency_steps = np.mean(curr_obs["buffer_latencies"][slice_users]) if len(slice_users) > 0 else 0.0
                avg_latency_sec = avg_latency_steps * self.tti_duration

                violates = (min_drift < drift_violation_threshold) and (avg_buff >= EMPTY_BUFFER_THRESHOLD)

                # 延迟超过 5 秒也算违约
                if avg_latency_sec > 5.0:
                    violates = True

                violation_details.append({
                    'slice': s_idx,
                    'min_drift': min_drift,
                    'avg_buffer': avg_buff,
                    'avg_latency_ms': avg_latency_sec * 1000,  # Corrected for display
                    'violates': violates
                })

                if violates:
                    violation_found = True
                    p_weight = 1.0 + slice_priority[s_idx]
                    audit_penalty += abs(min_drift) * TERMINAL_PENALTY_FACTOR * p_weight

            audit_penalty = min(audit_penalty, MAX_AUDIT_PENALTY)

            # 极端延迟额外惩罚
            if extreme_latency_detected:
                audit_penalty = min(audit_penalty + 5.0, MAX_AUDIT_PENALTY + 5.0)

            # ✅ 打印（每 10 个 Episode 且只打印一次）
            if self.episode_count % 10 == 0 and not hasattr(self, '_audit_printed'):
                self._audit_printed = True  # 标记已打印

                print(f"\n{'─' * 80}")
                print(f"⚖️  [SLA AUDIT] Episode {self.episode_count}")
                print(
                    f"  阶段: {'预热' if self.episode_count < WARMUP_EPISODES else ('过渡' if self.episode_count < TRANSITION_EPISODES else '正常')}")
                print(f"  延迟容忍: {latency_tolerance_factor:.1f}x | 违约阈值: {drift_violation_threshold:.1f}")
                print(f"{'─' * 80}")
                for v in violation_details:
                    status = "❌ FAIL" if v['violates'] else "✅ PASS"
                    print(f"  Slice {v['slice']}: {status} | "
                          f"Drift={v['min_drift']:+.3f}, "
                          f"Buffer={v['avg_buffer']:.2%}, "
                          f"Latency={v['avg_latency_ms']:.1f}ms")
                print(f"{'─' * 80}")
                print(f"  Violation: {violation_found} | Accumulated: {self.episode_accumulated_reward:.2f}")

            if violation_found:
                past_accumulated = self.episode_accumulated_reward
                abs_accumulated = abs(past_accumulated)
                clawback_amount = min(abs_accumulated * CLAWBACK_RATE, MAX_CLAWBACK)
                final_reward = -clawback_amount - audit_penalty

                if self.episode_count % 10 == 0 and hasattr(self, '_audit_printed'):
                    print(f"  ├─ Accumulated: {past_accumulated:.2f}")
                    print(f"  ├─ Clawback: -{clawback_amount:.2f} | Audit: -{audit_penalty:.2f}")
                    print(f"  └─ Terminal: {final_reward:.2f}")
                    print(f"{'─' * 80}\n")
            else:
                final_reward += 15.0

                if self.episode_count % 10 == 0 and hasattr(self, '_audit_printed'):
                    print(f"  ✅ No Violations | Terminal: {final_reward:.2f}")
                    print(f"{'─' * 80}\n")

        if not is_final_step:
            self.episode_accumulated_reward += final_reward

            # ✅ 数值溢出保护（累积奖励也需要上限）
            if self.episode_accumulated_reward < -10000:
                print(f"[OVERFLOW PROTECTION] Accumulated={self.episode_accumulated_reward:.2f}, 重置为 -1000")
                self.episode_accumulated_reward = -1000.0

        # ✅ 最终 Clipping（终极保护）
        final_reward_safe = np.clip(final_reward, -100.0, 20.0)

        # ✅ 安全检查
        if not np.isfinite(final_reward_safe):
            print(f"[SAFETY] final_reward={final_reward} is not finite, 强制设为 -50.0")
            final_reward_safe = -50.0

        return final_reward_safe

    # Baseline Rewards (Compressed)
    def _compute_immediate_reward_sparse(self, *args, **kwargs):
        return 0.0

    def _extract_metrics_indices(self, slice_req, s_idx):
        s_conf = slice_req.get(f'slice_{s_idx}', {})
        if not s_conf or 'parameters' not in s_conf: return []
        m_map = {"throughput": 0, "reliability": 1, "latency": 2}
        return [m_map[p['name']] for p in s_conf['parameters'].values() if p['name'] in m_map]

    def _compute_immediate_reward_mean(self, old, new, action, s_ue, s_prio, bs_slice, s_req, **kwargs):
        if action[0] == 0: return 0.0
        u_idx = action[0] - 1
        s_idx = self._get_user_slice(u_idx, s_ue)
        if s_idx is None or bs_slice[s_idx] == 0: return 0.0

        idxs = self._extract_metrics_indices(s_req, s_idx)
        if not idxs: return -0.05

        s_users = s_ue[s_idx].nonzero()[0]
        old_d, new_d = old[s_idx, :len(s_users)], new[s_idx, :len(s_users)]
        diff = np.mean(new_d - old_d, axis=0)[idxs]

        rew = np.mean(diff) * (1.0 + 0.5 * (s_prio[s_idx] if s_idx < len(s_prio) else 0))
        if np.min(new_d[:, idxs]) > 0.7: rew *= 0.5
        return np.clip(rew, -1.0, 2.0)

    def _compute_immediate_reward_positive_sum(self, old, new, action, s_ue, s_prio, bs_slice, s_req, **kwargs):
        if action[0] == 0: return 0.0
        u_idx = action[0] - 1
        s_idx = self._get_user_slice(u_idx, s_ue)
        if s_idx is None or bs_slice[s_idx] == 0: return 0.0

        idxs = self._extract_metrics_indices(s_req, s_idx)
        if not idxs: return -0.05

        s_users = s_ue[s_idx].nonzero()[0]
        diff = np.mean(new[s_idx, :len(s_users)] - old[s_idx, :len(s_users)], axis=0)[idxs]

        pos = diff[diff > 0]
        base = np.sum(pos) if len(pos) > 0 else (np.sum(diff[diff < 0]) if np.any(diff < 0) else 0.0)

        rew = base * (1.0 + 0.5 * (s_prio[s_idx] if s_idx < len(s_prio) else 0))
        if np.min(new[s_idx, :len(s_users)][:, idxs]) > 0.7: rew *= 0.5
        return np.clip(rew, -1.0, 2.0)

    def _compute_immediate_reward_weighted(self, old, new, action, s_ue, s_prio, bs_slice, s_req, **kwargs):
        if action[0] == 0: return 0.0
        u_idx = action[0] - 1
        s_idx = self._get_user_slice(u_idx, s_ue)
        if s_idx is None or bs_slice[s_idx] == 0: return 0.0

        s_conf = s_req.get(f'slice_{s_idx}', {})
        if not s_conf or 'parameters' not in s_conf: return -0.05

        s_users = s_ue[s_idx].nonzero()[0]
        old_d, new_d = old[s_idx, :len(s_users)], new[s_idx, :len(s_users)]
        m_map = {"throughput": 0, "reliability": 1, "latency": 2}

        idxs, weights = [], []
        for p in s_conf['parameters'].values():
            if p['name'] in m_map:
                idx = m_map[p['name']]
                idxs.append(idx)
                w = np.min(old_d[:, idx])
                weights.append(min(abs(w), 1.0) if w < 0 else (0.5 if w < 0.3 else (0.3 if w < 0.7 else 0.1)))

        if not idxs: return 0.0

        diff = np.mean(new_d - old_d, axis=0)[idxs]
        weights = np.array(weights)

        pos_mask = diff > 0
        base = np.sum(diff[pos_mask] * weights[pos_mask]) if np.any(pos_mask) else np.sum(
            diff[diff < 0] * weights[diff < 0])

        rew = base * (1.0 + 0.5 * (s_prio[s_idx] if s_idx < len(s_prio) else 0))
        if np.min(new_d[:, idxs]) > 0.7: rew *= 0.5
        return np.clip(rew, -1.0, 2.0)