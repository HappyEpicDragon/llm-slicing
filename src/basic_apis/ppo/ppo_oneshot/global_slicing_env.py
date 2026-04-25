import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
from hydra.utils import get_class

from src.basic_apis.network_slicing_business.network_slicing_business_executor \
    import ComponentConfig, ComponentClasses, ComponentFactory, NetworkSlicingBusinessExecutor
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.ppo.utils import calculate_reward_no_mask, intent_drift_calc

from src.diagnostic_utils.diagnostic_decision_tracking import DecisionTracker
from src.diagnostic_utils.diagnostic_reward_action_causal import RewardActionAnalyzer


class GlobalSlicingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, env_settings, np_random, path_context: PathContext):
        super().__init__()
        self.config = env_settings
        self.np_random = np_random
        self.path_context = path_context

        # === 1. 初始化业务组件 ===
        self.mode = self.config.mode
        self.scenario_mode = self.config.scenario_mode
        self.components_config = ComponentConfig(self.config.components)
        component_classes = ComponentClasses(
            ChannelClass=get_class(self.config.components.channel[self.scenario_mode].class_path),
            AssociationClass=get_class(self.config.components.association[self.scenario_mode].class_path),
            TrafficClass=get_class(self.config.components.traffic.class_path),
            MobilityClass=get_class(self.config.components.mobility.class_path)
        )
        self.component_factory = ComponentFactory(
            self.components_config, component_classes, self.np_random, self.path_context
        )
        self.business_executor: Optional[NetworkSlicingBusinessExecutor] = None
        self.components = None

        # === 2. 定义环境常量 ===
        self.num_rbgs = 27
        self.max_users = self.components_config.ue_config.max_number_ues
        self.max_ues_per_slice = 5
        self.max_bs_power = self.config.components.basestations.total_power
        self.num_slices = self.components_config.slice_config.max_number_slices
        self.num_phys_rbs = int(self.components_config.basestation_config.num_available_rbs[0])
        self.rbg_size = self.num_phys_rbs // self.num_rbgs
        self.num_power_levels = 1

        # === 3. 定义动作空间 ===
        # [User_ID] * 27 RBGs. 值域: 0 (Wait), 1~N (User)
        nvec = [self.max_users + 1] * self.num_rbgs
        self.action_space = spaces.MultiDiscrete(nvec)
        self.max_users_per_slice = 5

        # === 4. 定义观测空间（简化版）===
        self.observation_space = spaces.Dict({
            # 全局信息 (3维)
            'timestep': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'step_in_episode': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),

            # CSI简化：只保留统计量（75维：25用户×3统计量）
            'csi_mean_per_user': spaces.Box(low=-1, high=1, shape=(self.max_users,), dtype=np.float32),
            'csi_max_per_user': spaces.Box(low=-1, high=1, shape=(self.max_users,), dtype=np.float32),
            'csi_std_per_user': spaces.Box(low=0, high=2, shape=(self.max_users,), dtype=np.float32),

            # 用户状态 (25维)
            'user_buffer_status': spaces.Box(low=0, high=1, shape=(self.max_users,), dtype=np.float32),

            # 分配历史 (10+50=60维)
            'slice_allocated_power': spaces.Box(low=0, high=1, shape=(1, self.num_slices), dtype=np.float32),
            'slice_allocated_rbs': spaces.Box(low=0, high=1, shape=(1, self.num_slices), dtype=np.float32),
            'user_allocated_rbs': spaces.Box(low=0, high=1, shape=(1, self.max_users), dtype=np.float32),
            'user_allocated_power': spaces.Box(low=0, high=1, shape=(1, self.max_users), dtype=np.float32),

            # Intent Drift（修正：75维 = 5切片 × 5用户/切片 × 3指标）
            'intent_drift': spaces.Box(
                low=-2, high=1,
                shape=(self.num_slices, self.max_users_per_slice, 3),  # (5, 5, 3)
                dtype=np.float32
            ),

            # SLA信息 (30维：5×3×2)
            'sla_target': spaces.Box(low=-1, high=2, shape=(self.num_slices, 3), dtype=np.float32),
            'active_sla': spaces.Box(low=0, high=1, shape=(self.num_slices, 3), dtype=np.float32),

            # 拓扑信息 (5+125+25+5=160维)
            'slice_priority': spaces.Box(low=0, high=1, shape=(self.num_slices,), dtype=np.float32),
            'user_to_slice': spaces.Box(low=0, high=1, shape=(self.num_slices, self.max_users), dtype=np.int32),
            'active_users': spaces.Box(low=0, high=1, shape=(1, self.max_users), dtype=np.int8),
            'active_slices': spaces.Box(low=0, high=1, shape=(1, self.num_slices), dtype=np.int8),
        })

        # === 5. 状态变量初始化 ===
        self.mode_config = self.config[self.scenario_mode][self.mode]
        self.cross_config = self.config[self.scenario_mode]
        self.scenario_list = self.mode_config.active_scenario_list
        self.init_ep = self.mode_config.init_scenario_episode
        self.max_ep = self.mode_config.max_scenario_episodes
        self.skip_step = self.cross_config.episode_cross_scenario_skip
        self.max_timesteps = self.config.env_state_config.max_channel_timesteps

        self.scenario_pointer = 0
        self.internal_episode_ptr = self.init_ep
        self.current_episode_idx = 0
        self.current_timestep = 0
        self.total_episodes = 0

        self.cached_action_mask = None
        self.last_unformatted_obs_deque = deque(maxlen=10)
        self.last_raw_obs = None
        self.last_base_reward = 0.0
        self.last_shaping_reward = 0.0
        self.last_guidance_reward = 0.0  # 新增记录

        # === [新增] PF 算法所需的历史状态 ===
        # 初始化为非零极小值，防止除以零
        self.user_history_rate = np.ones(self.max_users, dtype=np.float32) * 1e-6
        # 遗忘因子 (Alpha): 决定了PF的时间窗口。0.01 代表平均过去 100ms 的速率
        # 这与 Baseline 的参数通常是一致的
        self.pf_alpha = 0.01

        self.decision_tracker = DecisionTracker()
        self.reward_action_analyzer = RewardActionAnalyzer()

    def _compute_expert_action(self):
        """
        [Smart Capped Expert]
        智能专家策略：
        1. 识别 HP 用户。
        2. 计算每个用户的"有效需求" (Capped Demand)：
           - 取 min(实际Buffer积压, 1.5倍SLA速率需求)。
           - 防止专家为了清空积压已久的 Buffer 而耗尽所有 RBG，导致 NHP 饿死。
        3. 对 HP 用户进行公平轮询 (Round Robin)，直到满足"有效需求"或资源耗尽。
        4. 未被使用的 RBG 保持为 0，留给 RL Agent 在 step 函数中填充给 NHP。
        """
        expert_action = np.zeros(self.num_rbgs, dtype=np.int32)
        remaining_rbgs = set(range(self.num_rbgs))

        # ==================================================
        # 1. 数据准备 & 维度对齐
        # ==================================================
        # 获取 CSI: [Users, Phys_RBs] -> 确保转置为 [Users, Phys_RBs] 以便后续切片
        raw_csi = self.last_raw_obs.get('target_cell_power', np.zeros((self.max_users, self.num_phys_rbs)))
        if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
        # 这里的维度检查需要根据实际物理层定义，通常 num_phys_rbs (135) > max_users (25)
        if raw_csi.shape[0] == self.num_phys_rbs:
            raw_csi = raw_csi.T

        # 获取 Buffer 状态
        buf_norm = self.last_raw_obs.get("buffer_occupancies", np.zeros(self.max_users))
        incoming = self.last_raw_obs.get("pkt_incoming_bits", np.zeros(self.max_users))
        ue_assoc = self.components.slices.ue_assoc

        # 获取物理层常数 (用于计算容量)
        max_pkts = self.components.ues.max_buffer_pkts
        pkt_sizes = self.components.ues.pkt_sizes
        bandwidth = self.components.basestations.bandwidths[0]
        tti = 0.001  # 1ms
        noise = 1e-14

        # 计算每个 RBG 的带宽
        bandwidth_per_rbg = bandwidth * (self.rbg_size / self.num_phys_rbs)

        # ==================================================
        # 2. 计算 Capped Demand (限流需求)
        # ==================================================
        current_demand_bits = np.zeros(self.max_users)

        for s_idx in range(self.num_slices):
            # 获取优先级和 SLA
            req = self.components.slices.requirements.get(f'slice_{s_idx}', {})
            prio = req.get('priority', 0)

            # 只处理 HP 切片
            if prio > 0:
                # 获取 SLA 吞吐目标 (Mbps)
                target_mbps = 10.0  # 默认保底 10 Mbps
                if 'parameters' in req:
                    for p in req['parameters'].values():
                        if p.get('name') == 'throughput':
                            target_mbps = p.get('value', 10.0)
                            break

                # 转换为本 TTI 的 Bit 需求
                # 设定上限为 1.5 倍 SLA 速率，留出 50% 余量用于追赶进度
                target_bits_per_tti = target_mbps * 1e6 * tti
                cap_bits = target_bits_per_tti * 1.0

                # 获取该切片下的用户
                u_indices = np.where(ue_assoc[s_idx] > 0)[0]

                for u in u_indices:
                    # 计算实际物理积压 (Bits)
                    real_buffer_bits = buf_norm[u] * max_pkts[u] * pkt_sizes[u] + incoming[u]

                    # [关键] 取 Buffer 和 Cap 的较小值
                    # 这意味着即使 Buffer 积压了 100MB，专家也只会在这一帧传 cap_bits
                    # 剩下的资源就释放给了 NHP
                    current_demand_bits[u] = min(real_buffer_bits, cap_bits)

        # ==================================================
        # 3. 初始化 HP 活跃用户池
        # ==================================================
        hp_active_users = []
        user_satisfied_bits = {}  # 记录已分配容量

        for u in range(self.max_users):
            if current_demand_bits[u] > 1e-9:  # 有有效需求
                hp_active_users.append(u)
                user_satisfied_bits[u] = 0.0

        if not hp_active_users:
            return expert_action

        # ==================================================
        # 4. 公平轮询分配 (Smart Round Robin)
        # ==================================================
        user_pointer = 0

        while remaining_rbgs and hp_active_users:
            # 循环指针
            if user_pointer >= len(hp_active_users):
                user_pointer = 0

            u = hp_active_users[user_pointer]

            # --- [Check 1] 饱和检测 ---
            # 如果已分配容量达到了 Capped Demand，停止给该用户分配
            if user_satisfied_bits[u] >= current_demand_bits[u]:
                hp_active_users.pop(user_pointer)
                # 移除后，指针自动指向下一个元素，无需增加
                continue

                # --- [Check 2] 寻找最佳 RBG ---
            best_rbg = -1
            best_gain = -1.0

            for rbg in list(remaining_rbgs):
                start_rb = rbg * self.rbg_size
                end_rb = start_rb + self.rbg_size

                # 计算该 RBG 的平均信道增益
                # 边界保护
                limit = min(end_rb, raw_csi.shape[1])
                if start_rb < limit:
                    gain = np.mean(raw_csi[u, start_rb:limit])
                else:
                    gain = 0.0

                if gain > best_gain:
                    best_gain = gain
                    best_rbg = rbg

            # --- [Check 3] 执行分配 ---
            # 只要有信道就分配 (移除阈值限制以防死锁)
            if best_rbg != -1 and best_gain > 0:
                expert_action[best_rbg] = u + 1  # 1-based
                remaining_rbgs.remove(best_rbg)

                # 估算获得的容量 (Shannon Capacity)
                # C = B * log2(1 + SNR) * T
                snr = best_gain / noise
                capacity_bits = bandwidth_per_rbg * np.log2(1 + snr) * tti
                user_satisfied_bits[u] += capacity_bits

                user_pointer += 1
            else:
                # 该用户在剩余 RBG 上无信号，移除出队列
                hp_active_users.pop(user_pointer)

        return expert_action

    # =========================================================================
    # [核心修改 2] 奖励函数：Base + Shaping + Guidance
    # =========================================================================
    # def _calculate_reward(self, metrics, agent_action) -> float:
    #     """
    #     真正的Baseline风格Reward
    #     使用实际drift值，而不是二元判断
    #     """
    #
    #     # 获取数据
    #     raw_intent_drift = metrics.get("intent_drift",
    #                                    self.last_unformatted_obs_deque[0]["intent_drift"])
    #     slice_assoc = self.components.slices.ue_assoc
    #     current_slice_req = metrics.get('slice_req', self.components.slices.requirements)
    #
    #     if isinstance(current_slice_req, (list, deque)):
    #         current_slice_req = current_slice_req[-1]
    #
    #     # === 1. 计算每个slice的score（木桶效应）===
    #     slice_scores = []
    #     slice_priorities = []
    #
    #     for s_idx in range(self.num_slices):
    #         req = current_slice_req.get(f'slice_{s_idx}', {})
    #         priority = req.get('priority', 0)
    #
    #         # 检查该slice是否激活
    #         active_users = np.where(slice_assoc[s_idx, :] > 0)[0]
    #         if len(active_users) == 0:
    #             continue  # 跳过没有用户的slice
    #
    #         # 收集该slice的所有有效metrics
    #         valid_drifts = []
    #         valid_limit = min(len(active_users), raw_intent_drift.shape[1])
    #
    #         for u_idx in range(valid_limit):
    #             for m_idx in range(3):  # 3个指标：thr, lat, rel
    #                 drift = raw_intent_drift[s_idx, u_idx, m_idx]
    #                 # 过滤无效值（-2表示该指标不活跃）
    #                 if drift > -1.5:
    #                     valid_drifts.append(drift)
    #
    #         if len(valid_drifts) > 0:
    #             # 木桶效应：取最差的指标
    #             min_drift = np.min(valid_drifts)
    #             slice_scores.append(min_drift)
    #             slice_priorities.append(priority)
    #         else:
    #             # 没有有效指标，认为是满足的
    #             slice_scores.append(0.0)
    #             slice_priorities.append(priority)
    #
    #     # === 2. 聚合reward ===
    #     if len(slice_scores) == 0:
    #         return 0.0
    #
    #     slice_scores = np.array(slice_scores)
    #     slice_priorities = np.array(slice_priorities)
    #
    #     # 2.1 所有slice都满足SLA
    #     if np.all(slice_scores >= 0):
    #         reward = float(np.mean(slice_scores))
    #
    #     # 2.2 有HP违约（严厉惩罚）
    #     elif np.any((slice_scores < 0) & (slice_priorities > 0)):
    #         hp_bad_scores = slice_scores[(slice_scores < 0) & (slice_priorities > 0)]
    #         reward = float(np.mean(hp_bad_scores)) - 1.0  # 额外-1惩罚
    #
    #     # 2.3 只有NHP违约（普通惩罚）
    #     else:
    #         nhp_bad_scores = slice_scores[slice_scores < 0]
    #         if len(nhp_bad_scores) > 0:
    #             reward = float(np.mean(nhp_bad_scores))
    #         else:
    #             reward = 0.0
    #
    #     # === 3. 保存用于日志 ===
    #     self.last_base_reward = reward
    #     self.last_shaping_reward = 0.0
    #     self.last_guidance_reward = 0.0
    #
    #     # === 4. 防御性检查 ===
    #     if not np.isfinite(reward):
    #         print(f"⚠️ NaN Reward detected! Returning 0.0")
    #         return 0.0
    #
    #     return reward

    def _calculate_reward(self, metrics, agent_action) -> float:
        """
        完整的Reward函数

        组成：
        1. Base Reward: Baseline风格的SLA violation惩罚（连续值）
        2. Guidance Reward: 模仿Expert的软约束（逐步衰减）

        Training时：base + guidance
        Testing时：base only
        """

        # =====================================================================
        # 1. 数据准备
        # =====================================================================
        raw_intent_drift = metrics.get("intent_drift",
                                       self.last_unformatted_obs_deque[0]["intent_drift"])
        slice_assoc = self.components.slices.ue_assoc
        current_slice_req = metrics.get('slice_req', self.components.slices.requirements)

        if isinstance(current_slice_req, (list, deque)):
            current_slice_req = current_slice_req[-1]

        # =====================================================================
        # 2. Base Reward: 计算每个Slice的Score（Baseline风格）
        # =====================================================================
        slice_scores = []
        slice_priorities = []

        for s_idx in range(self.num_slices):
            req = current_slice_req.get(f'slice_{s_idx}', {})
            priority = req.get('priority', 0)

            # 检查该slice是否有激活的用户
            active_users = np.where(slice_assoc[s_idx, :] > 0)[0]
            if len(active_users) == 0:
                continue  # 跳过没有用户的slice

            # 收集该slice的所有有效drift值
            valid_drifts = []
            valid_limit = min(len(active_users), raw_intent_drift.shape[1])

            for u_idx in range(valid_limit):
                for m_idx in range(3):  # 3个指标：throughput, latency, reliability
                    drift = raw_intent_drift[s_idx, u_idx, m_idx]
                    # 过滤无效值（-2表示该指标不活跃）
                    if drift > -1.5:
                        valid_drifts.append(drift)

            # 木桶效应：取最差的指标作为该slice的score
            if len(valid_drifts) > 0:
                min_drift = np.min(valid_drifts)
                slice_scores.append(min_drift)
                slice_priorities.append(priority)
            else:
                # 没有有效指标，认为是满足SLA
                slice_scores.append(0.0)
                slice_priorities.append(priority)

        # =====================================================================
        # 3. Base Reward: 聚合所有Slice（Baseline逻辑）
        # =====================================================================
        base_reward = 0.0

        if len(slice_scores) == 0:
            base_reward = 0.0
        else:
            slice_scores = np.array(slice_scores)
            slice_priorities = np.array(slice_priorities)

            # 情况1：所有slice都满足SLA（所有score >= 0）
            if np.all(slice_scores >= 0):
                base_reward = float(np.mean(slice_scores))

            # 情况2：有HP违约（严厉惩罚）
            elif np.any((slice_scores < 0) & (slice_priorities > 0)):
                hp_bad_scores = slice_scores[(slice_scores < 0) & (slice_priorities > 0)]
                # HP违约：平均违约drift - 额外惩罚1.0
                base_reward = float(np.mean(hp_bad_scores)) - 1.0

            # 情况3：只有NHP违约（普通惩罚）
            else:
                nhp_bad_scores = slice_scores[slice_scores < 0]
                if len(nhp_bad_scores) > 0:
                    base_reward = float(np.mean(nhp_bad_scores))
                else:
                    base_reward = 0.0

        # =====================================================================
        # 4. Guidance Reward: 模仿Expert（Training Only，带Annealing）
        # =====================================================================
        guidance_reward = 0.0

        if self.mode == 'training':
            # 4.1 获取Expert的决策
            expert_action = self._compute_expert_action()

            # 4.2 防御性检查
            if expert_action is None or np.isnan(expert_action).any():
                expert_action = np.zeros_like(agent_action)

            # 4.3 计算匹配率（有多少个RBG的决策和Expert一致）
            match_rate = float(np.mean(agent_action == expert_action))

            # 4.4 Annealing Schedule
            curr_ep = self.total_episodes

            if curr_ep < 20:
                # 阶段1 (0-50 episodes): 强模仿
                guidance_scale = 10.0
            elif curr_ep < 70:
                # 阶段2 (50-150 episodes): 线性衰减
                progress = (curr_ep - 50) / 100.0
                guidance_scale = 10.0 * (1.0 - progress)  # 10.0 → 0
            else:
                # 阶段3 (150+ episodes): 完全自主
                guidance_scale = 0.0

            guidance_reward = match_rate * guidance_scale

            # 4.5 调试日志（每200步打印一次）
            if self.current_timestep % 200 == 0 and guidance_scale > 0:
                print(f"\n[Guidance Debug]")
                print(f"  Episode: {curr_ep}")
                print(f"  Guidance Scale: {guidance_scale:.2f}")
                print(f"  Match Rate: {match_rate:.3f} ({int(match_rate * self.num_rbgs)}/{self.num_rbgs} RBGs)")
                print(f"  Guidance Reward: {guidance_reward:.3f}")

        # =====================================================================
        # 5. 保存各部分Reward（用于日志和监控）
        # =====================================================================
        self.last_base_reward = base_reward
        self.last_guidance_reward = guidance_reward
        self.last_shaping_reward = 0.0  # 暂时不用shaping

        # =====================================================================
        # 6. 计算总Reward并返回
        # =====================================================================
        total_reward = base_reward + guidance_reward

        # 6.1 防御性检查：避免NaN/Inf
        if not np.isfinite(total_reward):
            print(f"⚠️ NaN/Inf Reward detected!")
            print(f"  Base: {base_reward}")
            print(f"  Guidance: {guidance_reward}")
            print(f"  Slice Scores: {slice_scores if len(slice_scores) > 0 else 'empty'}")
            return 0.0

        # 6.2 额外调试：每100步打印详细reward分解
        if self.current_timestep % 200 == 0:
            print(f"\n[Reward Breakdown @ Step {self.current_timestep}]")
            print(f"  Base Reward:     {base_reward:+.3f}")
            print(f"  Guidance Reward: {guidance_reward:+.3f}")
            print(f"  Total Reward:    {total_reward:+.3f}")
            if len(slice_scores) > 0:
                print(f"  Slice Scores:    {[f'{s:.2f}' for s in slice_scores]}")
                print(f"  Slice Priorities: {slice_priorities.tolist()}")

        return total_reward

    # =========================================================================
    # Step & Reset
    # =========================================================================
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # === [诊断3] Episode汇总 ===
        if self.current_timestep > 0:  # 不是第一个episode
            print(f"\n{'=' * 80}")
            print(f"Episode {self.current_episode_idx} Summary:")
            print(f"  Last Base Reward: {self.last_base_reward:.3f}")
            print(f"  Last Guidance Reward: {self.last_guidance_reward:.3f}")
            print(f"{'=' * 80}\n")

        current_scenario_id = self.scenario_list[self.scenario_pointer]
        scenario_base = current_scenario_id * self.skip_step
        self.current_episode_idx = scenario_base + self.internal_episode_ptr

        self.internal_episode_ptr += 1
        self.total_episodes += 1
        if self.internal_episode_ptr >= self.max_ep:
            self.internal_episode_ptr = self.init_ep
            self.scenario_pointer = (self.scenario_pointer + 1) % len(self.scenario_list)

        self.current_timestep = 0
        self.last_unformatted_obs_deque.clear()
        self.last_raw_obs = None

        self.components = self.component_factory.create_scenario_components(
            episode_number=self.current_episode_idx, step_number=0
        )
        self.business_executor = NetworkSlicingBusinessExecutor(
            self.components, self.components_config
        )
        self.components.metrics.reset()
        if hasattr(self.components.ues, 'buffer_state'):
            self.components.ues.buffer_state.fill(0)

        self.cached_action_mask = self._compute_action_masks()
        self._prefill_observation()

        # if self.mode in ['testing', 'evaluating']:
        #     print(f"\n💾 Saving diagnostic data for episode {self.current_episode_idx}...")
        #
        #     self.decision_tracker.save_episode(self.current_episode_idx)
        #     self.reward_action_analyzer.analyze_episode(self.current_episode_idx)
        #
        #     print(f"✅ Diagnostic data saved\n")

        return self._get_observation(), {}

    # =========================================================================
    # 2. Step Function (混合执行版)
    # =========================================================================
    def step(self, action: np.ndarray):
        final_action = action.copy()  # 初始：Agent 的动作

        # ==========================================================
        # [Physics Execution]
        # ==========================================================
        user_ids = final_action

        raw_powers = np.ones(self.num_rbgs, dtype=np.float32)
        active_mask = (user_ids > 0)
        requested_powers = raw_powers * active_mask
        total_requested_power = np.sum(requested_powers)

        if total_requested_power > 1e-9:
            scale_factor = self.max_bs_power / total_requested_power
        else:
            scale_factor = 0.0

        actual_powers = requested_powers * scale_factor

        self.business_executor.temp_rb_allocation.fill(0.0)
        self.business_executor.temp_rb_ues_association.fill(0.0)

        for rbg_idx in range(self.num_rbgs):
            u_id = user_ids[rbg_idx]
            if u_id > 0:
                u_idx = u_id - 1
                start_rb = rbg_idx * self.rbg_size
                end_rb = start_rb + self.rbg_size

                self.business_executor.temp_rb_allocation[u_idx, start_rb:end_rb] = actual_powers[rbg_idx]
                self.business_executor.temp_rb_ues_association[0, u_idx, start_rb:end_rb] = 1

        full_metrics_data = self.business_executor.execute_tti_physics(
            channel_timestep=self.current_timestep,
            episode_number=self.current_episode_idx
        )

        self.last_unformatted_obs_deque.appendleft(full_metrics_data)
        intent_drift = intent_drift_calc(
            self.last_unformatted_obs_deque,
            max_number_ues_slice=self.max_ues_per_slice,
            intent_overfulfillment_rate=0.2
        )
        full_metrics_data["intent_drift"] = intent_drift
        self._update_priority_in_obs(full_metrics_data)
        self.last_raw_obs = full_metrics_data
        self.components.metrics.step(full_metrics_data)

        reward = self._calculate_reward(full_metrics_data, action)

        self.current_timestep += 1
        truncated = (self.current_timestep >= self.max_timesteps)
        terminated = False
        info = self._build_info_dict(full_metrics_data, scale_factor)
        # [🔥🔥🔥 新增 🔥🔥🔥] 将最终动作注入 Info，以便采集脚本获取
        info["final_executed_action"] = final_action.copy()

        obs = self._get_observation()

        if self.mode == 'testing' and (terminated or truncated):
            self._handle_testing_save()



        return obs, reward, terminated, truncated, info

    # ... (其余辅助函数 _prefill_observation, _get_observation, _compute_action_masks 等保持你原有的逻辑不变) ...
    # 注意：_get_observation 需要确保包含 user_buffer_status (上一轮已提供)
    def _prefill_observation(self):
        # ... (同上一轮代码) ...
        # 确保 last_raw_obs 初始化完整
        mobilities = self.components.mobility.step(0, self.current_episode_idx)
        dummy_alloc = np.zeros((self.max_users, self.num_phys_rbs))
        spectral_efficiencies, target_cell_power = self.components.channel.step(0, self.current_episode_idx,
                                                                                dummy_alloc)
        traffics = self.components.traffic.step(self.components.slices.ue_assoc, self.components.slices.requirements, 0,
                                                self.current_episode_idx)
        step_metrics = self.components.ues.step(np.zeros((1, self.max_users, self.num_phys_rbs)), traffics,
                                                spectral_efficiencies, self.components.basestations.bandwidths,
                                                self.components_config.basestation_config.num_available_rbs,
                                                update_state=False)

        self.last_raw_obs = {
            **step_metrics,
            "mobility": mobilities,
            "spectral_efficiencies": spectral_efficiencies,
            "sched_decision": dummy_alloc,
            "target_cell_power": target_cell_power,
            "basestation_ue_assoc": self.components.basestations.ue_assoc,
            "basestation_slice_assoc": self.components.basestations.slice_assoc,
            "slice_ue_assoc": self.components.slices.ue_assoc,
            "slice_req": self.components.slices.requirements,
            "intent_drift": np.zeros((self.num_slices, self.max_ues_per_slice, 3), dtype=np.float32)
        }
        self._update_priority_in_obs(self.last_raw_obs)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        简化版观测空间
        总维度：~400维（vs 之前的1000+维）
        """
        if self.last_raw_obs is None:
            return self._get_dummy_obs()

        raw = self.last_raw_obs

        # =========================================================
        # 1. CSI简化处理（保持不变）
        # =========================================================
        csi_raw = raw.get("target_cell_power", np.zeros((self.max_users, self.num_phys_rbs)))
        if csi_raw.ndim > 2:
            csi_raw = np.squeeze(csi_raw)

        if csi_raw.shape[0] == self.num_phys_rbs and csi_raw.shape[1] == self.max_users:
            csi_raw = csi_raw.T

        rbs_per_rbg = self.num_phys_rbs // self.num_rbgs
        if csi_raw.shape == (self.max_users, self.num_phys_rbs):
            csi_rbg = csi_raw.reshape(self.max_users, self.num_rbgs, rbs_per_rbg).mean(axis=2)
        else:
            csi_rbg = np.zeros((self.max_users, self.num_rbgs), dtype=np.float32)

        csi_safe = csi_rbg + 1e-33
        csi_dbm = 10 * np.log10(csi_safe)
        csi_norm = np.clip((csi_dbm + 150.0) / 60.0 - 1.0, -1.0, 1.0)

        csi_mean_per_user = csi_norm.mean(axis=1)
        csi_max_per_user = csi_norm.max(axis=1)
        csi_std_per_user = csi_norm.std(axis=1)

        # =========================================================
        # 2. 其他观测（保持不变）
        # =========================================================
        rb_alloc = raw["sched_decision"]
        user_pwr = np.sum(rb_alloc, axis=1).reshape(1, -1)
        user_rbs = np.sum(rb_alloc > 0, axis=1).reshape(1, -1)
        slice_assoc = raw["slice_ue_assoc"]
        slice_pwr = (slice_assoc @ user_pwr.T).T
        slice_rbs = (slice_assoc @ user_rbs.T).T

        norm_user_pwr = user_pwr / (self.max_bs_power + 1e-6)
        norm_user_rbs = user_rbs / self.num_phys_rbs
        norm_slice_pwr = slice_pwr / (self.max_bs_power + 1e-6)
        norm_slice_rbs = slice_rbs / self.num_phys_rbs

        sla_target_norm, active_sla = self._extract_sla_info_normalized(raw["slice_req"])
        user_buffer_status = raw.get("buffer_occupancies", np.zeros(self.max_users, dtype=np.float32))

        # =========================================================
        # 3. Intent Drift（直接使用，应该已经是(5,5,3)）
        # =========================================================
        raw_intent_drift = raw.get("intent_drift",
                                   np.zeros((self.num_slices, self.max_users_per_slice, 3), dtype=np.float32))

        # 防御性检查
        if raw_intent_drift.shape != (self.num_slices, self.max_users_per_slice, 3):
            print(
                f"⚠️ Warning: intent_drift shape mismatch. Got {raw_intent_drift.shape}, expected {(self.num_slices, self.max_users_per_slice, 3)}")
            corrected_drift = np.zeros((self.num_slices, self.max_users_per_slice, 3), dtype=np.float32)
            min_slices = min(raw_intent_drift.shape[0], self.num_slices)
            min_users = min(raw_intent_drift.shape[1], self.max_users_per_slice)
            min_metrics = min(raw_intent_drift.shape[2], 3)
            corrected_drift[:min_slices, :min_users, :min_metrics] = raw_intent_drift[
                :min_slices, :min_users, :min_metrics]
            raw_intent_drift = corrected_drift

        # =========================================================
        # 4. 返回观测
        # =========================================================
        return {
            # 全局信息 (3维)
            'timestep': np.array([self.current_timestep / self.max_timesteps], dtype=np.float32),
            'step_in_episode': np.array([self.current_timestep / 1000.0], dtype=np.float32),

            # CSI统计量 (75维：3×25)
            'csi_mean_per_user': csi_mean_per_user.astype(np.float32),
            'csi_max_per_user': csi_max_per_user.astype(np.float32),
            'csi_std_per_user': csi_std_per_user.astype(np.float32),

            # 用户状态 (25维)
            'user_buffer_status': user_buffer_status.astype(np.float32),

            # 分配历史 (60维)
            'slice_allocated_power': norm_slice_pwr.astype(np.float32),
            'slice_allocated_rbs': norm_slice_rbs.astype(np.float32),
            'user_allocated_rbs': norm_user_rbs.astype(np.float32),
            'user_allocated_power': norm_user_pwr.astype(np.float32),

            # Intent Drift（75维：5×5×3）
            'intent_drift': np.clip(raw_intent_drift, -2.0, 1.0).astype(np.float32),

            # SLA信息 (30维)
            'sla_target': sla_target_norm.astype(np.float32),
            'active_sla': active_sla.astype(np.float32),

            # 拓扑信息 (160维)
            'slice_priority': raw["slice_priority"].astype(np.float32),
            'user_to_slice': raw["slice_ue_assoc"].astype(np.int32),
            'active_users': raw["basestation_ue_assoc"].reshape(1, -1).astype(np.int8),
            'active_slices': raw["basestation_slice_assoc"].reshape(1, -1).astype(np.int8),
        }

    # ... 其他辅助函数如 _extract_sla_info_normalized, _update_priority_in_obs, _build_info_dict, _handle_testing_save, _get_dummy_obs, action_masks 保持不变 ...
    def _extract_sla_info_normalized(self, slice_req: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取并归一化SLA目标信息

        Returns:
            sla_target: (num_slices, 3) - 归一化的SLA目标
            active_sla: (num_slices, 3) - 哪些SLA指标是活跃的
        """
        sla_target = np.zeros((self.num_slices, 3), dtype=np.float32)
        active_sla = np.zeros((self.num_slices, 3), dtype=np.float32)

        for s_idx in range(self.num_slices):
            slice_key = f'slice_{s_idx}'
            if slice_key in slice_req:
                req = slice_req[slice_key]

                # Throughput (Mbps)
                if 'throughput' in req:
                    sla_target[s_idx, 0] = req['throughput'] / 100.0  # 归一化
                    active_sla[s_idx, 0] = 1.0

                # Latency (ms)
                if 'latency' in req:
                    sla_target[s_idx, 1] = req['latency'] / 100.0  # 归一化
                    active_sla[s_idx, 1] = 1.0

                # Reliability
                if 'reliability' in req:
                    sla_target[s_idx, 2] = req['reliability']
                    active_sla[s_idx, 2] = 1.0

        return sla_target, active_sla

    def _update_priority_in_obs(self, raw_obs):
        prio = np.zeros(self.num_slices, dtype=np.float32)
        for i in range(self.num_slices): prio[i] = raw_obs["slice_req"].get(f'slice_{i}', {}).get('priority', 0)
        raw_obs["slice_priority"] = prio

    def _build_info_dict(self, metrics, scale_factor) -> Dict:
        """构建返回的 info 字典，包含详细的 SLA 违约信息和统计数据"""
        # 1. 基础信息
        info = {
            "power_scale_factor": scale_factor,
            "throughput": np.sum(metrics['pkt_effective_thr']),
            "reward/sla_base": self.last_base_reward,
            "reward/shaping": self.last_shaping_reward,
            "reward/guidance": self.last_guidance_reward,
        }

        # 2. 流量统计 (Mbps)
        traffics_1ms = metrics["pkt_incoming_bits"]
        ue_assoc = self.components.slices.ue_assoc
        for s_idx in range(self.num_slices):
            u_indices = np.where(ue_assoc[s_idx] > 0)[0]
            if len(u_indices) > 0:
                bits_arrived = np.sum(traffics_1ms[u_indices])
                info[f"arrival/slice_{s_idx}"] = (bits_arrived * 1000) / 1e6
            else:
                info[f"arrival/slice_{s_idx}"] = 0.0

        # 3. 详细违约信息 (Drift & Violation)
        info.update(self._extract_detailed_violation_info(metrics))

        # 4. 元数据 (Active Slices & SLA Req Masks)
        # [核心修改] 显式注入 Priority 信息
        active_slices_mask = self.components.basestations.slice_assoc[0]
        _, active_sla_mask = self._extract_sla_info(metrics["slice_req"])
        metric_names = ['thr', 'rel', 'lat']

        for s_idx in range(self.num_slices):
            info[f"meta/slice_{s_idx}_active"] = int(active_slices_mask[s_idx])

            # [新增] 获取并写入优先级
            prio = metrics["slice_req"].get(f'slice_{s_idx}', {}).get('priority', 0)
            info[f"meta/slice_{s_idx}_priority"] = int(prio)

            for m_idx, m_name in enumerate(metric_names):
                info[f"meta/slice_{s_idx}_{m_name}_req"] = int(active_sla_mask[s_idx, m_idx])

        return info

    def _extract_detailed_violation_info(self, metrics):
        # Copy from previous turn
        drift = metrics.get("intent_drift");
        details = {}
        if drift is None: return {}
        metric_names = ['thr', 'rel', 'lat']
        for s_idx in range(self.num_slices):
            slice_drift = drift[s_idx];
            valid_mask = slice_drift > -1.5
            for m_idx, m_name in enumerate(metric_names):
                valid_vals = slice_drift[:, m_idx][valid_mask[:, m_idx]]
                key = f"slice_{s_idx}_{m_name}"
                if len(valid_vals) > 0:
                    min_val = np.min(valid_vals)
                    details[f"drift/{key}"] = min_val
                    details[f"violation/{key}"] = 1.0 if min_val < 0 else 0.0
                else:
                    details[f"drift/{key}"] = 0.0; details[f"violation/{key}"] = 0.0
        return details

    def _extract_sla_info(self, slice_req):
        # Copy from previous turn
        sla_target = np.zeros((self.num_slices, 3), dtype=np.float32)
        active_sla = np.zeros((self.num_slices, 3), dtype=np.float32)
        metric_map = {'throughput': 0, 'reliability': 1, 'latency': 2}
        for s_idx in range(self.num_slices):
            req = slice_req.get(f'slice_{s_idx}', {})
            if not req or 'parameters' not in req: continue
            for param in req['parameters'].values():
                m_name = param.get('name', '').lower()
                if m_name in metric_map:
                    idx = metric_map[m_name]
                    sla_target[s_idx, idx] = param.get('value', 0.0)
                    active_sla[s_idx, idx] = 1.0
        return sla_target, active_sla

    def _handle_testing_save(self):
        curr_ep = self.current_episode_idx
        self.business_executor.save_metric(self.config.model_name, f"ep_{curr_ep}.npz")
        if hasattr(self.business_executor, 'reset_metric'): self.business_executor.reset_metric()

    def action_masks(self) -> List[bool]:
        if self.cached_action_mask is None: self.cached_action_mask = self._compute_action_masks()
        return self.cached_action_mask

    def _compute_action_masks(self) -> List[bool]:
        # Copy from previous turn (Traffic Aware Masking)
        assoc = self.components.slices.ue_assoc
        connected_indices = set(np.nonzero(np.sum(assoc, axis=0))[0].tolist()) if assoc is not None else set()
        if self.last_raw_obs is not None:
            buffer_occ = self.last_raw_obs.get("buffer_occupancies", np.zeros(self.max_users))
            incoming_bits = self.last_raw_obs.get("pkt_incoming_bits", np.zeros(self.max_users))
            incoming_pkts = self.last_raw_obs.get("pkt_incoming", np.zeros(self.max_users))
            has_traffic_mask = (buffer_occ > 1e-9) | (incoming_bits > 1e-9) | (incoming_pkts > 0)
        else:
            has_traffic_mask = np.ones(self.max_users, dtype=bool)
        valid_user_flags = []
        for i in range(self.max_users):
            is_valid = (i in connected_indices) and has_traffic_mask[i]
            valid_user_flags.append(is_valid)
        rbg_mask = [True] + valid_user_flags
        return rbg_mask * self.num_rbgs

    def _extract_violations_from_metrics(self, metrics: dict) -> List[Dict]:
        """
        从metrics中提取violations列表

        Returns:
            List of violations: [
                {'slice': int, 'user': int, 'metric': str, 'drift': float, 'severity': str},
                ...
            ]
        """
        violations = []
        intent_drift = metrics.get('intent_drift')

        if intent_drift is None:
            return violations

        for s_idx in range(intent_drift.shape[0]):
            for u_idx in range(intent_drift.shape[1]):
                for m_idx, m_name in enumerate(['throughput', 'reliability', 'latency']):
                    drift = intent_drift[s_idx, u_idx, m_idx]

                    # 只关心有效的drift值
                    if drift <= -1.5:  # -2表示无效
                        continue

                    # 检测违约
                    if drift < 0:
                        if drift < -0.5:
                            severity = 'critical'
                        elif drift < -0.2:
                            severity = 'moderate'
                        else:
                            severity = 'minor'

                        violations.append({
                            'slice': int(s_idx),
                            'user': int(u_idx),
                            'metric': m_name,
                            'drift': float(drift),
                            'severity': severity
                        })

        return violations

    def _get_dummy_obs(self) -> Dict[str, np.ndarray]:
        """
        返回dummy观测，用于环境初始化
        确保所有维度与observation_space定义完全一致
        """
        return {
            # 全局信息 (3维)
            'timestep': np.zeros((1,), dtype=np.float32),
            'step_in_episode': np.zeros((1,), dtype=np.float32),

            # CSI统计量 (75维：3×25)
            'csi_mean_per_user': np.zeros((self.max_users,), dtype=np.float32),
            'csi_max_per_user': np.zeros((self.max_users,), dtype=np.float32),
            'csi_std_per_user': np.zeros((self.max_users,), dtype=np.float32),

            # 用户状态 (25维)
            'user_buffer_status': np.zeros((self.max_users,), dtype=np.float32),

            # 分配历史 (60维)
            'slice_allocated_power': np.zeros((1, self.num_slices), dtype=np.float32),
            'slice_allocated_rbs': np.zeros((1, self.num_slices), dtype=np.float32),
            'user_allocated_rbs': np.zeros((1, self.max_users), dtype=np.float32),
            'user_allocated_power': np.zeros((1, self.max_users), dtype=np.float32),

            # Intent Drift（修正：5×5×3 = 75维）
            'intent_drift': np.zeros((self.num_slices, self.max_users_per_slice, 3), dtype=np.float32),

            # SLA信息 (30维：5×3×2)
            'sla_target': np.zeros((self.num_slices, 3), dtype=np.float32),
            'active_sla': np.zeros((self.num_slices, 3), dtype=np.float32),

            # 拓扑信息 (160维)
            'slice_priority': np.zeros((self.num_slices,), dtype=np.float32),
            'user_to_slice': np.zeros((self.num_slices, self.max_users), dtype=np.int32),
            'active_users': np.zeros((1, self.max_users), dtype=np.int8),
            'active_slices': np.zeros((1, self.num_slices), dtype=np.int8),
        }
