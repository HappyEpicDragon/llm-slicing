import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from hydra.utils import get_class

from src.basic_apis.network_slicing_business.network_slicing_business_executor \
    import ComponentConfig, ComponentClasses, ComponentFactory, NetworkSlicingBusinessExecutor
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.ppo.utils import intent_drift_calc


class HierarchicalSlicingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, env_settings, np_random, path_context: PathContext):
        super().__init__()
        self.config = env_settings
        self.np_random = np_random
        self.path_context = path_context

        # === 1. 业务组件初始化 ===
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

        self.components = None
        self.business_executor = None
        self.last_raw_obs = None
        self.last_unformatted_obs_deque = deque(maxlen=10)

        # === 2. 常量定义 (PRB 粒度) ===
        self.num_phys_rbs = int(self.components_config.basestation_config.num_available_rbs[0])  # 135
        self.alloc_unit_count = self.num_phys_rbs

        self.num_slices = self.components_config.slice_config.max_number_slices
        self.users_per_slice = 5
        self.max_users = self.components_config.ue_config.max_number_ues
        self.max_bs_power = self.config.components.basestations.total_power

        self.max_timesteps = self.config.env_state_config.max_channel_timesteps
        self.current_timestep = 0

        # 场景控制
        self.mode_config = self.config[self.scenario_mode][self.mode]
        self.scenario_list = self.mode_config.active_scenario_list
        self.init_ep = self.mode_config.init_scenario_episode
        self.max_ep = self.mode_config.max_scenario_episodes
        self.skip_step = self.config[self.scenario_mode].episode_cross_scenario_skip
        self.scenario_pointer = 0
        self.internal_episode_ptr = self.init_ep
        self.current_episode_idx = 0
        self.total_episodes = 0

        # === 3. 动作空间 ===
        # 30维连续 Logits -> Softmax -> 135 PRB Counts
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(30,), dtype=np.float32)

        # === 4. 观测空间 (回归消融实验最佳配置) ===
        self.observation_space = spaces.Dict({
            # Inter (4维): [Priority, Drift, Traffic, Last_Alloc] -> 移除 Target
            "inter_feat": spaces.Box(low=-5, high=10, shape=(self.num_slices, 4), dtype=np.float32),

            # Intra (7维): [Buffer, Prio, Drift, CSI, HOL, Arrival, RB_Cost] -> 保留增强感知
            # "intra_feat": spaces.Box(low=-5, high=10, shape=(self.max_users, 7), dtype=np.float32),
            "intra_feat": spaces.Box(low=-5, high=10, shape=(self.max_users, 5), dtype=np.float32),

            "global_feat": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })

        self.last_inter_alloc_ratio = np.zeros(self.num_slices, dtype=np.float32)
        self.last_intra_alloc_ratio = np.zeros(self.max_users, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        self.last_inter_alloc_ratio.fill(0.0)
        self.last_intra_alloc_ratio.fill(0.0)
        self.components = self.component_factory.create_scenario_components(
            episode_number=self.current_episode_idx, step_number=0
        )
        self.business_executor = NetworkSlicingBusinessExecutor(
            self.components, self.components_config
        )
        self.components.metrics.reset()
        self._prefill_observation()
        return self._get_hierarchical_observation(), {}

    def step(self, action: np.ndarray):
        # --- 1. 动作解析 (连续 -> 离散计数) ---
        inter_logits = action[:self.num_slices]
        slice_active_mask = self._get_slice_active_mask()
        inter_quotas = self._logits_to_counts(inter_logits, slice_active_mask, self.alloc_unit_count)

        intra_logits_all = action[self.num_slices:].reshape(self.num_slices, self.users_per_slice)
        final_user_counts = np.zeros(self.max_users, dtype=int)
        slice_ue_map = self.components.slices.ue_assoc

        for s_idx in range(self.num_slices):
            quota = inter_quotas[s_idx]
            if quota == 0: continue

            u_logits = intra_logits_all[s_idx]
            user_active_mask = self._get_slice_user_active_mask(s_idx, slice_ue_map)
            global_indices = np.where(slice_ue_map[s_idx] > 0)[0]

            if len(global_indices) > 0:
                if np.sum(user_active_mask) == 0: user_active_mask[:] = 1.0
                allocs = self._logits_to_counts(u_logits, user_active_mask, quota)
                for i, u_idx in enumerate(global_indices):
                    final_user_counts[u_idx] = allocs[i]

        self.last_inter_alloc_ratio = inter_quotas / self.alloc_unit_count
        self.last_intra_alloc_ratio = final_user_counts / self.alloc_unit_count

        # --- 2. 物理映射 (简单版智能保底 + Max-CQI) ---
        allocation_matrix, oneshot_action = self._map_counts_to_prbs_simple_guarantee(final_user_counts)

        self.business_executor.temp_rb_allocation[:] = allocation_matrix
        self.business_executor.temp_rb_ues_association.fill(0)
        self.business_executor.temp_rb_ues_association[0] = (allocation_matrix > 1e-9).astype(float)

        # --- 3. 执行 ---
        self.current_timestep += 1
        metrics = self.business_executor.execute_tti_physics(
            channel_timestep=self.current_timestep,
            episode_number=self.current_episode_idx
        )

        self.last_unformatted_obs_deque.appendleft(metrics)
        metrics["intent_drift"] = intent_drift_calc(
            self.last_unformatted_obs_deque, self.users_per_slice, 0.2
        )
        self._update_priority_in_obs(metrics)
        self.last_raw_obs = metrics
        self.components.metrics.step(metrics)

        # --- 4. Reward (简单线性) ---
        reward = self._calculate_simple_reward(metrics)

        obs = self._get_hierarchical_observation()
        done = (self.current_timestep >= self.max_timesteps)
        info = self._get_info(metrics, reward)
        info['final_executed_action'] = oneshot_action

        if self.mode == 'testing' and done:
            self._handle_testing_save()

        return obs, reward, done, False, info

    # =========================================================
    # [物理映射] 统一阈值保底 (Back to 0.3)
    # =========================================================
    # def _map_counts_to_prbs_simple_guarantee(self, user_counts: np.ndarray):
    #     raw_csi = self.last_raw_obs.get('target_cell_power')
    #     if raw_csi is None: raw_csi = np.zeros((self.max_users, self.alloc_unit_count))
    #     if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
    #     if raw_csi.shape[0] == self.alloc_unit_count: raw_csi = raw_csi.T
    #
    #     raw_lat = self.last_raw_obs.get('buffer_latencies', np.zeros(self.max_users))
    #     hol_delay_norm = raw_lat / 100.0
    #
    #     occupied_prbs = set()
    #     allocation_matrix = np.zeros((self.max_users, self.alloc_unit_count), dtype=np.float32)
    #     power_per_prb = self.max_bs_power / self.alloc_unit_count
    #     oneshot_action = np.zeros(self.alloc_unit_count, dtype=int)
    #
    #     remaining_counts = user_counts.copy()
    #
    #     # --- Phase 1: 统一阈值救急 ---
    #     active_users = np.where(remaining_counts > 0)[0]
    #
    #     for u_idx in active_users:
    #         if remaining_counts[u_idx] <= 0: continue
    #
    #         # [回退] 统一阈值 0.3，不区分 HP/NHP
    #         if hol_delay_norm[u_idx] < 0.3:
    #             continue
    #
    #         best_prb = -1
    #         best_gain = -1e9
    #         for r in range(self.alloc_unit_count):
    #             if r not in occupied_prbs:
    #                 gain = raw_csi[u_idx, r]
    #                 if gain > best_gain:
    #                     best_gain = gain
    #                     best_prb = r
    #
    #         if best_prb != -1:
    #             occupied_prbs.add(best_prb)
    #             remaining_counts[u_idx] -= 1
    #             oneshot_action[best_prb] = u_idx + 1
    #             allocation_matrix[u_idx, best_prb] = power_per_prb
    #
    #     # --- Phase 2: 全局竞争 ---
    #     candidates = []
    #     for u_idx in range(self.max_users):
    #         if remaining_counts[u_idx] > 0:
    #             for prb_idx in range(self.alloc_unit_count):
    #                 if prb_idx not in occupied_prbs:
    #                     candidates.append((raw_csi[u_idx, prb_idx], u_idx, prb_idx))
    #
    #     candidates.sort(key=lambda x: x[0], reverse=True)
    #
    #     for gain, u_idx, prb_idx in candidates:
    #         if prb_idx in occupied_prbs: continue
    #         if remaining_counts[u_idx] <= 0: continue
    #         occupied_prbs.add(prb_idx)
    #         remaining_counts[u_idx] -= 1
    #         oneshot_action[prb_idx] = u_idx + 1
    #         allocation_matrix[u_idx, prb_idx] = power_per_prb
    #         if len(occupied_prbs) == self.alloc_unit_count: break
    #
    #     return allocation_matrix, oneshot_action

    # def _map_counts_to_prbs_simple_guarantee(self, user_counts: np.ndarray):
    #     # 1. 准备数据 (同前)
    #     raw_csi = self.last_raw_obs.get('target_cell_power')
    #     if raw_csi is None: raw_csi = np.zeros((self.max_users, self.alloc_unit_count))
    #     if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
    #     if raw_csi.shape[0] == self.alloc_unit_count: raw_csi = raw_csi.T
    #
    #     # [新增] 获取 HOL Delay 用于判断是否"急诊"
    #     raw_lat = self.last_raw_obs.get('buffer_latencies', np.zeros(self.max_users))
    #     # 归一化，假设 100ms 为红线
    #     hol_delay_norm = raw_lat / 100.0
    #
    #     occupied_prbs = set()
    #     allocation_matrix = np.zeros((self.max_users, self.alloc_unit_count), dtype=np.float32)
    #     power_per_prb = self.max_bs_power / self.alloc_unit_count
    #     oneshot_action = np.zeros(self.alloc_unit_count, dtype=int)
    #
    #     remaining_counts = user_counts.copy()
    #
    #     # === 阶段 1: 智能最小保障 (Smart Triage) ===
    #     active_users = np.where(remaining_counts > 0)[0]
    #
    #     for u_idx in active_users:
    #         if remaining_counts[u_idx] <= 0: continue
    #
    #         # [关键修改] 只有当延迟紧迫 (e.g. > 0.3) 时，才给予物理层保底
    #         # 否则去阶段2竞争，这样能省下好位置给真正急的人
    #         if hol_delay_norm[u_idx] < 0.3:
    #             continue
    #
    #         # 寻找该用户当前最好的空闲 PRB
    #         best_prb = -1
    #         best_gain = -1e9
    #         for r in range(self.alloc_unit_count):
    #             if r not in occupied_prbs:
    #                 gain = raw_csi[u_idx, r]
    #                 if gain > best_gain:
    #                     best_gain = gain
    #                     best_prb = r
    #
    #         if best_prb != -1:
    #             occupied_prbs.add(best_prb)
    #             remaining_counts[u_idx] -= 1
    #             oneshot_action[best_prb] = u_idx + 1
    #             allocation_matrix[u_idx, best_prb] = power_per_prb
    #
    #     # === 阶段 2: 全局竞争 (Max-CQI) ===
    #     # (代码保持不变，负责分配剩下的资源)
    #     candidates = []
    #     for u_idx in range(self.max_users):
    #         if remaining_counts[u_idx] > 0:
    #             for prb_idx in range(self.alloc_unit_count):
    #                 if prb_idx not in occupied_prbs:
    #                     candidates.append((raw_csi[u_idx, prb_idx], u_idx, prb_idx))
    #
    #     candidates.sort(key=lambda x: x[0], reverse=True)
    #
    #     for gain, u_idx, prb_idx in candidates:
    #         if prb_idx in occupied_prbs: continue
    #         if remaining_counts[u_idx] <= 0: continue
    #
    #         occupied_prbs.add(prb_idx)
    #         remaining_counts[u_idx] -= 1
    #         oneshot_action[prb_idx] = u_idx + 1
    #         allocation_matrix[u_idx, prb_idx] = power_per_prb
    #         if len(occupied_prbs) == self.alloc_unit_count: break
    #
    #     return allocation_matrix, oneshot_action

    # def _map_counts_to_prbs_simple_guarantee(self, user_counts: np.ndarray):
    #     """
    #     物理资源映射函数 (Baseline灵感版: 动态池化 + 熔断保护)
    #
    #     设计理念:
    #     1. [HP Override]: 像 Baseline 一样，优先保证 HP 业务的连通性，无视 RL 的僵化配额。
    #     2. [Resource Pooling]: 对紧急 HP 用户，将所有空闲 PRB 视为一个公共资源池进行抢救。
    #     3. [Safety Cap]: 引入 60% 熔断阈值，防止在过载场景(Scenario 3)下 HP 挤死 NHP，修复 Scenario 0 的倒退。
    #     """
    #     # --- 1. 准备信道数据 ---
    #     raw_csi = self.last_raw_obs.get('target_cell_power')
    #     if raw_csi is None: raw_csi = np.zeros((self.max_users, self.alloc_unit_count))
    #     if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
    #     if raw_csi.shape[0] == self.alloc_unit_count: raw_csi = raw_csi.T
    #
    #     # --- 2. 准备基础数据 ---
    #     raw_lat = self.last_raw_obs.get('buffer_latencies', np.zeros(self.max_users))
    #
    #     # 获取用户优先级
    #     slice_assoc = self.components.slices.ue_assoc
    #     slice_prio = self.last_raw_obs.get("slice_priority", np.zeros(self.num_slices))
    #     user_prio = slice_assoc.T @ slice_prio
    #
    #     # 获取用户级 SLA
    #     user_sla_limits = np.full(self.max_users, 100.0, dtype=np.float32)
    #     try:
    #         slice_reqs = self.components.slices.requirements
    #         for s_idx in range(self.num_slices):
    #             if f"slice_{s_idx}" not in slice_reqs: continue
    #             req_data = slice_reqs[f"slice_{s_idx}"]
    #             limit = 100.0
    #             if 'parameters' in req_data:
    #                 for p_val in req_data['parameters'].values():
    #                     if p_val.get('name') == 'latency':
    #                         limit = float(p_val.get('value', 100.0))
    #                         break
    #             users_in_slice = np.where(slice_assoc[s_idx] > 0)[0]
    #             if len(users_in_slice) > 0:
    #                 user_sla_limits[users_in_slice] = limit
    #     except:
    #         pass
    #
    #     # --- 初始化 ---
    #     occupied_prbs = set()
    #     allocation_matrix = np.zeros((self.max_users, self.alloc_unit_count), dtype=np.float32)
    #     power_per_prb = self.max_bs_power / self.alloc_unit_count
    #     oneshot_action = np.zeros(self.alloc_unit_count, dtype=int)
    #
    #     # 复制 RL 配额
    #     remaining_counts = user_counts.copy()
    #
    #     # ==========================================================
    #     # Phase 1: HP Critical Rescue (模拟 Baseline 的优先调度)
    #     # ==========================================================
    #
    #     # [Safety Cap] 熔断阈值：最多只允许 60% 的资源用于“强行救急”
    #     # 这保留了 40% 的资源给 NHP 或非紧急 HP，防止 Scenario 0/2 中 NHP 饿死
    #     MAX_RESCUE_RBS = int(self.alloc_unit_count * 0.60)
    #     current_rescue_rbs = 0
    #
    #     hp_users = np.where(user_prio > 0)[0]
    #
    #     # 筛选出真正紧急的 HP 用户 (Latency > 40% SLA)
    #     emergency_queue = []
    #     for u in hp_users:
    #         ratio = raw_lat[u] / (user_sla_limits[u] + 1e-6)
    #         if ratio > 0.4:  # 门槛：40% 时间耗尽
    #             emergency_queue.append((ratio, u))
    #
    #     # 按紧急程度排序 (最急的先拿资源)
    #     emergency_queue.sort(key=lambda x: x[0], reverse=True)
    #
    #     for ratio, u_idx in emergency_queue:
    #         if current_rescue_rbs >= MAX_RESCUE_RBS:
    #             break  # 触发熔断，停止救援
    #
    #         # 动态剂量：弱信道(Cell Edge)给多点(5)，强信道给少点(3)
    #         # 这能防止弱信道用户因为发不完包而一直卡在队列头
    #         avg_gain = np.mean(raw_csi[u_idx])
    #         rescue_budget = 5 if avg_gain < 1e-10 else 3
    #
    #         allocated = 0
    #         while allocated < rescue_budget:
    #             if current_rescue_rbs >= MAX_RESCUE_RBS: break
    #
    #             # 寻找最佳空闲 PRB
    #             best_prb = -1
    #             best_gain = -1e9
    #             for r in range(self.alloc_unit_count):
    #                 if r not in occupied_prbs:
    #                     gain = raw_csi[u_idx, r]
    #                     if gain > best_gain:
    #                         best_gain = gain
    #                         best_prb = r
    #
    #             if best_prb != -1:
    #                 occupied_prbs.add(best_prb)
    #                 oneshot_action[best_prb] = u_idx + 1
    #                 allocation_matrix[u_idx, best_prb] = power_per_prb
    #
    #                 # 更新计数
    #                 allocated += 1
    #                 current_rescue_rbs += 1
    #                 # 扣减 RL 配额 (防止 Phase 2 重复分配太多)
    #                 remaining_counts[u_idx] -= 1
    #             else:
    #                 break
    #
    #     # ==========================================================
    #     # Phase 2: Max-CQI Filling (模拟 Baseline 的高效吞吐)
    #     # ==========================================================
    #     # 对剩余资源进行基于信道质量(CSI)的分配
    #     # 只要 RL 给了配额 (remaining_counts > 0)，就有资格竞争
    #
    #     candidates = []
    #     for u_idx in range(self.max_users):
    #         if remaining_counts[u_idx] > 0:
    #             for r in range(self.alloc_unit_count):
    #                 if r not in occupied_prbs:
    #                     gain = raw_csi[u_idx, r]
    #                     candidates.append((gain, u_idx, r))
    #
    #     # 全局排序：谁信道好给谁 (最大化系统吞吐量)
    #     candidates.sort(key=lambda x: x[0], reverse=True)
    #
    #     for gain, u_idx, r in candidates:
    #         if r in occupied_prbs: continue
    #         if remaining_counts[u_idx] <= 0: continue
    #
    #         occupied_prbs.add(r)
    #         remaining_counts[u_idx] -= 1
    #         oneshot_action[r] = u_idx + 1
    #         allocation_matrix[u_idx, r] = power_per_prb
    #
    #         if len(occupied_prbs) == self.alloc_unit_count: break
    #
    #     return allocation_matrix, oneshot_action

    def _map_counts_to_prbs_simple_guarantee(self, user_counts: np.ndarray):
        """
        物理资源映射函数 (最终版: 混合 Triage + Safety Cap)

        设计理念:
        1. 仅对 HP 用户启用基于 Latency OR Buffer 的混合 Triage。
        2. 对 Critical HP 用户启用 Quota Override (无视 RL 0 配额)。
        3. 引入 60% 熔断机制，确保 NHP 仍有资源进行 Max-CQI 分配。
        """
        # --- 1. 准备信道数据 ---
        raw_csi = self.last_raw_obs.get('target_cell_power')
        if raw_csi is None: raw_csi = np.zeros((self.max_users, self.alloc_unit_count))
        if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
        if raw_csi.shape[0] == self.alloc_unit_count: raw_csi = raw_csi.T

        # --- 2. 准备基础数据 (延迟 & 缓冲区) ---
        raw_lat = self.last_raw_obs.get('buffer_latencies', np.zeros(self.max_users))
        # [新增] 缓冲区占用率，用于溢出风险检查
        buffer_occ = self.last_raw_obs.get('buffer_occupancies', np.zeros(self.max_users))

        # 获取用户优先级
        slice_assoc = self.components.slices.ue_assoc
        slice_prio = self.last_raw_obs.get("slice_priority", np.zeros(self.num_slices))
        user_prio = slice_assoc.T @ slice_prio

        # --- 3. 获取用户级 SLA (时延限制) ---
        user_sla_limits = np.full(self.max_users, 100.0, dtype=np.float32)
        try:
            slice_reqs = self.components.slices.requirements
            for s_idx in range(self.num_slices):
                if f"slice_{s_idx}" not in slice_reqs: continue
                req_data = slice_reqs[f"slice_{s_idx}"]
                limit = 100.0
                if 'parameters' in req_data:
                    for p_val in req_data['parameters'].values():
                        if p_val.get('name') == 'latency':
                            limit = float(p_val.get('value', 100.0))
                            break
                users_in_slice = np.where(slice_assoc[s_idx] > 0)[0]
                if len(users_in_slice) > 0:
                    user_sla_limits[users_in_slice] = limit
        except:
            pass

        # --- 4. 初始化 ---
        occupied_prbs = set()
        allocation_matrix = np.zeros((self.max_users, self.alloc_unit_count), dtype=np.float32)
        power_per_prb = self.max_bs_power / self.alloc_unit_count
        oneshot_action = np.zeros(self.alloc_unit_count, dtype=int)

        # 复制 RL 配额
        remaining_counts = user_counts.copy()

        # ==========================================================
        # Phase 1: HP Critical Rescue (混合 Triage + 熔断)
        # ==========================================================

        # [Safety Cap] 熔断阈值：最多只允许 60% 的资源用于“强行救急”
        MAX_RESCUE_RBS = int(self.alloc_unit_count * 0.60)
        current_rescue_rbs = 0

        hp_users = np.where(user_prio > 0)[0]

        # 筛选出真正紧急的 HP 用户 (Hybrid Triage Logic)
        emergency_queue = []
        for u in hp_users:
            danger_ratio = raw_lat[u] / (user_sla_limits[u] + 1e-6)

            # 1. 延迟风险 (Deadline Risk): 超过 40% SLA
            latency_risk = danger_ratio > 0.4

            # 2. 溢出风险 (Overflow Risk): 缓冲区占用超过 80%
            overflow_risk = buffer_occ[u] > 0.8

            if latency_risk or overflow_risk:
                # 优先级排序：以最高的风险（延迟或溢出）作为权重
                highest_risk = max(danger_ratio, buffer_occ[u])
                emergency_queue.append((highest_risk, u))

        # 按紧急程度排序
        emergency_queue.sort(key=lambda x: x[0], reverse=True)

        for ratio, u_idx in emergency_queue:
            if current_rescue_rbs >= MAX_RESCUE_RBS:
                break

                # 动态剂量：弱信道给5个，强信道给3个
            avg_gain = np.mean(raw_csi[u_idx])
            rescue_budget = 5 if avg_gain < 1e-10 else 3

            allocated = 0
            while allocated < rescue_budget:
                if current_rescue_rbs >= MAX_RESCUE_RBS: break

                best_prb = -1
                best_gain = -1e9

                # 寻找最佳空闲 PRB
                for r in range(self.alloc_unit_count):
                    if r not in occupied_prbs:
                        gain = raw_csi[u_idx, r]
                        if gain > best_gain:
                            best_gain = gain
                            best_prb = r

                if best_prb != -1:
                    occupied_prbs.add(best_prb)
                    oneshot_action[best_prb] = u_idx + 1
                    allocation_matrix[u_idx, best_prb] = power_per_prb

                    # 更新计数
                    allocated += 1
                    current_rescue_rbs += 1
                    # 扣减 RL 配额 (Quota Override)
                    remaining_counts[u_idx] -= 1
                else:
                    break

        # ==========================================================
        # Phase 2: Standard Allocation (RL Agent Control)
        # ==========================================================
        # 剩余资源 Max-CQI 分配

        candidates = []
        for u_idx in range(self.max_users):
            # 只有配额 > 0 才有资格竞争
            if remaining_counts[u_idx] > 0:
                for r in range(self.alloc_unit_count):
                    if r not in occupied_prbs:
                        gain = raw_csi[u_idx, r]
                        candidates.append((gain, u_idx, r))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for gain, u_idx, r in candidates:
            if r in occupied_prbs: continue
            if remaining_counts[u_idx] <= 0: continue

            occupied_prbs.add(r)
            remaining_counts[u_idx] -= 1
            oneshot_action[r] = u_idx + 1
            allocation_matrix[u_idx, r] = power_per_prb

            if len(occupied_prbs) == self.alloc_unit_count: break

        # 返回值必须包含两个矩阵
        return allocation_matrix, oneshot_action


    # =========================================================
    # [特征] 保留 7 维 Intra (含 RB_Cost 和 HOL)
    # =========================================================
    def _get_hierarchical_observation(self) -> Dict[str, np.ndarray]:
        if self.last_raw_obs is None: return self._get_dummy_obs()
        raw = self.last_raw_obs

        # Inter: 4 维 (无 Target)
        inter_feat = np.zeros((self.num_slices, 4), dtype=np.float32)
        drift = raw.get("intent_drift");
        slice_drift_mean = np.mean(np.maximum(drift, -1.0), axis=(1, 2))
        in_bits = raw.get("pkt_incoming_bits", np.zeros(self.max_users))
        slice_assoc = self.components.slices.ue_assoc;
        slice_traffic = (slice_assoc @ in_bits) / 1e6
        prio = raw.get("slice_priority", np.zeros(self.num_slices))

        for s in range(self.num_slices):
            inter_feat[s, 0] = prio[s]
            inter_feat[s, 1] = slice_drift_mean[s]
            inter_feat[s, 2] = np.clip(slice_traffic[s], 0, 5.0)
            inter_feat[s, 3] = self.last_inter_alloc_ratio[s]

        # Intra: 7 维 (含增强感知)
        # intra_feat = np.zeros((self.max_users, 7), dtype=np.float32)
        intra_feat = np.zeros((self.max_users, 5), dtype=np.float32)
        buffer = raw.get("buffer_occupancies", np.zeros(self.max_users))

        raw_csi = raw.get('target_cell_power', np.zeros((self.max_users, self.num_phys_rbs)))
        if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi).T
        mean_gain = np.mean(raw_csi, axis=1) + 1e-30
        csi_norm = np.clip(10 * np.log10(mean_gain) / 100.0 + 1.0, 0, 1)

        # RB Cost
        snr = mean_gain / 1e-14
        cap_proxy = np.log2(1 + snr)
        rb_cost = buffer / (cap_proxy + 1e-6)
        rb_cost_norm = np.log1p(rb_cost * 100.0)

        # HOL & Arrival
        raw_lat = raw.get('buffer_latencies', np.zeros(self.max_users))
        hol_delay_norm = np.clip(raw_lat / 100.0, 0, 1.0)
        arrival_norm = np.clip(np.log1p(in_bits) / 10.0, 0, 1.0)

        user_prio = slice_assoc.T @ prio
        user_drift_min = np.zeros(self.max_users)
        for s in range(self.num_slices):
            users = np.where(slice_assoc[s] > 0)[0]
            for i, u in enumerate(users):
                if i < 5: user_drift_min[u] = np.min(drift[s, i, :])

        for u in range(self.max_users):
            intra_feat[u, 0] = buffer[u]
            intra_feat[u, 1] = user_prio[u]
            intra_feat[u, 2] = user_drift_min[u]
            intra_feat[u, 3] = csi_norm[u]
            intra_feat[u, 4] = hol_delay_norm[u]
            # intra_feat[u, 5] = arrival_norm[u]
            # intra_feat[u, 6] = rb_cost_norm[u]

        global_feat = np.array([self.current_timestep / 1000.0, np.sum(self.last_inter_alloc_ratio)], dtype=np.float32)
        return {"inter_feat": inter_feat, "intra_feat": intra_feat, "global_feat": global_feat}

    # =========================================================
    # [Reward] 简单线性版 (无 Capping, 无 Risk)
    # =========================================================
    # def _calculate_simple_reward(self, metrics):
    #     raw_intent_drift = metrics.get("intent_drift")
    #     slice_assoc = self.components.slices.ue_assoc
    #     slice_reqs = metrics.get('slice_req', {})
    #
    #     slice_scores = []
    #     slice_priorities = []
    #
    #     for s_idx in range(self.num_slices):
    #         req = slice_reqs.get(f'slice_{s_idx}', {})
    #         priority = req.get('priority', 0)
    #         active_users = np.where(slice_assoc[s_idx] > 0)[0]
    #         if len(active_users) == 0: continue
    #         valid_drifts = []
    #         limit = min(len(active_users), self.users_per_slice)
    #         for u_local_idx in range(limit):
    #             for m_idx in range(3):
    #                 drift = raw_intent_drift[s_idx, u_local_idx, m_idx]
    #                 if drift > -1.5: valid_drifts.append(drift)
    #         if len(valid_drifts) > 0:
    #             slice_scores.append(np.min(valid_drifts))
    #             slice_priorities.append(priority)
    #         else:
    #             slice_scores.append(0.0)
    #             slice_priorities.append(priority)
    #
    #     slice_scores = np.array(slice_scores)
    #     slice_priorities = np.array(slice_priorities)
    #     if len(slice_scores) == 0: return 0.0
    #
    #     # [回退] 无 Capping (鼓励 NHP 优化)
    #     # [回退] 线性 HP 惩罚
    #     hp_penalty = 0.0
    #     hp_violation_mask = (slice_scores < 0) & (slice_priorities > 0)
    #     if np.any(hp_violation_mask):
    #         hp_bad_scores = slice_scores[hp_violation_mask]
    #         # 线性惩罚
    #         hp_penalty = -5.0 + np.sum(hp_bad_scores) * 10.0
    #
    #     # [回退] Base Reward (Min/Mean)
    #     min_score = np.min(slice_scores)
    #     if min_score < 0:
    #         base_reward = min_score * 2.0
    #     else:
    #         base_reward = np.mean(slice_scores)
    #
    #     # [回退] 无 Risk Penalty
    #     return base_reward + hp_penalty

    def _calculate_simple_reward(self, metrics):
        raw_intent_drift = metrics.get("intent_drift")
        slice_assoc = self.components.slices.ue_assoc
        slice_reqs = metrics.get('slice_req', {})

        slice_scores = []
        slice_priorities = []

        for s_idx in range(self.num_slices):
            req = slice_reqs.get(f'slice_{s_idx}', {})
            priority = req.get('priority', 0)
            active_users = np.where(slice_assoc[s_idx] > 0)[0]
            if len(active_users) == 0: continue

            valid_drifts = []
            limit = min(len(active_users), self.users_per_slice)
            for u_local_idx in range(limit):
                for m_idx in range(3):
                    drift = raw_intent_drift[s_idx, u_local_idx, m_idx]
                    if drift > -1.5: valid_drifts.append(drift)

            if len(valid_drifts) > 0:
                slice_scores.append(np.min(valid_drifts))
                slice_priorities.append(priority)
            else:
                slice_scores.append(0.0)
                slice_priorities.append(priority)

        slice_scores = np.array(slice_scores)
        slice_priorities = np.array(slice_priorities)
        if len(slice_scores) == 0: return 0.0

        # 1. NHP 封顶
        # capped_scores = np.where(slice_scores > 0, slice_scores * 0.05, slice_scores)
        capped_scores = np.where(slice_scores > 0, slice_scores * 0.2, slice_scores)

        # 2. HP 指数重罚
        hp_penalty = 0.0
        hp_violation_mask = (slice_scores < 0) & (slice_priorities > 0)
        if np.any(hp_violation_mask):
            hp_bad_scores = slice_scores[hp_violation_mask]
            hp_penalty = -10.0 - np.sum(np.exp(np.abs(hp_bad_scores) * 2.0))

        # 3. Base Reward
        min_score = np.min(capped_scores)
        if min_score < 0:
            base_reward = min_score * 2.0
        else:
            base_reward = np.mean(capped_scores)

        total_base = base_reward + hp_penalty

        # 4. Risk Penalty
        buffer_occ = metrics.get("buffer_occupancies", np.zeros(self.max_users))
        risk_loss = 0.0
        danger_mask = buffer_occ > 0.8
        if np.any(danger_mask):
            risk_loss = np.sum(np.exp(buffer_occ[danger_mask] * 5.0))

        risk_coef = getattr(self, 'reward_weights', {}).get('risk_coef', 0.2)
        risk_penalty = -1.0 * risk_loss * risk_coef

        total_reward = total_base + risk_penalty

        self.last_reward_info = {
            "base": total_base,
            "risk": risk_penalty
        }
        return total_reward

    # ... (其他辅助函数 _logits_to_counts, _get_dummy_obs 等同前) ...
    def _logits_to_counts(self, logits, mask, total_count):
        if total_count == 0: return np.zeros_like(logits, dtype=int)
        masked_logits = logits.copy()
        if np.sum(mask) == 0: mask[:] = 1.0
        masked_logits[mask == 0] = -1e9
        max_l = np.max(masked_logits)
        exp_l = np.exp(masked_logits - max_l)
        probs = exp_l / np.sum(exp_l)
        raw_counts = probs * total_count
        int_counts = np.floor(raw_counts).astype(int)
        remainders = raw_counts - int_counts
        diff = total_count - np.sum(int_counts)
        if diff > 0:
            indices = np.argsort(remainders)[-diff:][::-1]
            int_counts[indices] += 1
        return int_counts

    def _get_slice_active_mask(self):
        slice_assoc = self.components.slices.ue_assoc
        return (np.sum(slice_assoc, axis=1) > 0).astype(float)

    def _get_slice_user_active_mask(self, s_idx, slice_ue_map):
        global_indices = np.where(slice_ue_map[s_idx] > 0)[0]
        mask = np.zeros(self.users_per_slice)
        buffer = self.last_raw_obs.get("buffer_occupancies", np.zeros(self.max_users))
        arrival = self.last_raw_obs.get("pkt_incoming_bits", np.zeros(self.max_users))
        for i, u_idx in enumerate(global_indices):
            if buffer[u_idx] > 1e-6 or arrival[u_idx] > 1e-6:
                mask[i] = 1.0
        if len(global_indices) > 0 and np.sum(mask) == 0:
            mask[:len(global_indices)] = 1.0
        return mask

    def _prefill_observation(self):
        dummy_alloc = np.zeros((self.max_users, self.num_phys_rbs))
        self.business_executor.temp_rb_allocation[:] = dummy_alloc
        self.business_executor.temp_rb_ues_association.fill(0)
        metrics = self.business_executor.execute_tti_physics(0, self.current_episode_idx)
        metrics["intent_drift"] = np.zeros((self.num_slices, 5, 3))
        self._update_priority_in_obs(metrics)
        self.last_raw_obs = metrics

    def _update_priority_in_obs(self, raw_obs):
        prio = np.zeros(self.num_slices, dtype=np.float32)
        for i in range(self.num_slices):
            prio[i] = raw_obs.get("slice_req", {}).get(f'slice_{i}', {}).get('priority', 0)
        raw_obs["slice_priority"] = prio

    def _get_dummy_obs(self):
        return {
            "inter_feat": np.zeros((self.num_slices, 4), dtype=np.float32),
            # "intra_feat": np.zeros((self.max_users, 7), dtype=np.float32),  # 7 dims
            "intra_feat": np.zeros((self.max_users, 5), dtype=np.float32),
            "global_feat": np.zeros((2,), dtype=np.float32)
        }

    def _get_info(self, metrics, reward):
        info = {
            "throughput": np.sum(metrics.get('pkt_effective_thr', 0)),
            "reward_total": reward,
            "reward/base": reward,
            "reward/risk": 0.0
        }
        traffics_1ms = metrics.get("pkt_incoming_bits", np.zeros(self.max_users))
        slice_assoc = self.components.slices.ue_assoc
        for s_idx in range(self.num_slices):
            u_indices = np.where(slice_assoc[s_idx] > 0)[0]
            if len(u_indices) > 0:
                bits_arrived = np.sum(traffics_1ms[u_indices])
                info[f"arrival/slice_{s_idx}"] = (bits_arrived * 1000) / 1e6
            else:
                info[f"arrival/slice_{s_idx}"] = 0.0
        info.update(self._extract_detailed_violation_info(metrics))
        _, active_sla_mask = self._extract_sla_info(metrics.get("slice_req", {}))
        metric_names = ['thr', 'rel', 'lat']
        for s_idx in range(self.num_slices):
            is_active = 1 if np.sum(slice_assoc[s_idx]) > 0 else 0
            info[f"meta/slice_{s_idx}_active"] = is_active
            prio = metrics.get("slice_req", {}).get(f'slice_{s_idx}', {}).get('priority', 0)
            info[f"meta/slice_{s_idx}_priority"] = int(prio)
            for m_idx, m_name in enumerate(metric_names):
                info[f"meta/slice_{s_idx}_{m_name}_req"] = int(active_sla_mask[s_idx, m_idx])
        return info

    def _extract_detailed_violation_info(self, metrics):
        drift = metrics.get("intent_drift")
        details = {}
        if drift is None: return {}
        metric_names = ['thr', 'rel', 'lat']
        for s_idx in range(self.num_slices):
            slice_drift = drift[s_idx]
            valid_mask = slice_drift > -1.5
            for m_idx, m_name in enumerate(metric_names):
                valid_vals = slice_drift[:, m_idx][valid_mask[:, m_idx]]
                key = f"slice_{s_idx}_{m_name}"
                if len(valid_vals) > 0:
                    min_val = np.min(valid_vals)
                    details[f"drift/{key}"] = min_val
                    details[f"violation/{key}"] = 1.0 if min_val < 0 else 0.0
                else:
                    details[f"drift/{key}"] = 0.0
                    details[f"violation/{key}"] = 0.0
        return details

    def _extract_sla_info(self, slice_req):
        active_sla = np.zeros((self.num_slices, 3), dtype=np.float32)
        metric_map = {'throughput': 0, 'reliability': 1, 'latency': 2}
        for s_idx in range(self.num_slices):
            req = slice_req.get(f'slice_{s_idx}', {})
            if not req or 'parameters' not in req: continue
            for param in req['parameters'].values():
                m_name = param.get('name', '').lower()
                if m_name in metric_map:
                    idx = metric_map[m_name]
                    active_sla[s_idx, idx] = 1.0
        return None, active_sla

    def _handle_testing_save(self):
        self.business_executor.save_metric(self.config.model_name, f"ep_{self.current_episode_idx}.npz")
        if hasattr(self.business_executor, 'reset_metric'):
            self.business_executor.reset_metric()