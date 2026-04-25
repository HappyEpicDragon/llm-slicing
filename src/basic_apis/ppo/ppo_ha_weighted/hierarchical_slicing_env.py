import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from hydra.utils import get_class

from src.basic_apis.network_slicing_business.network_slicing_business_executor \
    import ComponentConfig, ComponentClasses, ComponentFactory, NetworkSlicingBusinessExecutor
# 导入必要的计算函数
from src.basic_apis.ppo.utils import intent_drift_calc
from src.basic_apis.codebook_utils import build_dirichlet_inter_quota_codebook


# 兼容性包装器 (如果有 utils 依赖)
class EnvCompatibilityWrapper:
    def __init__(self, env):
        self.comm_env = env


class HierarchicalSlicingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, env_settings, np_random, paths_cfg=None, workdir: str = None):
        super().__init__()
        self.config = env_settings
        self.np_random = np_random
        self.paths_cfg = getattr(paths_cfg, "paths_cfg", paths_cfg)
        self.workdir = workdir if workdir is not None else getattr(paths_cfg, "hydra_workdir", None)

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
            self.components_config,
            component_classes,
            self.np_random,
            paths_cfg=self.paths_cfg,
            workdir=self.workdir,
        )

        self.components = None
        self.business_executor = None
        self.last_raw_obs = None
        self.last_unformatted_obs_deque = deque(maxlen=10)

        # === 2. 常量定义 ===
        self.num_phys_rbs = int(self.components_config.basestation_config.num_available_rbs[0])
        self.alloc_unit_count = self.num_phys_rbs

        self.num_slices = self.components_config.slice_config.max_number_slices
        self.users_per_slice = 5
        self.max_users = self.components_config.ue_config.max_number_ues
        self.max_bs_power = self.config.components.basestations.total_power
        self.bandwidth = float(self.components_config.basestation_config.bandwidths[0])

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

        # === 5. 离散模式数据定义 ===
        self.inter_quota_patterns = self._define_inter_quota_patterns()

        # === 3. 动作空间 (MultiDiscrete) ===
        # Inter (N modes) + 5 * Intra (3 schedulers)
        num_inter_modes = len(self.inter_quota_patterns)
        action_dims = [num_inter_modes] + [3] * self.num_slices
        self.action_space = spaces.MultiDiscrete(action_dims)

        # === 4. 观测空间 ===
        self.observation_space = spaces.Dict({
            "inter_feat": spaces.Box(low=-5, high=10, shape=(self.num_slices, 4), dtype=np.float32),
            "intra_feat": spaces.Box(low=-5, high=10, shape=(self.max_users, 5), dtype=np.float32),
            "global_feat": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })

        self.last_inter_alloc_ratio = np.zeros(self.num_slices, dtype=np.float32)
        self.last_intra_alloc_ratio = np.zeros(self.max_users, dtype=np.float32)
        self.last_rescue_rbs = 0
        self.last_demand_scores = np.zeros(self.num_slices, dtype=np.float32)
        self.last_sorted_slice_indices = np.arange(self.num_slices, dtype=int)

        # === 消融开关（A3 / A4）===
        self.disable_critical_rescue = bool(self.config.get('disable_critical_rescue', False))
        self.mapping_mode = str(self.config.get('mapping_mode', 'urgency_rank'))

        self.env_wrapper = EnvCompatibilityWrapper(self)

        # === prev-step margin cache for reward function ===
        self._prev_min_margins: list = [0.0] * self.num_slices
        self._prev_mean_margins: list = [0.0] * self.num_slices

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
        self._prev_min_margins = [0.0] * self.num_slices
        self._prev_mean_margins = [0.0] * self.num_slices
        self.last_unformatted_obs_deque.clear()
        self.last_raw_obs = None
        self.last_inter_alloc_ratio.fill(0.0)
        self.last_intra_alloc_ratio.fill(0.0)
        self.last_rescue_rbs = 0
        self.last_demand_scores.fill(0.0)
        self.last_sorted_slice_indices = np.arange(self.num_slices, dtype=int)
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

        # --- 1. 动作解析 ---
        inter_mode = action[0]
        intra_modes = action[1:]

        # 获取 Agent 选中的 '数值配方' (Pattern)
        raw_pattern = self.inter_quota_patterns[inter_mode]

        # =========================================================
        # [NEW] 动态映射逻辑 (Smart Mapping)
        # =========================================================

        # A. 获取当前状态用于计算需求
        if self.last_raw_obs is None:
            slice_prio = np.zeros(self.num_slices)
            slice_drift = np.zeros(self.num_slices)
            slice_traffic = np.zeros(self.num_slices)
        else:
            # 从 last_raw_obs 解析数据
            slice_prio = self.last_raw_obs.get("slice_priority", np.zeros(self.num_slices))
            in_bits = self.last_raw_obs.get("pkt_incoming_bits", np.zeros(self.max_users))
            slice_assoc = self.components.slices.ue_assoc
            slice_traffic = (slice_assoc @ in_bits) / 1e6

            # Drift (Slices, Users, Metrics) -> Slice min
            drift_raw = self.last_raw_obs.get("intent_drift", np.zeros((self.num_slices, 5, 3)))
            slice_drift = np.min(drift_raw, axis=(1, 2))

        # B. 映射模式：urgency_rank（默认）或 fixed_order（A4 消融用）
        if self.mapping_mode == 'fixed_order':
            # A4 消融：固定切片顺序，不做需求排序
            sorted_slice_indices = np.arange(self.num_slices)
            sorted_quotas = np.sort(raw_pattern)[::-1]
            demand_scores = np.zeros(self.num_slices, dtype=np.float32)
        else:
            # 默认：按饥渴度降序动态映射
            demand_scores = (slice_traffic * 10.0) + \
                            (slice_prio * 5.0) + \
                            ((1.0 - slice_drift) * 5.0)
            sorted_slice_indices = np.argsort(demand_scores)[::-1]
            sorted_quotas = np.sort(raw_pattern)[::-1]

        self.last_demand_scores = np.array(demand_scores, dtype=np.float32)
        self.last_sorted_slice_indices = np.array(sorted_slice_indices, dtype=int)

        # 映射: Rank 1 的切片拿 Rank 1 的配额
        inter_quotas = np.zeros(self.num_slices, dtype=int)
        for rank, s_idx in enumerate(sorted_slice_indices):
            inter_quotas[s_idx] = sorted_quotas[rank]

        # ----------------------------------------------------

        # --- 2. 物理映射 (使用映射后的 Quota + Native Schedulers + Final Sweep) ---
        allocation_matrix, oneshot_action, rescue_rbs = self._map_discrete_actions_to_prbs_native(
            inter_quotas, intra_modes
        )
        self.last_rescue_rbs = int(rescue_rbs)

        # --- 3. 执行 & Reward ---
        self.last_inter_alloc_ratio = inter_quotas / self.alloc_unit_count

        user_allocs = np.sum(allocation_matrix > 0, axis=1)
        if np.sum(user_allocs) > 0:
            self.last_intra_alloc_ratio = user_allocs / np.sum(user_allocs)
        else:
            self.last_intra_alloc_ratio.fill(0.0)

        self.business_executor.temp_rb_allocation[:] = allocation_matrix
        self.business_executor.temp_rb_ues_association.fill(0)
        self.business_executor.temp_rb_ues_association[0] = (allocation_matrix > 1e-9).astype(float)

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

        reward = self._calculate_simple_reward(metrics)

        obs = self._get_hierarchical_observation()
        done = (self.current_timestep >= self.max_timesteps)
        info = self._get_info(metrics, reward)
        info['final_executed_action'] = oneshot_action

        # if self.mode == 'testing' and done:
        #     self._handle_testing_save()

        return obs, reward, done, False, info

    # =========================================================
    # 原生实现的标准调度器 (Native Schedulers + Sweep)
    # =========================================================

    def _define_inter_quota_patterns(self):
        """
        按 system model 的 Dirichlet 有序统计方式构建 codebook。
        约束：每个模板降序、每片 >= c_min、总和严格等于 num_phys_rbs。
        """
        return build_dirichlet_inter_quota_codebook(
            num_slices=self.num_slices,
            num_prbs=self.num_phys_rbs,
            c_min=10,
            num_concentration_levels=10,
            num_mc_samples=10_000,
            beta_min=0.1,
            beta_max=50.0,
            seed=2025,
        )

    # =========================================================
    # [CORE] 离散调度器 (Triage + Native Schedulers + Sweep)
    # =========================================================
    # def _map_discrete_actions_to_prbs_native(self, inter_quotas: np.ndarray, intra_modes: np.ndarray):
    #     """
    #     原生实现的调度器映射逻辑 (Hybrid V2)。
    #     集成：Phase 1 (Triage) -> Phase 2 (Discrete Schedulers) -> Phase 3 (Sweep)
    #     """
    #     # --- 0. 初始化与数据准备 ---
    #     allocation_matrix = np.zeros((self.max_users, self.alloc_unit_count), dtype=np.float32)
    #     power_per_prb = self.max_bs_power / self.alloc_unit_count
    #     oneshot_action = np.zeros(self.alloc_unit_count, dtype=int)
    #     occupied_prbs = set()
    #
    #     slice_assoc = self.components.slices.ue_assoc
    #     latest_obs = self.last_raw_obs
    #
    #     # 信道与缓冲区
    #     raw_csi = latest_obs.get('target_cell_power')
    #     if raw_csi is None: raw_csi = np.zeros((self.max_users, self.alloc_unit_count))
    #     if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
    #     if raw_csi.shape[0] == self.alloc_unit_count: raw_csi = raw_csi.T
    #
    #     buffer_occ = latest_obs.get('buffer_occupancies', np.zeros(self.max_users))
    #     raw_lat = latest_obs.get('buffer_latencies', np.zeros(self.max_users))
    #     pkt_sizes = self.components.ues.pkt_sizes
    #     max_buffer_pkts = self.components.ues.max_buffer_pkts
    #
    #     # 优先级与SLA
    #     user_prio = slice_assoc.T @ latest_obs.get("slice_priority", np.zeros(self.num_slices))
    #     user_sla_limits = self._get_user_sla_limits()
    #
    #     # 历史吞吐量 (用于 PF)
    #     if len(self.last_unformatted_obs_deque) > 0:
    #         hist_thr = np.mean([
    #             obs.get('pkt_effective_thr', np.zeros(self.max_users))
    #             for obs in self.last_unformatted_obs_deque
    #         ], axis=0)
    #     else:
    #         hist_thr = np.zeros(self.max_users)
    #
    #     # 复制一份配额用于扣减
    #     remaining_slice_quotas = inter_quotas.copy()
    #
    #     # ==========================================================
    #     # Phase 1: HP Critical Rescue (Triage) - 复制自连续版逻辑
    #     # ==========================================================
    #     MAX_RESCUE_RBS = int(self.alloc_unit_count * 0.60)
    #     current_rescue_rbs = 0
    #     hp_users = np.where(user_prio > 0)[0]
    #
    #     # 构建紧急队列
    #     emergency_queue = self._build_emergency_queue(hp_users, raw_lat, buffer_occ, user_sla_limits)
    #     emergency_queue.sort(key=lambda x: x[0], reverse=True)
    #
    #     for ratio, u_idx in emergency_queue:
    #         if current_rescue_rbs >= MAX_RESCUE_RBS: break
    #
    #         avg_gain = np.mean(raw_csi[u_idx])
    #         rescue_budget = 5 if avg_gain < 1e-10 else 3
    #
    #         allocated = 0
    #         while allocated < rescue_budget:
    #             if current_rescue_rbs >= MAX_RESCUE_RBS: break
    #
    #             best_prb = -1
    #             best_gain = -1e9
    #             # 寻找全局最佳空闲 PRB
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
    #                 allocated += 1
    #                 current_rescue_rbs += 1
    #
    #                 # [关键修改]: 扣减该用户所属切片的配额
    #                 # 找到用户所属 Slice ID
    #                 s_indices = np.where(slice_assoc[:, u_idx] == 1)[0]
    #                 if len(s_indices) > 0:
    #                     s_idx = s_indices[0]
    #                     remaining_slice_quotas[s_idx] -= 1
    #                     # 注意：这里允许减成负数，Phase 2 会处理
    #             else:
    #                 break
    #
    #     # ==========================================================
    #     # Phase 2: Native Schedulers (按剩余配额分配)
    #     # ==========================================================
    #     # 由于 Phase 1 已经把资源打散了（占用了特定的 best_prb），
    #     # 我们不能再用简单的 [start_rb, end_rb] 范围。
    #     # 我们需要扫描所有空闲 RB，供调度器挑选。
    #
    #     for s_idx in range(self.num_slices):
    #         # 获取剩余配额 (如果 Triage 用超了，这里就是负数或0，直接跳过)
    #         quota = remaining_slice_quotas[s_idx]
    #         if quota <= 0: continue
    #
    #         slice_ues = np.where(slice_assoc[s_idx] == 1)[0]
    #         if len(slice_ues) == 0: continue
    #
    #         scheduler_id = intra_modes[s_idx]
    #
    #         # 获取当前所有可用的 RB 列表
    #         free_prbs_list = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
    #         if not free_prbs_list: break  # 没有资源了
    #
    #         # 如果剩余空闲资源 < 配额，则只能分配这么多
    #         alloc_limit = min(quota, len(free_prbs_list))
    #         if alloc_limit <= 0: continue
    #
    #         # === 执行具体的调度算法 (适配非连续 RB) ===
    #
    #         # 1. Round Robin (简单地从 free_prbs_list 前面拿)
    #         if scheduler_id == 0:
    #             base_alloc = alloc_limit // len(slice_ues)
    #             remainder = alloc_limit % len(slice_ues)
    #
    #             rb_ptr = 0
    #             for i, u_idx in enumerate(slice_ues):
    #                 count = base_alloc + (1 if i < remainder else 0)
    #                 if count > 0:
    #                     for k in range(count):
    #                         r = free_prbs_list[rb_ptr + k]
    #                         allocation_matrix[u_idx, r] = power_per_prb
    #                         oneshot_action[r] = u_idx + 1
    #                         occupied_prbs.add(r)
    #                     rb_ptr += count
    #
    #         # 2. Proportional Fairness (PF)
    #         # 在 free_prbs_list 中挑选得分最高的 (User, RB) 组合
    #         # 为简化计算，我们假设 PF 主要由 User 决定，然后给该 User 分配其在 free_prbs 中最好的 RB
    #         elif scheduler_id == 1:
    #             # 计算用户级 PF 得分
    #             ue_spectral_eff = np.mean(raw_csi, axis=1)  # 简化：用户平均 CSI
    #             scores = []
    #             for u_idx in slice_ues:
    #                 inst_rate = ue_spectral_eff[u_idx] * (alloc_limit * self.bandwidth / self.alloc_unit_count)
    #                 buffer_demand = buffer_occ[u_idx] * max_buffer_pkts[u_idx] * pkt_sizes[u_idx]
    #                 achievable_rate = min(inst_rate, buffer_demand)
    #                 avg_rate = hist_thr[u_idx] + 1e-6
    #                 pf_score = achievable_rate / avg_rate
    #                 scores.append(pf_score)
    #
    #             scores = np.array(scores)
    #             sum_scores = np.sum(scores)
    #
    #             if sum_scores > 0:
    #                 # 计算每个用户应得的 RB 数量
    #                 raw_counts = (scores / sum_scores) * alloc_limit
    #                 int_counts = np.floor(raw_counts).astype(int)
    #                 remainders = raw_counts - int_counts
    #                 diff = alloc_limit - np.sum(int_counts)
    #                 if diff > 0:
    #                     indices = np.argsort(remainders)[-diff:][::-1]
    #                     int_counts[indices] += 1
    #
    #                 # 分配具体的 RB (Greedy Max-CSI per user)
    #                 # 这是一个优化：既然确定了用户要拿多少个，就让他去 free_prbs 里挑对自己最好的
    #
    #                 # 这里的逻辑稍微复杂：为了避免循环嵌套过深，我们简化为：
    #                 # 按得分高低顺序，让用户轮流挑走分配给他的 RB 数量
    #                 user_order = np.argsort(scores)[::-1]  # 得分高的先挑
    #
    #                 current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
    #
    #                 for u_idx_idx in user_order:  # 遍历排序后的索引
    #                     u_idx = slice_ues[u_idx_idx]
    #                     count = int_counts[u_idx_idx]
    #
    #                     if count > 0:
    #                         # 找出该用户在 current_free_prbs 中最好的 count 个 RB
    #                         user_gains = raw_csi[u_idx, current_free_prbs]
    #                         best_indices = np.argsort(user_gains)[-count:]  # 最好的 count 个的索引
    #
    #                         for local_idx in best_indices:
    #                             r = current_free_prbs[local_idx]
    #                             allocation_matrix[u_idx, r] = power_per_prb
    #                             oneshot_action[r] = u_idx + 1
    #                             occupied_prbs.add(r)
    #
    #                         # 更新 current_free_prbs (移除已用的)
    #                         # 效率优化：因为是从列表中移除，且 count 不大，可以直接重建
    #                         current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
    #
    #             else:
    #                 # Fallback to RR
    #                 pass  # 简化起见，此处省略 Fallback 代码，实际应复制 RR 逻辑
    #
    #         # 3. Max Throughput (MT)
    #         elif scheduler_id == 2:
    #             # 逻辑与 PF 类似，只是得分不同
    #             scores = []
    #             for u_idx in slice_ues:
    #                 # MT Score (Clipped by buffer)
    #                 ue_spectral_eff = np.mean(raw_csi, axis=1)
    #                 inst_rate = ue_spectral_eff[u_idx] * (alloc_limit * self.bandwidth / self.alloc_unit_count)
    #                 buffer_demand = buffer_occ[u_idx] * max_buffer_pkts[u_idx] * pkt_sizes[u_idx]
    #                 mt_score = min(inst_rate, buffer_demand)
    #                 scores.append(mt_score)
    #
    #             scores = np.array(scores)
    #             sum_scores = np.sum(scores)
    #
    #             if sum_scores > 0:
    #                 # (完全相同的分配逻辑，复制粘贴 PF 的后半部分)
    #                 raw_counts = (scores / sum_scores) * alloc_limit
    #                 int_counts = np.floor(raw_counts).astype(int)
    #                 remainders = raw_counts - int_counts
    #                 diff = alloc_limit - np.sum(int_counts)
    #                 if diff > 0:
    #                     indices = np.argsort(remainders)[-diff:][::-1]
    #                     int_counts[indices] += 1
    #
    #                 user_order = np.argsort(scores)[::-1]
    #                 current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
    #
    #                 for u_idx_idx in user_order:
    #                     u_idx = slice_ues[u_idx_idx]
    #                     count = int_counts[u_idx_idx]
    #                     if count > 0:
    #                         user_gains = raw_csi[u_idx, current_free_prbs]
    #                         best_indices = np.argsort(user_gains)[-count:]
    #                         for local_idx in best_indices:
    #                             r = current_free_prbs[local_idx]
    #                             allocation_matrix[u_idx, r] = power_per_prb
    #                             oneshot_action[r] = u_idx + 1
    #                             occupied_prbs.add(r)
    #                         current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
    #
    #     # =========================================================
    #     # Phase 3: Final Sweep (全局捡漏)
    #     # =========================================================
    #     free_rb_indices = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
    #
    #     if len(free_rb_indices) > 0:
    #         hungry_users = []
    #         for u in range(self.max_users):
    #             if buffer_occ[u] > 1e-6:
    #                 hungry_users.append(u)
    #
    #         if len(hungry_users) > 0:
    #             for r in free_rb_indices:
    #                 best_u = -1
    #                 best_gain = -1e9
    #                 for u in hungry_users:
    #                     gain = raw_csi[u, r]
    #                     if gain > best_gain:
    #                         best_gain = gain
    #                         best_u = u
    #
    #                 if best_u != -1:
    #                     allocation_matrix[best_u, r] = power_per_prb
    #                     oneshot_action[r] = best_u + 1
    #
    #     return allocation_matrix, oneshot_action
    #


    def _map_discrete_actions_to_prbs_native(self, inter_quotas: np.ndarray, intra_modes: np.ndarray):
        """
        原生调度器 (Hybrid V3: SLA-Triage + PF-Sweep)
        """
        # --- 0. 初始化 ---
        allocation_matrix = np.zeros((self.max_users, self.alloc_unit_count), dtype=np.float32)
        power_per_prb = self.max_bs_power / self.alloc_unit_count
        oneshot_action = np.zeros(self.alloc_unit_count, dtype=int)
        occupied_prbs = set()

        slice_assoc = self.components.slices.ue_assoc
        latest_obs = self.last_raw_obs

        # 数据准备
        raw_csi = latest_obs.get('target_cell_power')
        if raw_csi is None: raw_csi = np.zeros((self.max_users, self.alloc_unit_count))
        if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi)
        if raw_csi.shape[0] == self.alloc_unit_count: raw_csi = raw_csi.T

        buffer_occ = latest_obs.get('buffer_occupancies', np.zeros(self.max_users))
        raw_lat = latest_obs.get('buffer_latencies', np.zeros(self.max_users))
        pkt_sizes = self.components.ues.pkt_sizes
        max_buffer_pkts = self.components.ues.max_buffer_pkts

        user_prio = slice_assoc.T @ latest_obs.get("slice_priority", np.zeros(self.num_slices))
        user_sla_limits = self._get_user_sla_limits()
        # A3 消融会跳过 Phase-1；这里给出默认空集合，确保后续 Phase-3 引用安全。
        critical_users = np.array([], dtype=int)

        # 历史吞吐量 (用于 PF 计算)
        if len(self.last_unformatted_obs_deque) > 0:
            hist_thr = np.mean([
                obs.get('pkt_effective_thr', np.zeros(self.max_users))
                for obs in self.last_unformatted_obs_deque
            ], axis=0)
        else:
            hist_thr = np.zeros(self.max_users)

        remaining_slice_quotas = inter_quotas.copy()

        # ==========================================================
        # Phase 1: Critical Rescue (SLA-Aware Triage)
        # A3 消融：disable_critical_rescue=True 时跳过此阶段
        # ==========================================================
        current_rescue_rbs = 0
        if not self.disable_critical_rescue:
            MAX_RESCUE_RBS = int(self.alloc_unit_count * 0.60)

            critical_users = np.where((user_prio > 0) | (user_sla_limits < 30.0))[0]

            emergency_queue = self._build_emergency_queue(critical_users, raw_lat, buffer_occ, user_sla_limits)
            emergency_queue.sort(key=lambda x: x[0], reverse=True)

            for ratio, u_idx in emergency_queue:
                if current_rescue_rbs >= MAX_RESCUE_RBS: break

                avg_gain = np.mean(raw_csi[u_idx])
                rescue_budget = 5 if avg_gain < 1e-10 else 3

                allocated = 0
                while allocated < rescue_budget:
                    if current_rescue_rbs >= MAX_RESCUE_RBS: break

                    best_prb = -1
                    best_gain = -1e9
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
                        allocated += 1
                        current_rescue_rbs += 1

                        s_indices = np.where(slice_assoc[:, u_idx] == 1)[0]
                        if len(s_indices) > 0:
                            remaining_slice_quotas[s_indices[0]] -= 1
                    else:
                        break

        # ==========================================================
        # Phase 2: Native Schedulers (Quota Based)
        # ==========================================================
        # 按 urgency 排序后的切片顺序执行（与 system model 一致）
        for s_idx in self.last_sorted_slice_indices:
            quota = remaining_slice_quotas[s_idx]
            if quota <= 0: continue

            slice_ues = np.where(slice_assoc[s_idx] == 1)[0]
            if len(slice_ues) == 0: continue

            scheduler_id = intra_modes[s_idx]

            free_prbs_list = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
            if not free_prbs_list: break

            alloc_limit = min(quota, len(free_prbs_list))
            if alloc_limit <= 0: continue

            # --- 1. Round Robin ---
            if scheduler_id == 0:
                base_alloc = alloc_limit // len(slice_ues)
                remainder = alloc_limit % len(slice_ues)

                rb_ptr = 0
                for i, u_idx in enumerate(slice_ues):
                    count = base_alloc + (1 if i < remainder else 0)
                    if count > 0:
                        for k in range(count):
                            r = free_prbs_list[rb_ptr + k]
                            allocation_matrix[u_idx, r] = power_per_prb
                            oneshot_action[r] = u_idx + 1
                            occupied_prbs.add(r)
                        rb_ptr += count

            # --- 2. Proportional Fairness (PF) ---
            elif scheduler_id == 1:
                ue_spectral_eff = np.mean(raw_csi, axis=1)
                scores = []
                for u_idx in slice_ues:
                    inst_rate = ue_spectral_eff[u_idx] * (alloc_limit * self.bandwidth / self.alloc_unit_count)
                    buffer_demand = buffer_occ[u_idx] * max_buffer_pkts[u_idx] * pkt_sizes[u_idx]
                    achievable_rate = min(inst_rate, buffer_demand)
                    avg_rate = hist_thr[u_idx] + 1e-6
                    pf_score = achievable_rate / avg_rate
                    scores.append(pf_score)

                scores = np.array(scores)
                sum_scores = np.sum(scores)

                if sum_scores > 0:
                    raw_counts = (scores / sum_scores) * alloc_limit
                    int_counts = np.floor(raw_counts).astype(int)
                    remainders = raw_counts - int_counts
                    diff = alloc_limit - np.sum(int_counts)
                    if diff > 0:
                        indices = np.argsort(remainders)[-diff:][::-1]
                        int_counts[indices] += 1

                    user_order = np.argsort(scores)[::-1]
                    current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]

                    for u_idx_idx in user_order:
                        u_idx = slice_ues[u_idx_idx]
                        count = int_counts[u_idx_idx]
                        if count > 0:
                            user_gains = raw_csi[u_idx, current_free_prbs]
                            best_indices = np.argsort(user_gains)[-count:]
                            for local_idx in best_indices:
                                r = current_free_prbs[local_idx]
                                allocation_matrix[u_idx, r] = power_per_prb
                                oneshot_action[r] = u_idx + 1
                                occupied_prbs.add(r)
                            current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]

            # --- 3. Max Throughput (MT) ---
            elif scheduler_id == 2:
                scores = []
                for u_idx in slice_ues:
                    ue_spectral_eff = np.mean(raw_csi, axis=1)
                    inst_rate = ue_spectral_eff[u_idx] * (alloc_limit * self.bandwidth / self.alloc_unit_count)
                    buffer_demand = buffer_occ[u_idx] * max_buffer_pkts[u_idx] * pkt_sizes[u_idx]
                    mt_score = min(inst_rate, buffer_demand)
                    scores.append(mt_score)

                scores = np.array(scores)
                sum_scores = np.sum(scores)

                if sum_scores > 0:
                    raw_counts = (scores / sum_scores) * alloc_limit
                    int_counts = np.floor(raw_counts).astype(int)
                    remainders = raw_counts - int_counts
                    diff = alloc_limit - np.sum(int_counts)
                    if diff > 0:
                        indices = np.argsort(remainders)[-diff:][::-1]
                        int_counts[indices] += 1

                    user_order = np.argsort(scores)[::-1]
                    current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]

                    for u_idx_idx in user_order:
                        u_idx = slice_ues[u_idx_idx]
                        count = int_counts[u_idx_idx]
                        if count > 0:
                            user_gains = raw_csi[u_idx, current_free_prbs]
                            best_indices = np.argsort(user_gains)[-count:]
                            for local_idx in best_indices:
                                r = current_free_prbs[local_idx]
                                allocation_matrix[u_idx, r] = power_per_prb
                                oneshot_action[r] = u_idx + 1
                                occupied_prbs.add(r)
                            current_free_prbs = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]

        # =========================================================
        # [UPGRADE] Phase 3: Global PF-Sweep (公平捡漏)
        # =========================================================
        # 从 Max-CQI 升级为 Global Proportional Fairness
        # 避免 "饿死信道差的用户"

        free_rb_indices = [r for r in range(self.alloc_unit_count) if r not in occupied_prbs]
        num_free = len(free_rb_indices)

        if num_free > 0:
            hungry_users = []
            pf_scores = []

            # 1. 计算全网用户的 PF 得分
            ue_spectral_eff = np.mean(raw_csi, axis=1)

            for u in range(self.max_users):
                if buffer_occ[u] > 1e-6:
                    inst_rate = ue_spectral_eff[u] * (num_free * self.bandwidth / self.alloc_unit_count)
                    buffer_demand = buffer_occ[u] * max_buffer_pkts[u] * pkt_sizes[u]
                    achievable_rate = min(inst_rate, buffer_demand)
                    avg_rate = hist_thr[u] + 1e-6

                    score = achievable_rate / avg_rate

                    # [Bonus]: 如果是 HP 用户，PF 得分加倍，增加抢占力
                    if u in critical_users:
                        score *= 2.0

                    hungry_users.append(u)
                    pf_scores.append(score)

            if hungry_users:
                # 2. 按 PF 得分分配数量
                pf_scores = np.array(pf_scores)
                sum_scores = np.sum(pf_scores)

                if sum_scores > 0:
                    raw_counts = (pf_scores / sum_scores) * num_free
                    int_counts = np.floor(raw_counts).astype(int)
                    # 处理残差
                    remainders = raw_counts - int_counts
                    diff = num_free - np.sum(int_counts)
                    if diff > 0:
                        indices = np.argsort(remainders)[-diff:][::-1]
                        int_counts[indices] += 1

                    # 3. RB-level PF 分配：在给定份额下，对每个 RB 选择 PF 指标最高的用户
                    # PF_metric(u, r) = CSI(u, r) / avg_rate(u)，与 Phase 2 的 PF 目标一致。
                    remaining_counts = int_counts.copy()
                    for r in free_rb_indices:
                        eligible = np.where(remaining_counts > 0)[0]
                        if eligible.size == 0:
                            break

                        best_local_idx = -1
                        best_metric = -1e9
                        for local_idx in eligible:
                            u_idx = hungry_users[local_idx]
                            pf_metric = raw_csi[u_idx, r] / (hist_thr[u_idx] + 1e-6)
                            if u_idx in critical_users:
                                pf_metric *= 2.0
                            if pf_metric > best_metric:
                                best_metric = pf_metric
                                best_local_idx = local_idx

                        if best_local_idx >= 0:
                            u_idx = hungry_users[best_local_idx]
                            allocation_matrix[u_idx, r] = power_per_prb
                            oneshot_action[r] = u_idx + 1
                            occupied_prbs.add(r)
                            remaining_counts[best_local_idx] -= 1

        return allocation_matrix, oneshot_action, current_rescue_rbs

    # def _build_emergency_queue(self, hp_users, raw_lat, buffer_occ, user_sla_limits):
    #     """辅助函数：构建紧急队列"""
    #     emergency_queue = []
    #     for u in hp_users:
    #         danger_ratio = raw_lat[u] / (user_sla_limits[u] + 1e-6)
    #         latency_risk = danger_ratio > 0.4
    #         overflow_risk = buffer_occ[u] > 0.8
    #
    #         if latency_risk or overflow_risk:
    #             highest_risk = max(danger_ratio, buffer_occ[u])
    #             emergency_queue.append((highest_risk, u))
    #     return emergency_queue

    # === 补充的辅助函数 (必须添加) ===

    def _get_user_sla_limits(self):
        """辅助函数：获取用户级 SLA 限制"""
        slice_assoc = self.components.slices.ue_assoc
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
        return user_sla_limits

    def _build_emergency_queue(self, critical_users, raw_lat, buffer_occ, user_sla_limits):
        """辅助函数：构建紧急队列 (带优先级加权)"""
        # 获取优先级信息用于加权
        slice_assoc = self.components.slices.ue_assoc
        slice_prio = self.last_raw_obs.get("slice_priority", np.zeros(self.num_slices))
        user_prio = slice_assoc.T @ slice_prio

        emergency_queue = []
        for u in critical_users:
            danger_ratio = raw_lat[u] / (user_sla_limits[u] + 1e-6)

            # 基础风险分
            risk_score = max(danger_ratio, buffer_occ[u])

            # [Optimization]: 如果是真 HP (Priority > 0)，风险分放大 1.1 倍
            # 这保证了在同样紧急的情况下，HP 优先于 "低时延 NHP" (如 VR)
            if user_prio[u] > 0:
                risk_score *= 1.1

            # 门槛检查 (0.4)
            if risk_score > 0.4:
                emergency_queue.append((risk_score, u))

        return emergency_queue

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _prefill_observation(self):
        """执行一次物理模拟以填充初始状态，并推入 deque 防止 cold-start error"""
        dummy_alloc = np.zeros((self.max_users, self.num_phys_rbs))
        self.business_executor.temp_rb_allocation[:] = dummy_alloc
        self.business_executor.temp_rb_ues_association.fill(0)

        # Execute Step 0
        metrics = self.business_executor.execute_tti_physics(0, self.current_episode_idx)
        metrics["intent_drift"] = np.zeros((self.num_slices, 5, 3))
        self._update_priority_in_obs(metrics)
        self.last_raw_obs = metrics

        # [FIX] 将初始观测推入 deque
        self.last_unformatted_obs_deque.appendleft(metrics)

    def _get_slice_active_mask(self):
        slice_assoc = self.components.slices.ue_assoc
        return (np.sum(slice_assoc, axis=1) > 0).astype(float)

    def _update_priority_in_obs(self, raw_obs):
        prio = np.zeros(self.num_slices, dtype=np.float32)
        for i in range(self.num_slices):
            prio[i] = raw_obs.get("slice_req", {}).get(f'slice_{i}', {}).get('priority', 0)
        raw_obs["slice_priority"] = prio

    def _get_dummy_obs(self):
        return {
            "inter_feat": np.zeros((self.num_slices, 4), dtype=np.float32),
            "intra_feat": np.zeros((self.max_users, 5), dtype=np.float32),
            "global_feat": np.zeros((2,), dtype=np.float32)
        }

    def _get_hierarchical_observation(self):
        if self.last_raw_obs is None: return self._get_dummy_obs()
        raw = self.last_raw_obs

        # Inter: 4 维
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

        # Intra: 5 维
        intra_feat = np.zeros((self.max_users, 5), dtype=np.float32)
        buffer = raw.get("buffer_occupancies", np.zeros(self.max_users))

        raw_csi = raw.get('target_cell_power', np.zeros((self.max_users, self.num_phys_rbs)))
        if raw_csi.ndim > 2: raw_csi = np.squeeze(raw_csi).T
        mean_gain = np.mean(raw_csi, axis=1) + 1e-30
        csi_norm = np.clip(10 * np.log10(mean_gain) / 100.0 + 1.0, 0, 1)

        # HOL & Arrival
        raw_lat = raw.get('buffer_latencies', np.zeros(self.max_users))
        hol_delay_norm = np.clip(raw_lat / 100.0, 0, 1.0)

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

        global_feat = np.array([self.current_timestep / 1000.0, np.sum(self.last_inter_alloc_ratio)], dtype=np.float32)
        return {"inter_feat": inter_feat, "intra_feat": intra_feat, "global_feat": global_feat}

    def _compute_slice_info(self, metrics):
        """从 metrics 提取供 reward function 使用的结构化信号。

        Returns 9-tuple:
            (slice_min_margins, slice_mean_margins, slice_metric_margins,
             is_hp, num_active_slices, prev_min_margins, prev_mean_margins,
             slice_mean_buffer_occ, slice_max_buffer_occ)
        """
        raw_intent_drift = metrics.get("intent_drift")  # (num_slices, users_per_slice, 3)
        slice_assoc = self.components.slices.ue_assoc   # (num_slices, max_users)
        slice_reqs = metrics.get('slice_req', {})
        buffer_occ = metrics.get("buffer_occupancies", np.zeros(self.max_users))

        slice_min_margins = []
        slice_mean_margins = []
        slice_metric_margins = []  # list of {"thr": float, "rel": float, "lat": float}
        is_hp = []
        slice_mean_buffer_occ = []
        slice_max_buffer_occ = []

        for s_idx in range(self.num_slices):
            req = slice_reqs.get(f'slice_{s_idx}', {})
            priority = req.get('priority', 0)
            active_users = np.where(slice_assoc[s_idx] > 0)[0]
            if len(active_users) == 0:
                continue

            limit = min(len(active_users), self.users_per_slice)

            # --- composite drifts (across all metrics, all users) ---
            all_valid = []
            metric_valid = {m: [] for m in ['thr', 'rel', 'lat']}
            metric_names = ['thr', 'rel', 'lat']
            for u_local_idx in range(limit):
                for m_idx, m_name in enumerate(metric_names):
                    drift = raw_intent_drift[s_idx, u_local_idx, m_idx]
                    if drift > -1.5:
                        all_valid.append(drift)
                        metric_valid[m_name].append(drift)

            if all_valid:
                slice_min_margins.append(float(np.min(all_valid)))
                slice_mean_margins.append(float(np.mean(all_valid)))
            else:
                slice_min_margins.append(0.0)
                slice_mean_margins.append(0.0)

            per_metric = {}
            for m_name in ['thr', 'rel', 'lat']:
                vals = metric_valid[m_name]
                per_metric[m_name] = float(np.mean(vals)) if vals else 0.0
            slice_metric_margins.append(per_metric)

            is_hp.append(bool(priority > 0))

            # --- buffer occupancy stats for this slice ---
            users_in_slice = np.where(slice_assoc[s_idx] > 0)[0]
            if len(users_in_slice) > 0:
                bufs = buffer_occ[users_in_slice]
                slice_mean_buffer_occ.append(float(np.mean(bufs)))
                slice_max_buffer_occ.append(float(np.max(bufs)))
            else:
                slice_mean_buffer_occ.append(0.0)
                slice_max_buffer_occ.append(0.0)

        n = len(slice_min_margins)

        # snapshot prev, then update cache
        prev_min = list(self._prev_min_margins[:n])
        prev_mean = list(self._prev_mean_margins[:n])
        self._prev_min_margins[:n] = slice_min_margins
        self._prev_mean_margins[:n] = slice_mean_margins

        return (
            slice_min_margins,
            slice_mean_margins,
            slice_metric_margins,
            is_hp,
            n,
            prev_min,
            prev_mean,
            slice_mean_buffer_occ,
            slice_max_buffer_occ,
        )

    def _builtin_compute_reward(
        self,
        slice_min_margins,
        slice_mean_margins,
        slice_metric_margins,
        is_hp,
        num_active_slices,
        prev_min_margins,
        prev_mean_margins,
        slice_mean_buffer_occ,
        slice_max_buffer_occ,
    ):
        """当前手工设计的 reward，等价于原 _calculate_simple_reward（不含 clamp）。
        此方法为 compute_reward_fn 的 fallback，语义与原实现一致。
        """
        if num_active_slices == 0:
            return 0.0

        slice_scores = np.array(slice_min_margins, dtype=np.float64)
        slice_priorities = np.array([1 if hp else 0 for hp in is_hp], dtype=np.float64)

        # 1. NHP Capping
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
            base_reward = float(np.mean(capped_scores))

        # 4. Risk Penalty（近似等价，使用 per-slice max buffer occ）
        risk_loss = 0.0
        max_buf = np.array(slice_max_buffer_occ, dtype=np.float64)
        danger_mask = max_buf > 0.8
        if np.any(danger_mask):
            risk_loss = np.sum(np.exp(max_buf[danger_mask] * 5.0))
        risk_coef = getattr(self, 'reward_weights', {}).get('risk_coef', 0.2)
        risk_penalty = -1.0 * risk_loss * risk_coef

        return float(base_reward + hp_penalty + risk_penalty)

    def _calculate_simple_reward(self, metrics):
        """计算单步 reward，支持 compute_reward_fn 插拔。"""
        _REWARD_LOWER, _REWARD_UPPER = -10.0, 10.0
        info = self._compute_slice_info(metrics)
        reward_fn = getattr(self, 'compute_reward_fn', self._builtin_compute_reward)
        reward = float(np.clip(reward_fn(*info), _REWARD_LOWER, _REWARD_UPPER))
        return reward

    def _compute_risk_penalty(self, metrics):
        """原始 per-user risk penalty（保留供参考，不再被 _calculate_simple_reward 调用）。"""
        buffer_occ = metrics.get("buffer_occupancies", np.zeros(self.max_users))
        risk_loss = 0.0
        danger_mask = buffer_occ > 0.8
        if np.any(danger_mask):
            risk_loss = np.sum(np.exp(buffer_occ[danger_mask] * 5.0))
        risk_coef = getattr(self, 'reward_weights', {}).get('risk_coef', 0.2)
        return -1.0 * risk_loss * risk_coef

    def _handle_testing_save(self):
        self.business_executor.save_metric(self.config.model_name, f"ep_{self.current_episode_idx}.npz")
        if hasattr(self.business_executor, 'reset_metric'):
            self.business_executor.reset_metric()

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
            info[f"meta/slice_{s_idx}_demand_score"] = float(self.last_demand_scores[s_idx])
            rank_pos = int(np.where(self.last_sorted_slice_indices == s_idx)[0][0])
            info[f"meta/slice_{s_idx}_rank_pos"] = rank_pos
        info["meta/rescue_rb_count"] = int(self.last_rescue_rbs)
        info["meta/rescue_trigger"] = 1 if self.last_rescue_rbs > 0 else 0
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
                    mean_val = np.mean(valid_vals)
                    details[f"drift/{key}"] = mean_val
                    details[f"violation/{key}"] = 1.0 if mean_val < 0 else 0.0
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