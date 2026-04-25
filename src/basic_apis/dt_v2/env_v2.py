"""
EnvV2: HierarchicalSlicingEnv with:
  - Richer observation (channel stats in inter_feat/intra_feat)
  - Explicit slice ordering in action space (replaces urgency_rank)
  - Violation-based reward

Does NOT modify any original source files.
"""
import numpy as np
from gymnasium import spaces
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.utils import intent_drift_calc

INTER_DIM = 8   # original 4 + 4 new channel/demand features
INTRA_DIM = 7   # original 5 + 2 new features
HP_WEIGHT = 10.0
NHP_WEIGHT = 1.0
SATISFIED_BONUS = 0.05  # fixed bonus per satisfied slice (not proportional to margin)


class HierarchicalSlicingEnvV2(HierarchicalSlicingEnv):

    def __init__(self, env_settings, np_random, path_context):
        super().__init__(env_settings, np_random, path_context)

        # Override observation space
        self.observation_space = spaces.Dict({
            "inter_feat": spaces.Box(low=-10, high=10, shape=(self.num_slices, INTER_DIM), dtype=np.float32),
            "intra_feat": spaces.Box(low=-10, high=10, shape=(self.max_users, INTRA_DIM), dtype=np.float32),
            "global_feat": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })

        # Override action space: codebook(11) + ordering(5 x 5) + intra(5 x 3)
        num_inter_modes = len(self.inter_quota_patterns)
        action_dims = [num_inter_modes] + [self.num_slices] * self.num_slices + [3] * self.num_slices
        self.action_space = spaces.MultiDiscrete(action_dims)

        self._forced_ordering = None

    def set_forced_ordering(self, ordering):
        if ordering is not None:
            self._forced_ordering = np.array(ordering, dtype=int)
        else:
            self._forced_ordering = None

    # -----------------------------------------------------------------
    # Step: parse new action format, use agent ordering
    # -----------------------------------------------------------------
    def step(self, action: np.ndarray):
        inter_mode = action[0]
        order_actions = action[1:1 + self.num_slices]
        intra_modes = action[1 + self.num_slices:]

        raw_pattern = self.inter_quota_patterns[inter_mode]
        sorted_quotas = np.sort(raw_pattern)[::-1]

        if self._forced_ordering is not None:
            sorted_slice_indices = self._forced_ordering.copy()
        else:
            sorted_slice_indices = np.argsort(order_actions)[::-1]

        demand_scores = np.array(order_actions, dtype=np.float32)
        self.last_demand_scores = demand_scores
        self.last_sorted_slice_indices = sorted_slice_indices

        inter_quotas = np.zeros(self.num_slices, dtype=int)
        for rank, s_idx in enumerate(sorted_slice_indices):
            inter_quotas[s_idx] = sorted_quotas[rank]

        allocation_matrix, oneshot_action, rescue_rbs = self._map_discrete_actions_to_prbs_native(
            inter_quotas, intra_modes
        )
        self.last_rescue_rbs = int(rescue_rbs)
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

        reward = self._violation_reward(metrics)
        obs = self._get_v2_observation()
        done = (self.current_timestep >= self.max_timesteps)
        info = self._get_info(metrics, reward)

        return obs, reward, done, False, info

    # -----------------------------------------------------------------
    # Dense reward: violation penalty + positive-margin bonus
    # -----------------------------------------------------------------
    def _violation_reward(self, metrics):
        drift = metrics.get("intent_drift")
        prio = metrics.get("slice_priority", np.zeros(self.num_slices))
        if drift is None:
            return 0.0
        reward = 0.0
        active_slices = 0
        slice_assoc = self.components.slices.ue_assoc
        for s in range(self.num_slices):
            if np.sum(slice_assoc[s]) == 0:
                continue
            active_slices += 1
            mean_drift = np.mean(drift[s])
            w = HP_WEIGHT if prio[s] > 0 else NHP_WEIGHT
            if mean_drift < 0:
                reward += mean_drift * w
            else:
                reward += SATISFIED_BONUS * w
        if active_slices > 0:
            reward /= active_slices
        return float(reward)

    # -----------------------------------------------------------------
    # Enhanced observation with channel statistics
    # -----------------------------------------------------------------
    def _get_v2_observation(self):
        if self.last_raw_obs is None:
            return self._get_v2_dummy_obs()

        raw = self.last_raw_obs
        slice_assoc = self.components.slices.ue_assoc

        # --- Shared computations ---
        drift = raw.get("intent_drift", np.zeros((self.num_slices, 5, 3)))
        prio = raw.get("slice_priority", np.zeros(self.num_slices))
        in_bits = raw.get("pkt_incoming_bits", np.zeros(self.max_users))
        slice_traffic = (slice_assoc @ in_bits) / 1e6

        # Spectral efficiency: shape depends on channel impl, typically (1, U, R) or (U, R)
        raw_se = raw.get("spectral_efficiencies", np.zeros((1, self.max_users, self.num_phys_rbs)))
        if raw_se.ndim == 3:
            se = raw_se[0]  # (U, R)
        else:
            se = raw_se     # (U, R)
        user_se_mean = np.mean(se, axis=1)  # per-user mean spectral eff

        dropped = raw.get("dropped_pkts", np.zeros(self.max_users))
        effective_thr = raw.get("pkt_effective_thr", np.zeros(self.max_users))

        # --- inter_feat: (num_slices, 8) ---
        inter_feat = np.zeros((self.num_slices, INTER_DIM), dtype=np.float32)
        slice_drift_mean = np.mean(np.maximum(drift, -1.0), axis=(1, 2))

        for s in range(self.num_slices):
            users = np.where(slice_assoc[s] > 0)[0]
            n_users = len(users)

            inter_feat[s, 0] = prio[s]
            inter_feat[s, 1] = slice_drift_mean[s]
            inter_feat[s, 2] = np.clip(slice_traffic[s], 0, 5.0)
            inter_feat[s, 3] = self.last_inter_alloc_ratio[s]

            # New dim 4: slice aggregate spectral efficiency
            slice_se = np.mean(user_se_mean[users]) if n_users > 0 else 0.0
            inter_feat[s, 4] = np.clip(slice_se / 5.0, 0, 1.0)

            # New dim 5: demand/supply pressure
            alloc_capacity = self.last_inter_alloc_ratio[s] * self.alloc_unit_count * max(slice_se, 0.01)
            demand = slice_traffic[s]
            inter_feat[s, 5] = np.clip(demand / max(alloc_capacity, 0.01), 0, 5.0)

            # New dim 6: drop rate
            slice_in = np.sum(in_bits[users]) if n_users > 0 else 0.0
            slice_drop = np.sum(dropped[users]) if n_users > 0 else 0.0
            inter_feat[s, 6] = np.clip(slice_drop / max(slice_in, 1.0), 0, 1.0)

            # New dim 7: number of active users (normalized)
            inter_feat[s, 7] = n_users / self.users_per_slice

        # --- intra_feat: (max_users, 7) ---
        intra_feat = np.zeros((self.max_users, INTRA_DIM), dtype=np.float32)
        buffer = raw.get("buffer_occupancies", np.zeros(self.max_users))

        raw_csi = raw.get('target_cell_power', np.zeros((self.max_users, self.num_phys_rbs)))
        if raw_csi.ndim > 2:
            raw_csi = np.squeeze(raw_csi).T
        mean_gain = np.mean(raw_csi, axis=1) + 1e-30
        csi_norm = np.clip(10 * np.log10(mean_gain) / 100.0 + 1.0, 0, 1)

        raw_lat = raw.get('buffer_latencies', np.zeros(self.max_users))
        hol_delay_norm = np.clip(raw_lat / 100.0, 0, 1.0)

        user_prio = slice_assoc.T @ prio
        user_drift_min = np.zeros(self.max_users)
        for s in range(self.num_slices):
            users = np.where(slice_assoc[s] > 0)[0]
            for i, u in enumerate(users):
                if i < 5:
                    user_drift_min[u] = np.min(drift[s, i, :])

        for u in range(self.max_users):
            intra_feat[u, 0] = buffer[u]
            intra_feat[u, 1] = user_prio[u]
            intra_feat[u, 2] = user_drift_min[u]
            intra_feat[u, 3] = csi_norm[u]
            intra_feat[u, 4] = hol_delay_norm[u]
            # New dim 5: user mean spectral efficiency
            intra_feat[u, 5] = np.clip(user_se_mean[u] / 5.0, 0, 1.0)
            # New dim 6: throughput achieved ratio
            intra_feat[u, 6] = np.clip(effective_thr[u] / max(in_bits[u], 1.0), 0, 2.0)

        global_feat = np.array([
            self.current_timestep / 1000.0,
            np.sum(self.last_inter_alloc_ratio)
        ], dtype=np.float32)

        return {"inter_feat": inter_feat, "intra_feat": intra_feat, "global_feat": global_feat}

    def _get_v2_dummy_obs(self):
        return {
            "inter_feat": np.zeros((self.num_slices, INTER_DIM), dtype=np.float32),
            "intra_feat": np.zeros((self.max_users, INTRA_DIM), dtype=np.float32),
            "global_feat": np.zeros((2,), dtype=np.float32),
        }

    # Override reset to return v2 observation
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._get_v2_observation(), info
