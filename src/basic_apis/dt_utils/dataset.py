import torch
import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class HierarchicalDTDataset(Dataset):
    def __init__(self, dataset_path, context_len=20, rtg_scale=1.0, metadata_path="metadata.json"):
        self.context_len = context_len
        self.rtg_scale = rtg_scale

        # === 1. 加载标准化元数据 ===
        self.normalize = False
        if not os.path.exists(metadata_path):
            potential_path = os.path.join(os.path.dirname(dataset_path), "metadata.json")
            if os.path.exists(potential_path):
                metadata_path = potential_path

        if os.path.exists(metadata_path):
            print(f"📊 Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.meta = json.load(f)

            self.stats = {}
            obs_stats = self.meta.get("obs_stats", self.meta)

            for k in ['inter', 'intra', 'global']:
                if k in obs_stats:
                    self.stats[k] = {
                        'mean': np.array(obs_stats[k]['mean'], dtype=np.float32),
                        'std': np.array(obs_stats[k]['std'], dtype=np.float32) + 1e-6
                    }
            self.normalize = True
        else:
            print(f"⚠️ Metadata not found. Skipping normalization.")

        # === 2. 加载轨迹数据 ===
        self.trajectories = []
        self.indices = []

        if os.path.isdir(dataset_path):
            print(f"📂 Scanning directory: {dataset_path} ...")
            files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
            files.sort()

            for f_name in tqdm(files, desc="Loading Trajectories"):
                full_path = os.path.join(dataset_path, f_name)
                try:
                    with open(full_path, 'rb') as f:
                        traj = pickle.load(f)
                        self._process_and_append(traj)
                except Exception as e:
                    print(f"❌ Error loading {f_name}: {e}")
        else:
            print(f"📖 Loading single file: {dataset_path} ...")
            with open(dataset_path, 'rb') as f:
                raw_trajs = pickle.load(f)
                if isinstance(raw_trajs, list):
                    for traj in raw_trajs:
                        self._process_and_append(traj)
                else:
                    self._process_and_append(raw_trajs)

        print(f"✅ Dataset Loaded. Total Trajectories: {len(self.trajectories)}")

    def _process_and_append(self, traj):
        """
        处理单个轨迹：清洗、变换、以及【关键优化】格式转换
        """
        # 1. 基础数值清洗
        traj['rewards'] = np.nan_to_num(traj['rewards'], nan=0.0, posinf=0.0, neginf=-100.0)
        traj['actions'] = traj['actions'].astype(np.int64)

        # 2. RTG SymLog
        raw_rtg = self._discount_cumsum(traj['rewards'], gamma=1.0)
        traj['rtg_transformed'] = np.sign(raw_rtg) * np.log1p(np.abs(raw_rtg))

        # 3. [性能优化核心] 将 Observation 从 List[Dict] 转为 Dict[Array]
        # 原始数据中 obs 是一个列表，里面每个元素是 dict
        # 我们要把它变成一个 dict，里面每个元素是巨大的 array (Time, Features)

        raw_obs_list = traj['observations']

        # 预分配内存以加速
        T = len(raw_obs_list)

        # 假设第一帧包含了所有 key
        keys = ['inter_feat', 'intra_feat', 'global_feat']

        optimized_obs = {}
        for k in keys:
            # 拿到第一帧以确定形状
            if k in raw_obs_list[0]:
                shape = raw_obs_list[0][k].shape
                # 创建全零数组 [T, ...]
                arr = np.zeros((T, *shape), dtype=np.float32)

                # 填充数据并清洗 NaN
                for t, frame in enumerate(raw_obs_list):
                    val = frame[k]
                    # 此时顺便做清洗
                    val = np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)
                    arr[t] = val

                optimized_obs[k] = arr

        # 替换原始的 List[Dict]
        traj['observations'] = optimized_obs

        # 4. 存储长度
        traj_len = len(traj['actions'])
        traj['length'] = traj_len

        self.trajectories.append(traj)

        for step in range(traj_len):
            self.indices.append((len(self.trajectories) - 1, step))

    def __len__(self):
        return len(self.indices)

    # def __getitem__(self, idx):
    #     traj_idx, end_idx = self.indices[idx]
    #     traj = self.trajectories[traj_idx]
    #
    #     start_idx = end_idx - self.context_len + 1
    #     if start_idx < 0:
    #         window_start = 0
    #         pad_len = abs(start_idx)
    #     else:
    #         window_start = start_idx
    #         pad_len = 0
    #     window_end = end_idx + 1
    #
    #     obs_dict = traj['observations']
    #
    #     # === [优化后] 极速切片 ===
    #     # 现在 obs_dict['inter_feat'] 已经是 numpy array 了，直接切片，无需循环
    #     def get_normed_data(key, raw_slice):
    #         if not self.normalize: return raw_slice
    #
    #         stat_key = key.split('_')[0]
    #         if stat_key not in self.stats: return raw_slice
    #
    #         mean = self.stats[stat_key]['mean']
    #         std = self.stats[stat_key]['std']
    #
    #         safe_std = std.copy()
    #         safe_std[safe_std < 1e-2] = 1.0
    #
    #         normed = (raw_slice - mean) / safe_std
    #         return np.clip(normed, -5.0, 5.0)
    #
    #     # 直接切片，速度快几十倍
    #     inter = get_normed_data('inter_feat', obs_dict['inter_feat'][window_start:window_end])
    #     intra = get_normed_data('intra_feat', obs_dict['intra_feat'][window_start:window_end])
    #     glob = get_normed_data('global_feat', obs_dict['global_feat'][window_start:window_end])
    #
    #     # ... (后续 RTG, Action Shift, Padding 逻辑完全不变) ...
    #     rtg = traj['rtg_transformed'][window_start:window_end]
    #     actions_target = traj['actions'][window_start:window_end]
    #     timesteps = np.arange(window_start, window_end)
    #
    #     if window_start > 0:
    #         prev_act = traj['actions'][window_start - 1].reshape(1, -1)
    #         actions_input = np.concatenate([prev_act, actions_target[:-1]], axis=0)
    #     else:
    #         act_dim = actions_target.shape[-1]
    #         dummy = np.zeros((1, act_dim), dtype=np.float32)
    #         if len(actions_target) > 0:
    #             actions_input = np.concatenate([dummy, actions_target[:-1]], axis=0)
    #         else:
    #             actions_input = dummy
    #
    #     def pad_data(data, pad_len):
    #         t_data = torch.from_numpy(data).float()
    #         if pad_len == 0: return t_data
    #         shape = t_data.shape
    #         pad = torch.zeros((pad_len, *shape[1:]))
    #         return torch.cat([pad, t_data], dim=0)
    #
    #     return {
    #         "inter": pad_data(inter, pad_len),
    #         "intra": pad_data(intra, pad_len),
    #         "global": pad_data(glob, pad_len),
    #         "actions": pad_data(actions_input, pad_len),
    #         "action_targets": pad_data(actions_target, pad_len),
    #         "rtg": pad_data(rtg.reshape(-1, 1), pad_len),
    #         "timesteps": pad_data(timesteps, pad_len).long(),
    #         "mask": torch.cat([torch.zeros(pad_len), torch.ones(len(inter))])
    #     }

    def __getitem__(self, idx):
        traj_idx, end_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]

        start_idx = end_idx - self.context_len + 1
        if start_idx < 0:
            window_start = 0
            pad_len = abs(start_idx)
        else:
            window_start = start_idx
            pad_len = 0
        window_end = end_idx + 1

        obs_dict = traj['observations']

        # === 1. Observation 处理 (保持原样，省略 get_normed_data 细节以节省篇幅) ===
        def get_normed_data(key, raw_slice):
            if not self.normalize: return raw_slice
            stat_key = key.split('_')[0]
            if stat_key not in self.stats: return raw_slice
            mean = self.stats[stat_key]['mean']
            std = self.stats[stat_key]['std']
            safe_std = std.copy()
            safe_std[safe_std < 1e-2] = 1.0
            normed = (raw_slice - mean) / safe_std
            return np.clip(normed, -5.0, 5.0)

        inter = get_normed_data('inter_feat', obs_dict['inter_feat'][window_start:window_end])
        intra = get_normed_data('intra_feat', obs_dict['intra_feat'][window_start:window_end])
        glob = get_normed_data('global_feat', obs_dict['global_feat'][window_start:window_end])

        # === 2. RTG 处理 (保持原样) ===
        rtg = traj['rtg_transformed'][window_start:window_end]

        # === 3. [关键修改] Action 处理 (Multi-Discrete) ===
        # 确保动作是整数类型 [K, 6]
        actions_target = traj['actions'][window_start:window_end].astype(np.int64)

        # Action Shift: 输入 a_{t-1} 预测 a_t
        act_dim = actions_target.shape[-1]  # 应该是 6

        if window_start > 0:
            # 取前一步动作
            prev_act = traj['actions'][window_start - 1].reshape(1, -1).astype(np.int64)
            actions_input = np.concatenate([prev_act, actions_target[:-1]], axis=0)
        else:
            # t=0 时，dummy action 为全 0 整数向量
            dummy = np.zeros((1, act_dim), dtype=np.int64)
            if len(actions_target) > 0:
                actions_input = np.concatenate([dummy, actions_target[:-1]], axis=0)
            else:
                actions_input = dummy

        timesteps = np.arange(window_start, window_end)

        # === 4. Padding 函数 (支持 Int64 和 忽略值) ===
        def pad_data(data, pad_len, pad_val=0):
            # 将 numpy 转 tensor
            if data.dtype in [np.int64, np.int32, np.int16, int]:
                t_data = torch.from_numpy(data).long()
                dtype = torch.long
            else:
                t_data = torch.from_numpy(data).float()
                dtype = torch.float

            if pad_len == 0: return t_data

            # 创建 Padding
            shape = t_data.shape
            pad = torch.full((pad_len, *shape[1:]), pad_val, dtype=dtype)
            return torch.cat([pad, t_data], dim=0)

        # === 5. 返回 Batch ===
        return {
            "inter": pad_data(inter, pad_len),
            "intra": pad_data(intra, pad_len),
            "global": pad_data(glob, pad_len),

            # Input Action 用 0 填充
            "actions": pad_data(actions_input, pad_len, pad_val=0),

            # Target Action 用 -100 填充 (CrossEntropyLoss 忽略索引)
            "action_targets": pad_data(actions_target, pad_len, pad_val=-100),

            "rtg": pad_data(rtg.reshape(-1, 1), pad_len),
            "timesteps": pad_data(timesteps, pad_len).long(),
            "mask": torch.cat([torch.zeros(pad_len), torch.ones(len(inter))])
        }

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum