import torch
import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class BaselineDTDataset(Dataset):
    """
    Baseline 专用数据集
    逻辑与 Teacher 的 HierarchicalDTDataset 严格对齐：
    1. 预加载所有数据到内存 (Array 格式)
    2. 统一做 NaN 清洗和 RTG SymLog 变换
    3. 在 __getitem__ 中处理 Context Slicing 和 Action Shifting
    """

    def __init__(self, dataset_path, context_len=20, rtg_scale=1.0, metadata_path="metadata.json"):
        self.context_len = context_len
        self.rtg_scale = rtg_scale

        # === 1. 加载标准化元数据 ===
        self.normalize = False
        # 尝试自动寻找 metadata.json
        if not os.path.exists(metadata_path):
            potential_path = os.path.join(os.path.dirname(dataset_path), "metadata.json")
            if os.path.exists(potential_path):
                metadata_path = potential_path

        if os.path.exists(metadata_path):
            print(f"📊 [Baseline] Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.meta = json.load(f)

            # Baseline 使用 'flat' 键
            if "obs_stats" in self.meta and "flat" in self.meta["obs_stats"]:
                stats = self.meta["obs_stats"]["flat"]
                self.obs_mean = np.array(stats['mean'], dtype=np.float32)
                self.obs_std = np.array(stats['std'], dtype=np.float32) + 1e-6
                self.normalize = True
            else:
                print("⚠️ Metadata format mismatch (expected 'obs_stats.flat'). Skipping norm.")
        else:
            print(f"⚠️ Metadata not found. Skipping normalization.")

        # === 2. 加载轨迹数据 ===
        self.trajectories = []
        self.indices = []

        if os.path.isdir(dataset_path):
            print(f"📂 Scanning directory: {dataset_path} ...")
            files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
            files.sort()

            for f_name in tqdm(files, desc="Loading Baseline Trajectories"):
                full_path = os.path.join(dataset_path, f_name)
                try:
                    with open(full_path, 'rb') as f:
                        traj = pickle.load(f)
                        self._process_and_append(traj)
                except Exception as e:
                    print(f"❌ Error loading {f_name}: {e}")
        else:
            # 单文件模式
            with open(dataset_path, 'rb') as f:
                traj = pickle.load(f)
                self._process_and_append(traj)

        print(f"✅ Baseline Dataset Loaded. Total Trajectories: {len(self.trajectories)}")

    def _process_and_append(self, traj):
        """
        处理单个轨迹：清洗、变换
        """
        # 1. 基础数值清洗
        # Rewards
        traj['rewards'] = np.nan_to_num(traj['rewards'], nan=0.0, posinf=0.0, neginf=-100.0)

        # Observations: Baseline 收集时已经是 [T, Dim] 的 Numpy Array
        # 但为了保险，再次做一下 NaN 清洗
        traj['observations'] = np.nan_to_num(traj['observations'], nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        # Actions: Baseline 是 [T, 10] (5 Cont + 5 Disc_Idx)
        # 保持 float32，因为 Cont 是 float，Disc 索引暂存为 float
        traj['actions'] = traj['actions'].astype(np.float32)

        # 2. RTG SymLog (完全对齐 Teacher)
        raw_rtg = self._discount_cumsum(traj['rewards'], gamma=1.0)
        traj['rtg_transformed'] = np.sign(raw_rtg) * np.log1p(np.abs(raw_rtg))

        # 3. 存储长度并建立索引
        traj_len = len(traj['actions'])
        traj['length'] = traj_len

        self.trajectories.append(traj)

        for step in range(traj_len):
            self.indices.append((len(self.trajectories) - 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, end_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # 计算窗口
        start_idx = end_idx - self.context_len + 1
        if start_idx < 0:
            window_start = 0
            pad_len = abs(start_idx)
        else:
            window_start = start_idx
            pad_len = 0
        window_end = end_idx + 1

        # === 1. Obs 处理 (标准化) ===
        raw_obs = traj['observations'][window_start:window_end]
        if self.normalize:
            # (x - mean) / std
            # Clip 到 [-5, 5] 与 Teacher 保持一致
            obs = (raw_obs - self.obs_mean) / self.obs_std
            obs = np.clip(obs, -5.0, 5.0)
        else:
            obs = raw_obs

        # === 2. Action 处理 (Shift Logic) ===
        # Target: 原始动作序列 a_t
        actions_target = traj['actions'][window_start:window_end]

        # Input: 右移序列 a_{t-1}
        # Baseline Action Dim = 10
        act_dim = actions_target.shape[-1]

        if window_start > 0:
            # 取前一步动作
            prev_act = traj['actions'][window_start - 1].reshape(1, -1)
            actions_input = np.concatenate([prev_act, actions_target[:-1]], axis=0)
        else:
            # t=0 时，dummy action 为全 0
            dummy = np.zeros((1, act_dim), dtype=np.float32)
            if len(actions_target) > 0:
                actions_input = np.concatenate([dummy, actions_target[:-1]], axis=0)
            else:
                actions_input = dummy

        # === 3. RTG & Timesteps ===
        rtg = traj['rtg_transformed'][window_start:window_end]
        timesteps = np.arange(window_start, window_end)

        # === 4. Padding ===
        def pad_data(data, pad_len, pad_val=0):
            t_data = torch.from_numpy(data).float()  # 默认全转 float
            if pad_len == 0: return t_data

            shape = t_data.shape
            pad = torch.full((pad_len, *shape[1:]), pad_val, dtype=torch.float)
            return torch.cat([pad, t_data], dim=0)

        # 返回字典
        return {
            'obs': pad_data(obs, pad_len),

            # Input Action (Pad 0)
            'actions': pad_data(actions_input, pad_len, pad_val=0),

            # Target Action (Pad -100, 忽略 Loss)
            # 虽然是 Float Tensor，但在 Loss 计算时，Discrete 部分会被转为 Long
            'action_targets': pad_data(actions_target, pad_len, pad_val=-100),

            'rtg': pad_data(rtg.reshape(-1, 1), pad_len),
            'timesteps': pad_data(timesteps, pad_len).long(),  # Time 必须是 Long

            # Mask: 1 for valid, 0 for pad
            'mask': torch.cat([torch.zeros(pad_len), torch.ones(len(obs))])
        }

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum