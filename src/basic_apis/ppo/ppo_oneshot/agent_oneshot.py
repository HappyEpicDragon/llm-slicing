import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """简化版特征提取器"""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, net_config: dict = None):
        super().__init__(observation_space, features_dim)

        # === 1. 配置处理 ===
        default_config = {
            "global_encoder": [64, 32],
            "csi_encoder": [128, 64],
            "buffer_encoder": [64, 32],
            "slice_alloc_encoder": [64, 32],
            "user_alloc_encoder": [128, 64],
            "intent_drift_encoder": [128, 64],  # 修正：75维 -> 64维（vs 之前375维）
            "sla_encoder": [64, 32],
            "topology_encoder": [128, 64]
        }

        self.cfg = net_config if net_config is not None else default_config
        for k, v in default_config.items():
            if k not in self.cfg:
                self.cfg[k] = v

        # === 2. 获取环境维度 ===
        self.max_number_users = observation_space.spaces['user_buffer_status'].shape[0]
        self.max_number_slices = observation_space.spaces['slice_priority'].shape[0]

        # Intent Drift维度
        intent_drift_shape = observation_space.spaces['intent_drift'].shape
        self.intent_drift_dim = intent_drift_shape[0] * intent_drift_shape[1] * intent_drift_shape[2]  # 5×5×3=75

        print(f"CustomFeatureExtractor (Simplified) Initialized:")
        print(f"  - Max Users (total): {self.max_number_users}")
        print(f"  - Max Slices: {self.max_number_slices}")
        print(f"  - CSI Input: 75 dims (3×{self.max_number_users} statistics)")
        print(f"  - Intent Drift: {self.intent_drift_dim} dims ({intent_drift_shape})")

        # === 3. 辅助函数 ===
        def build_encoder(input_dim, config_key):
            dims = self.cfg[config_key]
            layers = []
            in_dim = input_dim
            for out_dim in dims:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            return nn.Sequential(*layers), dims[-1]

        # === 4. 构建所有子编码器 ===
        self.global_encoder, global_out = build_encoder(2, "global_encoder")
        self.csi_encoder, csi_out = build_encoder(3 * self.max_number_users, "csi_encoder")
        self.buffer_encoder, buffer_out = build_encoder(self.max_number_users, "buffer_encoder")
        self.slice_alloc_encoder, slice_out = build_encoder(2 * self.max_number_slices, "slice_alloc_encoder")
        self.user_alloc_encoder, user_out = build_encoder(2 * self.max_number_users, "user_alloc_encoder")

        # Intent Drift Encoder（修正：75维输入）
        self.intent_drift_encoder, intent_out = build_encoder(self.intent_drift_dim, "intent_drift_encoder")

        self.sla_encoder, sla_out = build_encoder(2 * self.max_number_slices * 3, "sla_encoder")
        topology_dim = (self.max_number_slices * (self.max_number_users + 2) + self.max_number_users)
        self.topology_encoder, topo_out = build_encoder(topology_dim, "topology_encoder")

        # === 5. Fusion Layer ===
        total_feature_dim = (global_out + csi_out + buffer_out + slice_out +
                             user_out + intent_out + sla_out + topo_out)

        print(f"  - Total Feature Dim before Fusion: {total_feature_dim}")

        self.fusion_layer = nn.Sequential(
            nn.Linear(total_feature_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

        print(f"  - Output Feature Dim: {features_dim}")

    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = observations['timestep'].shape[0]

        # === 1-5. 保持不变 ===
        global_feat = self.global_encoder(torch.cat([
            observations['timestep'],
            observations['step_in_episode'],
        ], dim=-1))

        csi_combined = torch.cat([
            observations['csi_mean_per_user'],
            observations['csi_max_per_user'],
            observations['csi_std_per_user'],
        ], dim=-1)
        csi_feat = self.csi_encoder(csi_combined)

        buffer_feat = self.buffer_encoder(observations['user_buffer_status'])

        slice_alloc_info = torch.cat([
            observations['slice_allocated_power'],
            observations['slice_allocated_rbs']
        ], dim=1).reshape(batch_size, -1)
        slice_alloc_feat = self.slice_alloc_encoder(slice_alloc_info)

        user_alloc_info = torch.cat([
            observations['user_allocated_rbs'],
            observations['user_allocated_power']
        ], dim=1).reshape(batch_size, -1)
        user_alloc_feat = self.user_alloc_encoder(user_alloc_info)

        # === 6. Intent Drift（flatten到75维）===
        intent_drift = observations['intent_drift'].reshape(batch_size, -1)  # (batch, 75)
        intent_drift_feat = self.intent_drift_encoder(intent_drift)

        # === 7-8. 保持不变 ===
        sla_info = torch.cat([
            observations['sla_target'].reshape(batch_size, -1),
            observations['active_sla'].reshape(batch_size, -1)
        ], dim=-1)
        sla_feat = self.sla_encoder(sla_info)

        topology_info = torch.cat([
            observations['slice_priority'],
            observations['user_to_slice'].reshape(batch_size, -1).float(),
            observations['active_users'].reshape(batch_size, -1).float(),
            observations['active_slices'].reshape(batch_size, -1).float(),
        ], dim=-1)
        topology_feat = self.topology_encoder(topology_info)

        # === 9. Fusion ===
        combined_feat = torch.cat([
            global_feat,
            csi_feat,
            buffer_feat,
            slice_alloc_feat,
            user_alloc_feat,
            intent_drift_feat,
            sla_feat,
            topology_feat,
        ], dim=-1)

        return self.fusion_layer(combined_feat)