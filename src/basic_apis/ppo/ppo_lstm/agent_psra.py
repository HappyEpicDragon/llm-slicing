# agent_psra.py
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
import numpy as np


# agent_psra.py
class CustomFeatureExtractor(BaseFeaturesExtractor):
    """自定义特征提取器"""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # ✓ 从观测空间自动获取实际维度
        self.max_number_users = observation_space.spaces['csi_current_rb'].shape[0]
        self.max_number_slices = observation_space.spaces['slice_priority'].shape[0]

        # ✓ 从 intent_drift 的观测空间获取实际的用户数
        intent_drift_shape = observation_space.spaces['intent_drift'].shape
        # intent_drift_shape = (n_slices, n_actual_users, 3)
        self.n_slices_intent = intent_drift_shape[0]
        self.n_users_intent = intent_drift_shape[1]  # 这里是5，不是25
        self.n_metrics = intent_drift_shape[2]

        print(f"CustomFeatureExtractor initialized:")
        print(f"  max_number_users (from csi): {self.max_number_users}")
        print(f"  max_number_slices: {self.max_number_slices}")
        print(f"  intent_drift shape: ({self.n_slices_intent}, {self.n_users_intent}, {self.n_metrics})")

        # ==================== 动态特征编码器 ====================

        # 全局进度信息编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # CSI编码器
        self.csi_encoder = nn.Sequential(
            nn.Linear(self.max_number_users, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 切片分配历史编码器
        self.slice_alloc_encoder = nn.Sequential(
            nn.Linear(2 * self.max_number_slices, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 用户分配历史编码器
        self.user_alloc_encoder = nn.Sequential(
            nn.Linear(2 * self.max_number_users, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # ✓ Intent Drift编码器 - 使用实际维度
        intent_drift_dim = self.n_slices_intent * self.n_users_intent * self.n_metrics
        print(f"  intent_drift_dim: {intent_drift_dim}")

        self.intent_drift_encoder = nn.Sequential(
            nn.Linear(intent_drift_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ==================== 静态特征编码器 ====================

        # SLA配置编码器
        self.sla_encoder = nn.Sequential(
            nn.Linear(2 * self.max_number_slices * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 拓扑关系编码器
        topology_dim = (self.max_number_slices * (self.max_number_users + 2)
                        + self.max_number_users)
        self.topology_encoder = nn.Sequential(
            nn.Linear(topology_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ==================== LSTM层 ====================

        total_feature_dim = 64 + 128 + 128 + 256 + 128 + 128 + 128  # = 960

        self.lstm_hidden_size = features_dim
        self.lstm = nn.LSTM(
            input_size=total_feature_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.lstm_states = {}

    def forward(self, observations: dict) -> torch.Tensor:
        """前向传播"""
        batch_size = observations['timestep'].shape[0]

        # ==================== 编码动态特征 ====================

        # 1. 全局进度信息
        global_info = torch.cat([
            observations['timestep'],
            observations['remaining_rbs'],
            observations['current_rb_id'].float(),
            observations['remaining_power'],
        ], dim=-1)
        global_feat = self.global_encoder(global_info)

        # 2. 当前RB的CSI
        csi_feat = self.csi_encoder(observations['csi_current_rb'])

        # 3. 切片分配历史
        slice_alloc_info = torch.cat([
            observations['slice_allocated_power'],
            observations['slice_allocated_rbs'],
        ], dim=1)
        slice_alloc_info = slice_alloc_info.reshape(batch_size, -1)
        slice_alloc_feat = self.slice_alloc_encoder(slice_alloc_info)

        # 4. 用户分配历史
        user_alloc_info = torch.cat([
            observations['user_allocated_rbs'],
            observations['user_allocated_power'],
        ], dim=1)
        user_alloc_info = user_alloc_info.reshape(batch_size, -1)
        user_alloc_feat = self.user_alloc_encoder(user_alloc_info)

        # 5. Intent Drift - ✓ 直接reshape，不检查具体维度
        intent_drift = observations['intent_drift']
        intent_drift = intent_drift.reshape(batch_size, -1)  # [batch, n_slices*n_users*n_metrics]
        intent_drift_feat = self.intent_drift_encoder(intent_drift)

        # ==================== 编码静态特征 ====================

        # 6. SLA配置
        sla_info = torch.cat([
            observations['sla_target'].reshape(batch_size, -1),
            observations['active_sla'].reshape(batch_size, -1),
        ], dim=-1)
        sla_feat = self.sla_encoder(sla_info)

        # 7. 拓扑关系
        topology_info = torch.cat([
            observations['slice_priority'],
            observations['user_to_slice'].reshape(batch_size, -1),
            observations['active_users'].reshape(batch_size, -1).float(),
            observations['active_slices'].reshape(batch_size, -1).float(),
        ], dim=-1)
        topology_feat = self.topology_encoder(topology_info)

        # ==================== 特征融合 ====================

        combined_feat = torch.cat([
            global_feat,
            csi_feat,
            slice_alloc_feat,
            user_alloc_feat,
            intent_drift_feat,
            sla_feat,
            topology_feat,
        ], dim=-1)

        # ==================== LSTM处理 ====================

        combined_feat = combined_feat.unsqueeze(1)  # [batch, 1, 960]

        if batch_size not in self.lstm_states:
            # 初始化新状态
            lstm_out, new_states = self.lstm(combined_feat)
            # ✓ 保存时detach，断开计算图
            self.lstm_states[batch_size] = (
                new_states[0].detach(),
                new_states[1].detach()
            )
        else:
            h, c = self.lstm_states[batch_size]

            if h.shape[1] != batch_size:
                # 重新初始化
                lstm_out, new_states = self.lstm(combined_feat)
                self.lstm_states[batch_size] = (
                    new_states[0].detach(),
                    new_states[1].detach()
                )
            else:
                # ✓ 使用状态前detach
                h_detached = h.detach()
                c_detached = c.detach()
                lstm_out, new_states = self.lstm(combined_feat, (h_detached, c_detached))

                # ✓ 保存新状态时也detach
                self.lstm_states[batch_size] = (
                    new_states[0].detach(),
                    new_states[1].detach()
                )

        lstm_feat = lstm_out.squeeze(1)  # [batch, features_dim]

        return lstm_feat

    def reset_lstm_states(self, batch_size=None):
        """重置LSTM状态"""
        if batch_size is None:
            self.lstm_states = {}
        elif batch_size in self.lstm_states:
            del self.lstm_states[batch_size]


class MaskableLSTMPolicy(MaskableActorCriticPolicy):
    """
    支持动作掩码的LSTM策略网络
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            features_extractor_class=CustomFeatureExtractor,  # ✓ 指定特征提取器
            features_extractor_kwargs=None,  # ✓ 特征提取器参数
            *args,
            **kwargs,
    ):
        # 设置特征提取器参数
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(features_dim=512)  # LSTM hidden size

        # 网络架构
        if net_arch is None:
            net_arch = dict(
                pi=[256, 256],  # Actor网络
                vf=[256, 256]  # Critic网络
            )

        # ✓ 调用父类初始化，传递特征提取器
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,  # ✓
            features_extractor_kwargs=features_extractor_kwargs,  # ✓
            *args,
            **kwargs,
        )


class LSTMStateResetCallback(BaseCallback):
    """
    在每个episode结束时重置LSTM状态的回调
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 检查是否有环境完成了episode
        dones = self.locals.get('dones', [])
        if isinstance(dones, np.ndarray):
            for idx, done in enumerate(dones):
                if done:
                    # 重置LSTM状态
                    if hasattr(self.model.policy, 'features_extractor'):
                        extractor = self.model.policy.features_extractor
                        if hasattr(extractor, 'reset_lstm_states'):
                            extractor.reset_lstm_states()
                            if self.verbose > 0:
                                # print(f"重置了环境 {idx} 的LSTM状态")
                                pass
                    break  # 只要有一个环境完成就重置
        return True