"""
Cost value function（2×128 MLP）用于 Lagrangian PPO。
"""
import torch
import torch.nn as nn


class CostNetwork(nn.Module):
    """
    预测从当前状态出发的累计约束违约代价（cost-to-go）。
    输入维度：obs_dim（flat 观测向量）
    输出维度：1（标量代价估计）
    """

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
