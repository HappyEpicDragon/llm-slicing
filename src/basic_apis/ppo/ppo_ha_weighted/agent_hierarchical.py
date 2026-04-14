import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


# === 1. 特征提取器 (HierarchicalAttentionExtractor) - 保持不变 ===
class HierarchicalAttentionExtractor(BaseFeaturesExtractor):
    """
    H+A 核心特征提取器：
    (逻辑与原代码相同，负责将 Dict 观察空间编码为单一特征向量)
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        inter_shape = observation_space['inter_feat'].shape
        self.num_slices = inter_shape[0]  # 5
        intra_shape = observation_space['intra_feat'].shape
        self.num_users = intra_shape[0]  # 25

        inter_in_dim = inter_shape[-1]
        intra_in_dim = intra_shape[-1]
        global_in_dim = observation_space['global_feat'].shape[0]

        hidden_dim = 128
        self.hidden_dim = hidden_dim

        # === 独立编码器 (Encoders) ===
        self.inter_encoder = nn.Sequential(
            nn.Linear(inter_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.intra_encoder = nn.Sequential(
            nn.Linear(intra_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # === Attention 交互层 ===
        self.attn_inter_query = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )

        self.attn_intra_query = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # === 融合层 (Fusion) ===
        flatten_dim = (self.num_slices * hidden_dim) + (self.num_users * hidden_dim) + global_in_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict, return_attn=False) -> torch.Tensor:
        inter_in = observations['inter_feat']
        intra_in = observations['intra_feat']
        global_in = observations['global_feat']

        batch_size = inter_in.shape[0]

        # 1. 独立编码
        inter_emb = self.inter_encoder(inter_in)
        intra_emb = self.intra_encoder(intra_in)

        # 2. Attention 交互
        inter_ctx, inter_attn_weights = self.attn_inter_query(
            query=inter_emb,
            key=intra_emb,
            value=intra_emb,
            need_weights=True,  # 确保计算权重
            average_attn_weights=True  # 如果是多头，取平均以便可视化
        )

        intra_ctx, _ = self.attn_intra_query(
            query=intra_emb,
            key=inter_emb,
            value=inter_emb
        )

        # 3. 融合
        inter_flat = inter_ctx.reshape(batch_size, -1)
        intra_flat = intra_ctx.reshape(batch_size, -1)

        combined = torch.cat([inter_flat, intra_flat, global_in], dim=1)

        # return self.fusion_layer(combined)
        # [修改] 根据 flag 返回不同内容
        features = self.fusion_layer(combined)

        if return_attn:
            return features, inter_attn_weights
        else:
            return features


# === 1b. MLP Feature Extractor (replaces Attention to avoid collapse) ===
class HierarchicalMLPExtractor(BaseFeaturesExtractor):
    """
    Flatten-and-MLP extractor that avoids the representation collapse
    observed in HierarchicalAttentionExtractor.

    Architecture:
      inter_feat (5,8) → flatten → 40
      intra_feat (25,7) → per-slice pooling → 5*7=35 (compress 25 users to 5 slices)
      global_feat (2,) → 2
      Total input: 40 + 35 + 2 = 77
      → MLP: 77 → 256 → 256 → features_dim

    The per-slice pooling reduces intra dimensionality while preserving
    slice-level structure, avoiding the 3842-dim flatten bottleneck.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        inter_shape = observation_space['inter_feat'].shape
        intra_shape = observation_space['intra_feat'].shape
        global_dim = observation_space['global_feat'].shape[0]

        self.num_slices = inter_shape[0]   # 5
        self.num_users = intra_shape[0]    # 25
        self.users_per_slice = self.num_users // self.num_slices  # 5
        self.intra_dim = intra_shape[-1]   # 7

        inter_flat_dim = int(np.prod(inter_shape))  # 40
        intra_pooled_dim = self.num_slices * intra_shape[-1]  # 35
        total_in = inter_flat_dim + intra_pooled_dim + global_dim  # 77

        self.mlp = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict, return_attn: bool = False):
        inter_in = observations['inter_feat']     # (B, 5, 8)
        intra_in = observations['intra_feat']     # (B, 25, 7)
        global_in = observations['global_feat']   # (B, 2)

        batch_size = inter_in.shape[0]

        inter_flat = inter_in.reshape(batch_size, -1)

        # Per-slice mean-pooling of intra features
        intra_reshaped = intra_in.reshape(
            batch_size, self.num_slices, self.users_per_slice, self.intra_dim
        )
        intra_pooled = intra_reshaped.mean(dim=2)  # (B, 5, 7)
        intra_flat = intra_pooled.reshape(batch_size, -1)  # (B, 35)

        combined = torch.cat([inter_flat, intra_flat, global_in], dim=1)
        return self.mlp(combined)


class HierarchicalPooledAttentionExtractor(BaseFeaturesExtractor):
    """
    Slice-token self-attention extractor.

    Compared with the original cross-attention variant, this first mean-pools
    intra-slice user features into 5 slice-level summaries, then runs
    self-attention over 5 slice tokens plus one global token.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        inter_shape = observation_space['inter_feat'].shape
        intra_shape = observation_space['intra_feat'].shape
        global_dim = observation_space['global_feat'].shape[0]

        self.num_slices = inter_shape[0]
        self.num_users = intra_shape[0]
        self.users_per_slice = self.num_users // self.num_slices
        self.intra_dim = intra_shape[-1]

        hidden_dim = 128
        self.hidden_dim = hidden_dim

        self.inter_encoder = nn.Sequential(
            nn.Linear(inter_shape[-1], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.intra_encoder = nn.Sequential(
            nn.Linear(self.intra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.slice_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        token_count = self.num_slices + 1
        self.fusion_layer = nn.Sequential(
            nn.Linear(token_count * hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict, return_attn=False) -> torch.Tensor:
        inter_in = observations['inter_feat']
        intra_in = observations['intra_feat']
        global_in = observations['global_feat']

        batch_size = inter_in.shape[0]

        inter_tokens = self.inter_encoder(inter_in)

        intra_reshaped = intra_in.reshape(
            batch_size, self.num_slices, self.users_per_slice, self.intra_dim
        )
        intra_pooled = intra_reshaped.mean(dim=2)
        intra_tokens = self.intra_encoder(intra_pooled)

        slice_tokens = self.slice_fusion(torch.cat([inter_tokens, intra_tokens], dim=-1))
        global_token = self.global_encoder(global_in).unsqueeze(1)
        tokens = torch.cat([slice_tokens, global_token], dim=1)

        attn_out, attn_weights = self.self_attn(
            tokens,
            tokens,
            tokens,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        features = self.fusion_layer(tokens.reshape(batch_size, -1))
        if return_attn:
            return features, attn_weights
        return features


class HierarchicalSliceAttnExtractor(BaseFeaturesExtractor):
    """
    Lightweight slice-token self-attention extractor (方案α).

    Same 77-dim MLP-pool preprocessing as HierarchicalMLPExtractor,
    but reshapes into 5 slice tokens + 1 global token and applies
    a single transformer layer to capture inter-slice interactions.

    Architecture:
      inter_feat (5,8) → 5 slice tokens (8-dim each)
      intra_feat (25,7) → per-slice mean-pool → 5 tokens (7-dim each)
      Concat per slice: 5 × (8+7) = 5 × 15
      → Linear(15 → hidden) → 5 slice tokens
      + global(2) → Linear(2 → hidden) → 1 global token
      → Self-Attention over 6 tokens
      → Flatten(6 × hidden) → MLP → features_dim
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256,
                 hidden_dim: int = 64, num_heads: int = 4):
        super().__init__(observation_space, features_dim)

        inter_shape = observation_space['inter_feat'].shape
        intra_shape = observation_space['intra_feat'].shape
        global_dim = observation_space['global_feat'].shape[0]

        self.num_slices = inter_shape[0]       # 5
        self.num_users = intra_shape[0]        # 25
        self.users_per_slice = self.num_users // self.num_slices  # 5
        self.inter_dim = inter_shape[-1]       # 8
        self.intra_dim = intra_shape[-1]       # 7
        self.hidden_dim = hidden_dim

        slice_in_dim = self.inter_dim + self.intra_dim  # 15
        self.slice_proj = nn.Sequential(
            nn.Linear(slice_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        token_count = self.num_slices + 1  # 6
        self.output_mlp = nn.Sequential(
            nn.Linear(token_count * hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict, return_attn: bool = False):
        inter_in = observations['inter_feat']     # (B, 5, 8)
        intra_in = observations['intra_feat']     # (B, 25, 7)
        global_in = observations['global_feat']   # (B, 2)

        batch_size = inter_in.shape[0]

        intra_reshaped = intra_in.reshape(
            batch_size, self.num_slices, self.users_per_slice, self.intra_dim
        )
        intra_pooled = intra_reshaped.mean(dim=2)  # (B, 5, 7)

        slice_in = torch.cat([inter_in, intra_pooled], dim=-1)  # (B, 5, 15)
        slice_tokens = self.slice_proj(slice_in)                 # (B, 5, hidden)
        global_token = self.global_proj(global_in).unsqueeze(1)  # (B, 1, hidden)

        tokens = torch.cat([slice_tokens, global_token], dim=1)  # (B, 6, hidden)

        attn_out, attn_weights = self.self_attn(
            tokens, tokens, tokens,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        features = self.output_mlp(tokens.reshape(batch_size, -1))
        if return_attn:
            return features, attn_weights
        return features


class HierarchicalSliceAttnPolicy(ActorCriticPolicy):
    """Policy using lightweight slice-token self-attention extractor."""

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            *args,
            **kwargs,
    ):
        kwargs["features_extractor_class"] = HierarchicalSliceAttnExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}

        if net_arch is None:
            net_arch = dict(pi=[256, 128], vf=[256, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )


# === 2. 策略网络 (HierarchicalSmartPolicy) - 简化版 ===
class HierarchicalSmartPolicy(ActorCriticPolicy):
    """
    H+A 专用策略类。

    由于环境现在的 Action Space 是 MultiDiscrete，SB3 的 ActorCriticPolicy 会自动
    使用 MultiCategoricalDistribution 来处理它。

    我们只需要在初始化时指定我们的特征提取器 (HierarchicalAttentionExtractor) 即可。
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            *args,
            **kwargs,
    ):
        # 强制指定 Feature Extractor
        kwargs["features_extractor_class"] = HierarchicalAttentionExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}

        # 默认网络架构
        if net_arch is None:
            net_arch = dict(pi=[256, 128], vf=[256, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )


class HierarchicalMLPPolicy(ActorCriticPolicy):
    """
    MLP-based policy using HierarchicalMLPExtractor instead of Attention.
    Drop-in replacement for HierarchicalSmartPolicy.
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            *args,
            **kwargs,
    ):
        kwargs["features_extractor_class"] = HierarchicalMLPExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}

        if net_arch is None:
            net_arch = dict(pi=[256, 128], vf=[256, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )


class HierarchicalPooledAttentionPolicy(ActorCriticPolicy):
    """
    Attention policy using pooled slice tokens before self-attention.
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            *args,
            **kwargs,
    ):
        kwargs["features_extractor_class"] = HierarchicalPooledAttentionExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}

        if net_arch is None:
            net_arch = dict(pi=[256, 128], vf=[256, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )