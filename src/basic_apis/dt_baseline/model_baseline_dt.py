import torch
import torch.nn as nn
import numpy as np


class BaselineStateEncoder(nn.Module):
    """
    Baseline 专用状态编码器 (Simple MLP)

    设计理由:
    PPO-Baseline (Ray/RLLib) 使用的是全连接网络 (TorchFC) 处理扁平化输入。
    为了保持公平性，DT-Baseline 也应使用 MLP 将拼接后的扁平状态映射到 Hidden Size，
    而不是使用 Teacher DT 那种复杂的 Attention 结构。
    """

    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        # 3层 MLP: Input -> 128 -> 256 -> Embed_Dim
        # 这种深度足以提取特征，同时不过于复杂
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, states):
        # states: [Batch, Seq, Input_Dim]
        return self.net(states)


class DecisionTransformerBaseline(nn.Module):
    """
    Baseline Decision Transformer (Hybrid Action Space)

    架构特点:
    1. 混合动作空间支持: 同时处理连续动作 (Player 0) 和离散动作 (Player 1-5).
    2. 主干与 Teacher DT 保持一致 (3 Layers, 4 Heads).
    """

    def __init__(self,
                 state_encoder,
                 state_dim,  # 拼接后的状态总维度
                 act_dim_cont=5,  # Player 0: 5个切片的权重
                 act_num_discrete=5,  # Player 1-5: 5个Agent
                 act_discrete_card=3,  # 每个离散Agent有3个选项 (0,1,2)
                 hidden_size=256,
                 max_length=20,
                 max_ep_len=1000):
        super().__init__()

        self.state_encoder = state_encoder
        self.hidden_size = hidden_size
        self.max_length = max_length

        # === 1. Embeddings ===

        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # 回报嵌入 (RTG)
        self.embed_rtg = nn.Linear(1, hidden_size)

        # 动作嵌入 (Hybrid)
        # A. 连续部分 (Player 0) -> Linear Projection
        self.embed_action_cont = nn.Linear(act_dim_cont, hidden_size)

        # B. 离散部分 (Player 1-5) -> Embeddings
        # 我们为这 5 个离散 Agent 各自创建一个 Embedding 层
        self.embed_action_discrete_layers = nn.ModuleList([
            nn.Embedding(act_discrete_card, hidden_size)
            for _ in range(act_num_discrete)
        ])

        self.embed_ln = nn.LayerNorm(hidden_size)

        # === 2. Transformer Backbone ===
        # 保持与 Teacher DT 完全一致的配置，确保比较公平
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # === 3. Prediction Heads (Hybrid) ===

        # Head A: 预测连续动作 (Player 0) -> Tanh (PPO Box 空间通常是 -1 到 1)
        self.predict_cont = nn.Sequential(
            nn.Linear(hidden_size, act_dim_cont),
            nn.Tanh()
        )

        # Head B: 预测离散动作 Logits (Player 1-5)
        # 输出维度 = 5个Agent * 3个选项 = 15
        self.total_discrete_logits = act_num_discrete * act_discrete_card
        self.predict_disc = nn.Linear(hidden_size, self.total_discrete_logits)

        self.act_num_discrete = act_num_discrete
        self.act_discrete_card = act_discrete_card

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Args:
            states: [Batch, K, State_Dim] (Flat Tensor)
            actions: [Batch, K, 10] (前5位是连续值，后5位是离散索引)
            returns: [Batch, K, 1]
            timesteps: [Batch, K]
        """
        B, K, _ = returns.shape

        # --- 1. Embedding Stage ---

        # A. State Embedding
        state_embeddings = self.state_encoder(states)

        # B. RTG & Time Embedding
        rtg_embeddings = self.embed_rtg(returns)
        time_embeddings = self.embed_timestep(timesteps)

        # C. Action Embedding (Hybrid Fusion)
        # 动作输入维度是 [B, K, 10]
        # 前 5 维: 连续动作
        # 后 5 维: 离散动作索引 (float 形式，需转 long)
        act_cont = actions[:, :, :5]
        act_disc = actions[:, :, 5:].long()

        # C1. 嵌入连续部分
        action_embeddings = self.embed_action_cont(act_cont)

        # C2. 嵌入离散部分并累加 (Sum)
        for i, emb_layer in enumerate(self.embed_action_discrete_layers):
            # 取出第 i 个离散动作的索引
            disc_idx = act_disc[:, :, i]
            # 查表并加到总 embedding 上
            action_embeddings = action_embeddings + emb_layer(disc_idx)

        # --- 2. Add Time Embeddings ---
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        # --- 3. Stack Inputs (R, s, a) ---
        # Shape: [B, 3*K, Hidden]
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * K, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # --- 4. Causal Mask ---
        mask = torch.triu(torch.ones(3 * K, 3 * K), diagonal=1).to(stacked_inputs.device)
        mask = mask.bool()

        # --- 5. Transformer Pass ---
        x = self.transformer(stacked_inputs, mask=mask)

        # --- 6. Extract Outputs ---
        # 我们需要根据 State 预测 Action
        # Input Layout: R, s, a
        # 我们取 's' 位置的输出 (Index 1, 4, 7...)
        x = x.reshape(B, K, 3, self.hidden_size)
        s_outputs = x[:, :, 1, :]

        # --- 7. Prediction ---

        # 预测连续动作 [-1, 1]
        preds_cont = self.predict_cont(s_outputs)  # [B, K, 5]

        # 预测离散 Logits
        # 输出 [B, K, 15] -> Reshape 为 [B, K, 5, 3] 以便计算 CrossEntropy
        preds_disc_logits = self.predict_disc(s_outputs)
        preds_disc_logits = preds_disc_logits.view(B, K, self.act_num_discrete, self.act_discrete_card)

        return preds_cont, preds_disc_logits