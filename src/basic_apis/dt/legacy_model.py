import torch
import torch.nn as nn
import math


class HierarchicalStateEncoder(nn.Module):
    """
    负责将 [Batch, Seq, ...] 分层状态 压缩成 [Batch, Seq, Embed_Dim]
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        # Inter: 4维 -> Hidden
        self.inter_enc = nn.Sequential(
            nn.Linear(4, 128), nn.LayerNorm(128), nn.ReLU()
        )
        # Intra: 5维 -> Hidden
        self.intra_enc = nn.Sequential(
            nn.Linear(5, 128), nn.LayerNorm(128), nn.ReLU()
        )

        # Attention Blocks
        self.attn_inter = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.attn_intra = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Fusion
        # Input: (5*128) + (25*128) + 2
        flat_dim = 640 + 3200 + 2
        self.fusion = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, inter, intra, glob, return_attn=False):
        # 输入维度: [Batch, Seq_Len, N_Slice, Feat]
        # 我们需要把 (Batch, Seq) 合并处理，因为 Encoder 只关心单帧内的空间关系
        B, K, N_S, F_Inter = inter.shape
        _, _, N_U, F_Intra = intra.shape

        # Fold batch and time
        inter = inter.view(-1, N_S, F_Inter)  # [B*K, 5, 4]
        intra = intra.view(-1, N_U, F_Intra)  # [B*K, 25, 5]
        glob = glob.view(-1, 2)  # [B*K, 2]

        # Encoding
        e_inter = self.inter_enc(inter)  # [B*K, 5, 128]
        e_intra = self.intra_enc(intra)  # [B*K, 25, 128]

        # Cross Attention
        # Inter query Intra
        # ctx_inter, _ = self.attn_inter(e_inter, e_intra, e_intra)
        ctx_inter, inter_attn_weights = self.attn_inter(
            e_inter, e_intra, e_intra,
            need_weights=True,
            average_attn_weights=True
        )
        # Intra query Inter
        ctx_intra, _ = self.attn_intra(e_intra, e_inter, e_inter)

        # Flatten & Fuse
        flat_inter = ctx_inter.reshape(B * K, -1)
        flat_intra = ctx_intra.reshape(B * K, -1)

        combined = torch.cat([flat_inter, flat_intra, glob], dim=1)
        embedding = self.fusion(combined)  # [B*K, 256]

        # Unfold
        # return embedding.view(B, K, -1)
        out = embedding.view(B, K, -1)

        if return_attn:
            # 同样需要 Unfold 权重: [B*K, 5, 25] -> [B, K, 5, 25]
            B_K, N_S, N_U = inter_attn_weights.shape
            attn_out = inter_attn_weights.view(B, K, N_S, N_U)
            return out, attn_out
        else:
            return out


class DecisionTransformer(nn.Module):

    def __init__(self, state_encoder,
                 action_dims,  # [新增] list, e.g., [10, 3, 3, 3, 3, 3]
                 hidden_size=256, max_length=20, max_ep_len=1000,
                 n_layer=3, n_head=4, dropout=0.1, activation="relu"):
        super().__init__()
        self.state_encoder = state_encoder
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.n_head = n_head
        self.action_dims = action_dims

        # Embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)

        # [关键修改] Multi-Discrete Action Embedding
        # 创建多个 Embedding 层，每个对应一个动作维度
        self.embed_action_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=hidden_size)
            for dim in action_dims
        ])

        self.embed_ln = nn.LayerNorm(hidden_size)

        # GPT Backbone (Pre-Norm recommended)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # [关键修改] Prediction Head
        # 输出 Logits 的总数 (所有维度基数之和)
        # 例如: 10 + 3*5 = 25
        total_logits = sum(action_dims)
        self.predict_action = nn.Linear(hidden_size, total_logits)
        # 注意: 只有 Linear，没有 Tanh/Softmax

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        # actions: [B, K, 6] (Long)
        B, K, _ = returns.shape

        # 1. Embeddings
        state_embeddings = self.state_encoder(states['inter'], states['intra'], states['global'])
        rtg_embeddings = self.embed_rtg(returns)
        time_embeddings = self.embed_timestep(timesteps)

        # [关键修改] Action Embedding 融合
        # 对每个维度分别 Embedding，然后相加 (Sum)
        action_embeddings = torch.zeros((B, K, self.hidden_size), device=actions.device)
        for i, emb_layer in enumerate(self.embed_action_layers):
            # 取出第 i 个动作维度的索引 [B, K]
            dim_act_idx = actions[:, :, i]
            # 查表并累加
            action_embeddings = action_embeddings + emb_layer(dim_act_idx)

        # 2. Add Time Embedding
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        # 3. Stack inputs: (R, s, a) -> [B, 3*K, Hidden]
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * K, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # 4. Build combined 3D attention mask (causal + padding).
        # Using a 3D mask avoids the PyTorch < 2.0 bug where
        # src_key_padding_mask + causal mask can produce all-masked rows
        # leading to softmax(all -inf) = NaN.
        seq_len = 3 * K
        causal = torch.triu(torch.ones(seq_len, seq_len, device=stacked_inputs.device), diagonal=1).bool()

        if attention_mask is not None:
            pad_key = ~(attention_mask.to(stacked_inputs.device) > 0)  # [B, K] True=padded
            pad_key = pad_key.repeat_interleave(3, dim=1)              # [B, 3K]
            pad_key_2d = pad_key.unsqueeze(1).expand(B, seq_len, seq_len)  # [B, 3K, 3K]
            combined = causal.unsqueeze(0) | pad_key_2d                    # [B, 3K, 3K]
            diag_idx = torch.arange(seq_len, device=stacked_inputs.device)
            combined[:, diag_idx, diag_idx] = False  # self-attention always allowed
        else:
            combined = causal.unsqueeze(0).expand(B, seq_len, seq_len)

        attn_mask = torch.zeros_like(combined, dtype=stacked_inputs.dtype)
        attn_mask.masked_fill_(combined, float('-inf'))
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_head, seq_len, seq_len)
        attn_mask = attn_mask.reshape(B * self.n_head, seq_len, seq_len)

        # 5. Transformer Pass (no src_key_padding_mask needed)
        x = self.transformer(stacked_inputs, mask=attn_mask)

        # 7. Extract outputs (predict action given state)
        # 取出 State 对应的输出位置 (index 1, 4, 7...)
        x = x.reshape(B, K, 3, self.hidden_size)
        s_outputs = x[:, :, 1, :]

        # 7. Logits Output [B, K, Total_Logits]
        action_preds = self.predict_action(s_outputs)

        return action_preds

    def get_context_embedding(self, states, actions, returns, timesteps, attention_mask=None):
        """
        [XAI 专用] 提取 Transformer 的输出特征 (Brain Scan)
        而不是最后的 Action Logits。
        """
        B, K, _ = returns.shape

        # 1. Embeddings (与 forward 保持一致)
        state_embeddings = self.state_encoder(states['inter'], states['intra'], states['global'])
        rtg_embeddings = self.embed_rtg(returns)
        time_embeddings = self.embed_timestep(timesteps)

        action_embeddings = torch.zeros((B, K, self.hidden_size), device=actions.device)
        for i, emb_layer in enumerate(self.embed_action_layers):
            dim_act_idx = actions[:, :, i]
            action_embeddings = action_embeddings + emb_layer(dim_act_idx)

        # 2. Add Time Embedding
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        # 3. Stack inputs
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * K, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # 4. Build combined 3D mask (same logic as forward)
        seq_len = 3 * K
        causal = torch.triu(torch.ones(seq_len, seq_len, device=stacked_inputs.device), diagonal=1).bool()
        if attention_mask is not None:
            pad_key = ~(attention_mask.to(stacked_inputs.device) > 0)
            pad_key = pad_key.repeat_interleave(3, dim=1)
            pad_key_2d = pad_key.unsqueeze(1).expand(B, seq_len, seq_len)
            combined = causal.unsqueeze(0) | pad_key_2d
            diag_idx = torch.arange(seq_len, device=stacked_inputs.device)
            combined[:, diag_idx, diag_idx] = False
        else:
            combined = causal.unsqueeze(0).expand(B, seq_len, seq_len)
        attn_mask = torch.zeros_like(combined, dtype=stacked_inputs.dtype)
        attn_mask.masked_fill_(combined, float('-inf'))
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_head, seq_len, seq_len)
        attn_mask = attn_mask.reshape(B * self.n_head, seq_len, seq_len)

        # 5. Transformer Pass
        x = self.transformer(stacked_inputs, mask=attn_mask)

        # 6. Extract ONLY State representations
        # reshape back to [B, K, 3, Hidden]
        x = x.reshape(B, K, 3, self.hidden_size)

        # 取出中间那个 (State 对应的输出)，这就是"决策上下文"
        s_outputs = x[:, :, 1, :]

        # 取最后一个时间步 (Current Step Reasoning)
        # Shape: [B, Hidden] -> [Hidden] (squeeze later)
        last_step_embedding = s_outputs[:, -1, :]

        return last_step_embedding


class DecisionMLP(nn.Module):
    """
    MLP ablation model with the same I/O interface as DecisionTransformer.
    It removes temporal self-attention and predicts actions from per-step
    concatenated embeddings: [RTG, State, PrevAction].
    """

    def __init__(self, state_encoder, action_dims, hidden_size=256, max_length=20, max_ep_len=1000):
        super().__init__()
        self.state_encoder = state_encoder
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.action_dims = action_dims

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=hidden_size)
            for dim in action_dims
        ])
        self.embed_ln = nn.LayerNorm(hidden_size)

        in_dim = hidden_size * 3
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        self.predict_action = nn.Linear(hidden_size, sum(action_dims))

    def _build_embeddings(self, states, actions, returns, timesteps):
        B, K, _ = returns.shape
        state_embeddings = self.state_encoder(states['inter'], states['intra'], states['global'])
        rtg_embeddings = self.embed_rtg(returns)
        time_embeddings = self.embed_timestep(timesteps)

        action_embeddings = torch.zeros((B, K, self.hidden_size), device=actions.device)
        for i, emb_layer in enumerate(self.embed_action_layers):
            action_embeddings = action_embeddings + emb_layer(actions[:, :, i])

        state_embeddings = self.embed_ln(state_embeddings + time_embeddings)
        action_embeddings = self.embed_ln(action_embeddings + time_embeddings)
        rtg_embeddings = self.embed_ln(rtg_embeddings + time_embeddings)
        return state_embeddings, action_embeddings, rtg_embeddings

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        state_embeddings, action_embeddings, rtg_embeddings = self._build_embeddings(
            states, actions, returns, timesteps
        )
        x = torch.cat([rtg_embeddings, state_embeddings, action_embeddings], dim=-1)  # [B, K, 3H]
        h = self.backbone(x)  # [B, K, H]
        return self.predict_action(h)  # [B, K, logits]

    def get_context_embedding(self, states, actions, returns, timesteps, attention_mask=None):
        state_embeddings, action_embeddings, rtg_embeddings = self._build_embeddings(
            states, actions, returns, timesteps
        )
        x = torch.cat([rtg_embeddings, state_embeddings, action_embeddings], dim=-1)
        h = self.backbone(x)
        return h[:, -1, :]


def build_decision_model(model_cfg, state_encoder, action_dims):
    """
    Factory: build DT backbone model from config.
    Supported backbones:
      - transformer (default)
      - mlp
    """
    backbone = str(getattr(model_cfg, "backbone", "transformer")).lower()
    common_kwargs = dict(
        state_encoder=state_encoder,
        action_dims=action_dims,
        hidden_size=model_cfg.embed_dim,
        max_length=model_cfg.context_len,
    )
    if backbone == "mlp":
        return DecisionMLP(**common_kwargs)
    return DecisionTransformer(
        **common_kwargs,
        n_layer=int(getattr(model_cfg, "n_layer", 3)),
        n_head=int(getattr(model_cfg, "n_head", 4)),
        dropout=float(getattr(model_cfg, "dropout", 0.1)),
        activation=str(getattr(model_cfg, "activation", "relu")),
    )