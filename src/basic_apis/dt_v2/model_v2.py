"""
State encoder for V2 observations (inter_feat: 8 dims, intra_feat: 7 dims).

Same architecture as HierarchicalStateEncoder but with parameterized input dims.
"""
import torch
import torch.nn as nn

from src.basic_apis.dt_utils.model_ha_dt import build_decision_model as _build_decision_model


class HierarchicalStateEncoderV2(nn.Module):
    """Parameterized version — accepts arbitrary inter/intra feature dimensions."""

    def __init__(self, embed_dim=256, inter_dim=8, intra_dim=7, num_slices=5, num_users=25, global_dim=2):
        super().__init__()
        hidden = 128

        self.inter_enc = nn.Sequential(
            nn.Linear(inter_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )
        self.intra_enc = nn.Sequential(
            nn.Linear(intra_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )

        self.attn_inter = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        self.attn_intra = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)

        flat_dim = (num_slices * hidden) + (num_users * hidden) + global_dim
        self.fusion = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, inter, intra, glob, return_attn=False):
        B, K, N_S, F_Inter = inter.shape
        _, _, N_U, F_Intra = intra.shape

        inter = inter.view(-1, N_S, F_Inter)
        intra = intra.view(-1, N_U, F_Intra)
        glob = glob.view(-1, glob.shape[-1])

        e_inter = self.inter_enc(inter)
        e_intra = self.intra_enc(intra)

        ctx_inter, inter_attn_weights = self.attn_inter(
            e_inter, e_intra, e_intra,
            need_weights=True, average_attn_weights=True,
        )
        ctx_intra, _ = self.attn_intra(e_intra, e_inter, e_inter)

        flat_inter = ctx_inter.reshape(B * K, -1)
        flat_intra = ctx_intra.reshape(B * K, -1)

        combined = torch.cat([flat_inter, flat_intra, glob], dim=1)
        embedding = self.fusion(combined)
        out = embedding.view(B, K, -1)

        if return_attn:
            attn_out = inter_attn_weights.view(B, K, N_S, -1)
            return out, attn_out
        return out


class MLPStateEncoderV2(nn.Module):
    """
    MLP state encoder aligned with HierarchicalMLPExtractor (PPO teacher).

    Architecture:
      inter (B,K,5,8) → flatten → 40
      intra (B,K,25,7) → per-slice mean-pool → 35
      global (B,K,2) → 2
      Total: 77 → MLP(256,256) → embed_dim
    """

    def __init__(self, embed_dim=256, inter_dim=8, intra_dim=7,
                 num_slices=5, num_users=25, global_dim=2):
        super().__init__()
        self.num_slices = num_slices
        self.users_per_slice = num_users // num_slices
        self.intra_dim = intra_dim

        inter_flat_dim = num_slices * inter_dim      # 40
        intra_pooled_dim = num_slices * intra_dim     # 35
        total_in = inter_flat_dim + intra_pooled_dim + global_dim  # 77

        self.mlp = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, inter, intra, glob, return_attn=False):
        B, K, N_S, F_Inter = inter.shape
        _, _, N_U, F_Intra = intra.shape

        inter = inter.view(B * K, N_S, F_Inter)
        intra = intra.view(B * K, N_U, F_Intra)
        glob = glob.view(B * K, -1)

        inter_flat = inter.reshape(B * K, -1)                      # (BK, 40)

        intra_reshaped = intra.reshape(
            B * K, self.num_slices, self.users_per_slice, self.intra_dim
        )
        intra_pooled = intra_reshaped.mean(dim=2)                   # (BK, 5, 7)
        intra_flat = intra_pooled.reshape(B * K, -1)                # (BK, 35)

        combined = torch.cat([inter_flat, intra_flat, glob], dim=1)  # (BK, 77)
        out = self.mlp(combined).view(B, K, -1)

        if return_attn:
            return out, None
        return out


class SliceAttnStateEncoderV2(nn.Module):
    """
    Lightweight slice-token self-attention encoder aligned with
    HierarchicalSliceAttnExtractor (PPO).

    Architecture:
      inter (B,K,5,8) → 5 slice tokens
      intra (B,K,25,7) → per-slice mean-pool → 5 tokens (7-dim)
      Concat per slice: 5 × 15 → Linear(15 → hidden) → 5 tokens
      + global(2) → Linear(2 → hidden) → 1 token
      → Self-Attention over 6 tokens
      → Flatten(6 × hidden) → MLP → embed_dim
    """

    def __init__(self, embed_dim=256, inter_dim=8, intra_dim=7,
                 num_slices=5, num_users=25, global_dim=2,
                 hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_slices = num_slices
        self.users_per_slice = num_users // num_slices
        self.intra_dim = intra_dim
        self.hidden_dim = hidden_dim

        slice_in_dim = inter_dim + intra_dim  # 15
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
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        token_count = num_slices + 1  # 6
        self.output_mlp = nn.Sequential(
            nn.Linear(token_count * hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, inter, intra, glob, return_attn=False):
        B, K, N_S, F_Inter = inter.shape
        _, _, N_U, F_Intra = intra.shape
        BK = B * K

        inter = inter.view(BK, N_S, F_Inter)
        intra = intra.view(BK, N_U, F_Intra)
        glob = glob.view(BK, -1)

        intra_reshaped = intra.reshape(
            BK, self.num_slices, self.users_per_slice, self.intra_dim
        )
        intra_pooled = intra_reshaped.mean(dim=2)                    # (BK, 5, 7)

        slice_in = torch.cat([inter, intra_pooled], dim=-1)          # (BK, 5, 15)
        slice_tokens = self.slice_proj(slice_in)                      # (BK, 5, hidden)
        global_token = self.global_proj(glob).unsqueeze(1)            # (BK, 1, hidden)

        tokens = torch.cat([slice_tokens, global_token], dim=1)       # (BK, 6, hidden)

        attn_out, attn_weights = self.self_attn(
            tokens, tokens, tokens,
            need_weights=return_attn, average_attn_weights=True,
        )
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        out = self.output_mlp(tokens.reshape(BK, -1)).view(B, K, -1)

        if return_attn:
            return out, attn_weights.view(B, K, self.num_slices + 1, -1)
        return out


_ENCODER_MAP = {
    "cross_attn": HierarchicalStateEncoderV2,
    "mlp": MLPStateEncoderV2,
    "slice_attn": SliceAttnStateEncoderV2,
}


def build_dt_v2(model_cfg, action_dims, inter_dim=8, intra_dim=7,
                embed_dim=512, encoder_type="cross_attn",
                encoder_hidden_dim=None, encoder_num_heads=None):
    """Build a Decision Transformer with V2 state encoder."""
    encoder_cls = _ENCODER_MAP[encoder_type]
    encoder_kwargs = dict(
        embed_dim=embed_dim, inter_dim=inter_dim, intra_dim=intra_dim,
    )
    if encoder_type == "slice_attn":
        if encoder_hidden_dim is not None:
            encoder_kwargs["hidden_dim"] = int(encoder_hidden_dim)
        if encoder_num_heads is not None:
            encoder_kwargs["num_heads"] = int(encoder_num_heads)
    state_encoder = encoder_cls(**encoder_kwargs)
    return _build_decision_model(
        model_cfg=model_cfg, state_encoder=state_encoder, action_dims=action_dims,
    )
