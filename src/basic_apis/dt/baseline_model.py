import torch
import torch.nn as nn
import math


class BaselineFlatStateEncoder(nn.Module):

    def __init__(self, obs_dim=145, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, flat_obs):
        return self.net(flat_obs)


class DecisionTransformerBaseline(nn.Module):

    def __init__(self, state_encoder,
                 cont_dim=5,
                 disc_dims=None,
                 hidden_size=256, max_length=20, max_ep_len=1000,
                 n_layer=3, n_head=4, dropout=0.1, activation="relu"):
        super().__init__()
        if disc_dims is None:
            disc_dims = [3, 3, 3, 3, 3]

        self.state_encoder = state_encoder
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.n_head = n_head
        self.cont_dim = cont_dim
        self.disc_dims = disc_dims

        # ---------- Embeddings ----------
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)

        # Action embedding: continuous part + discrete parts, summed
        self.embed_action_cont = nn.Linear(cont_dim, hidden_size)
        self.embed_action_disc = nn.ModuleList([
            nn.Embedding(d, hidden_size) for d in disc_dims
        ])

        self.embed_ln = nn.LayerNorm(hidden_size)

        # ---------- GPT-2 Backbone (Pre-Norm) ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # ---------- Prediction Heads ----------
        self.predict_action_cont = nn.Sequential(
            nn.Linear(hidden_size, cont_dim),
            nn.Tanh(),
        )
        self.predict_action_disc = nn.Linear(hidden_size, sum(disc_dims))

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _embed_actions(self, actions):
        """Embed mixed (cont + disc) action tensor.

        Args:
            actions: (B, K, cont_dim + len(disc_dims)) float tensor.
                     First cont_dim dims are continuous [-1, 1],
                     remaining dims are discrete indices cast to float.
        Returns:
            (B, K, hidden_size)
        """
        cont_part = actions[:, :, :self.cont_dim]
        disc_part = actions[:, :, self.cont_dim:].long()

        emb = self.embed_action_cont(cont_part)
        for i, emb_layer in enumerate(self.embed_action_disc):
            emb = emb + emb_layer(disc_part[:, :, i])
        return emb

    def _build_causal_mask(self, B, K, device, dtype, attention_mask=None):
        """Construct combined causal + padding 3D attention mask.

        Returns:
            (B * n_head, 3K, 3K) float mask with -inf for blocked positions.
        """
        seq_len = 3 * K
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        if attention_mask is not None:
            pad_key = ~(attention_mask.to(device) > 0)               # [B, K]
            pad_key = pad_key.repeat_interleave(3, dim=1)            # [B, 3K]
            pad_key_2d = pad_key.unsqueeze(1).expand(B, seq_len, seq_len)
            combined = causal.unsqueeze(0) | pad_key_2d
            diag_idx = torch.arange(seq_len, device=device)
            combined[:, diag_idx, diag_idx] = False
        else:
            combined = causal.unsqueeze(0).expand(B, seq_len, seq_len)

        attn_mask = torch.zeros_like(combined, dtype=dtype)
        attn_mask.masked_fill_(combined, float('-inf'))
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_head, seq_len, seq_len)
        return attn_mask.reshape(B * self.n_head, seq_len, seq_len)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Args:
            states:  (B, K, obs_dim) flat float tensor
            actions: (B, K, cont_dim + n_disc) float tensor
            returns: (B, K, 1)
            timesteps: (B, K) long
            attention_mask: (B, K) or None

        Returns:
            dict with:
                cont_preds:  (B, K, cont_dim) in [-1, 1]
                disc_preds:  (B, K, sum(disc_dims)) raw logits
        """
        B, K, _ = returns.shape

        state_embeddings = self.state_encoder(states)
        rtg_embeddings = self.embed_rtg(returns)
        time_embeddings = self.embed_timestep(timesteps)
        action_embeddings = self._embed_actions(actions)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * K, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        attn_mask = self._build_causal_mask(
            B, K, stacked_inputs.device, stacked_inputs.dtype, attention_mask
        )

        x = self.transformer(stacked_inputs, mask=attn_mask)

        x = x.reshape(B, K, 3, self.hidden_size)
        s_outputs = x[:, :, 1, :]

        cont_preds = self.predict_action_cont(s_outputs)
        disc_preds = self.predict_action_disc(s_outputs)

        return {"cont_preds": cont_preds, "disc_preds": disc_preds}

    def get_context_embedding(self, states, actions, returns, timesteps, attention_mask=None):
        B, K, _ = returns.shape

        state_embeddings = self.state_encoder(states)
        rtg_embeddings = self.embed_rtg(returns)
        time_embeddings = self.embed_timestep(timesteps)
        action_embeddings = self._embed_actions(actions)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * K, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        attn_mask = self._build_causal_mask(
            B, K, stacked_inputs.device, stacked_inputs.dtype, attention_mask
        )

        x = self.transformer(stacked_inputs, mask=attn_mask)

        x = x.reshape(B, K, 3, self.hidden_size)
        s_outputs = x[:, :, 1, :]

        return s_outputs[:, -1, :]


def build_dt_baseline(model_cfg, obs_dim=145, cont_dim=5,
                      disc_dims=None, embed_dim=512):
    if disc_dims is None:
        disc_dims = [3, 3, 3, 3, 3]

    state_encoder = BaselineFlatStateEncoder(obs_dim=obs_dim, embed_dim=embed_dim)

    model = DecisionTransformerBaseline(
        state_encoder=state_encoder,
        cont_dim=cont_dim,
        disc_dims=disc_dims,
        hidden_size=embed_dim,
        max_length=int(getattr(model_cfg, "context_len", 20)),
        max_ep_len=int(getattr(model_cfg, "max_ep_len", 1000)),
        n_layer=int(getattr(model_cfg, "n_layer", 3)),
        n_head=int(getattr(model_cfg, "n_head", 4)),
        dropout=float(getattr(model_cfg, "dropout", 0.1)),
        activation=str(getattr(model_cfg, "activation", "relu")),
    )
    return model
