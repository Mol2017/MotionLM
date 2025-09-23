import torch
import torch.nn as nn

from models.config import MotionLMConfig


class TokenEmbedding(nn.Module):
    """Create embeddings from motion tokens."""
    def __init__(self, cfg: MotionLMConfig):
        super().__init__()
        D = cfg.d_model
        self.motion_token_embed = nn.Embedding(cfg.vocab_size+1, D) # +1 for BOS
        self.time_embed         = nn.Embedding(cfg.T,       D)

    def forward(self, tok_ids, t_idx):
        """
        Inputs:
        tok_ids:            [B, T]
        t_idx:              [B, T]

        Returns:
        embeddings:         [B, T, D]
        """
        return self.motion_token_embed(tok_ids) + self.time_embed(t_idx)


class CrossAttn(nn.Module):
    """Cross-attention from the query motion token embeddings to the scene memory."""
    def __init__(self, d_model, nhead, ffw_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ffw_mult*d_model), nn.ReLU(),
                                nn.Linear(ffw_mult*d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_q, mem_kv):
        """
        Inputs:
        x_q:              [B, T, D]
        mem_kv:           [B, Hs, D]

        Returns:
        x:                [B, T, D]
        """
        z, _ = self.attn(query=x_q, key=mem_kv, value=mem_kv)
        x = self.ln1(x_q + z)
        return self.ln2(x + self.ff(x))


class MotionDecoder(nn.Module):
    """
    Motion Decoder for motion prediction.

    The decoder generates motion tokens by:
    1) applying self-attention to the query token embeddings.
    2) applying cross-attention from the query motion token embeddings to the scene memory.
    """
    def __init__(self, cfg: MotionLMConfig):
        super().__init__()
        D = cfg.d_model
        self.D = D
        self.T = cfg.T

        # Query token embedding
        self.embed = TokenEmbedding(cfg)
        
        # Query token self attention
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=D, nhead=cfg.nhead, batch_first=True,
                                       dim_feedforward=cfg.ffw_mult*D, dropout=cfg.dropout)
            for _ in range(cfg.N_dec_self_attn_layers)
        ])

        # Cross attention
        self.cross_attn = CrossAttn(D, cfg.nhead, ffw_mult=cfg.ffw_mult, dropout=cfg.dropout)

        # Output layer
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size)

    def _build_causal_mask(self, T: int, device: torch.device):
        """Build causal mask for [T, T] attention."""
        # Lower triangular mask (including diagonal)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, tokens, scene_mem):
        """
        Inputs:
        tokens:      [B, T]
        scene_mem:   [B, Hs, D]

        Returns:
        logits:      [B, T, V]
        """
        B, T = tokens.shape
        D = self.D
        device = tokens.device

        # ----- Build query token embedding -----
        # [T]
        t_idx = torch.arange(T, device=device)
        # [B, T]
        t_idx = t_idx.unsqueeze(0).expand(B, T)
        # [B, T, D]
        x = self.embed(tokens, t_idx)

        # ----- Query token self attention -----
        # [T, T] - causal mask
        causal_mask = self._build_causal_mask(T, device)

        for lyr in self.self_attn_layers:
            # [B, T, D]
            x = lyr(x, src_mask=causal_mask)

        # ----- Cross attention between query tokens and scene memory -----
        # [B, T, D]
        x = self.cross_attn(x, scene_mem)

        # ----- Output motion tokens -----
        # [B, T, D]
        out = self.ln(x)
        # [B, T, V]
        logits = self.head(out)
        return logits