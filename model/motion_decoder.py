"""Joint multi-agent causal decoder (design doc §Stage 3).

Operates on the flattened joint action sequence of shape ``[B, N*T, d]`` so the
block-staircase causal mask can let agents attend to each other **within the same
timestep** (recovers the true joint ``P(a₀_t, a₁_t | history)``).

Each of ``n_dec`` blocks is ``(self-attn → cross-attn → FFN)`` with pre-LN and
residuals. Cross-attn is per-agent routed: token ``(a, t)`` attends only to that
agent's scene latents.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.attention import MultiheadSDPA
from model.config import MotionLMConfig
from model.scene_encoder import sinusoidal_pe


def build_block_staircase_mask(N: int, T: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[N*T, N*T]`` mask (True = blocked) for the flattened sequence.

    Flattening convention: row-major with agent outer and time inner, so
    ``idx = a * T + t``. A query at ``(a_q, t_q)`` may attend to a key at
    ``(a_k, t_k)`` iff ``t_k ≤ t_q`` — identical to the diagram in §3.3a.
    """
    t_idx = torch.arange(T, device=device)
    t_flat = t_idx.view(1, T).expand(N, T).reshape(-1)  # [N*T]
    allow = t_flat.unsqueeze(0) <= t_flat.unsqueeze(1)  # [N*T, N*T]
    return ~allow


class DecoderBlock(nn.Module):
    """One ``(SA staircase → CA per-agent → FFN)`` block."""

    def __init__(self, d: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_sa = nn.LayerNorm(d)
        self.sa = MultiheadSDPA(d, heads, dropout=dropout)
        self.ln_ca = nn.LayerNorm(d)
        self.ca = MultiheadSDPA(d, heads, dropout=dropout)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(
        self,
        x: torch.Tensor,
        sa_mask: torch.Tensor,
        kv: torch.Tensor,
        kv_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:       [B, N*T, d]
        sa_mask: [N*T, N*T] bool
        kv:      [B, N, latents, d]
        kv_pad:  [B, N, latents] bool or None
        """
        B, NT, d = x.shape
        _, N, latents, _ = kv.shape
        T = NT // N

        # self-attn (block-staircase)
        h = self.ln_sa(x)
        sa_out = self.sa(h, h, h, attn_mask=sa_mask)
        x = x + sa_out

        # cross-attn per-agent routed
        q = self.ln_ca(x).view(B, N, T, d).reshape(B * N, T, d)
        kv_flat = kv.reshape(B * N, latents, d)
        kpad = kv_pad.reshape(B * N, latents) if kv_pad is not None else None
        ca_out = self.ca(q, kv_flat, kv_flat, key_padding_mask=kpad)
        x = x + ca_out.reshape(B, N, T, d).reshape(B, NT, d)

        # FFN
        x = x + self.ff(self.ln_ff(x))
        return x


class MotionDecoder(nn.Module):
    def __init__(self, cfg: MotionLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        # embedding table; +1 reserves BOS_ID
        self.tok_embed = nn.Embedding(cfg.vocab_size + 1, d)
        self.register_buffer("time_pe", sinusoidal_pe(cfg.T, d), persistent=False)
        self.agent_pe = nn.Parameter(torch.randn(cfg.N_max, d) * 0.02)

        self.blocks = nn.ModuleList(
            [DecoderBlock(d, cfg.heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_dec)]
        )
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)

    def forward(
        self,
        action_tokens: torch.Tensor,
        scene_latents: torch.Tensor,
        kv_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            action_tokens: [B, N, T] int (may contain BOS = cfg.BOS_ID)
            scene_latents: [B*N, latents, d]
            kv_pad:        [B*N, latents] bool (True = pad) or None

        Returns:
            logits: [B, N, T, vocab_size]
        """
        B, N, T = action_tokens.shape
        assert N <= self.cfg.N_max
        d = self.cfg.d
        latents = self.cfg.latents

        x = self.tok_embed(action_tokens)
        x = (
            x
            + self.time_pe.view(1, 1, T, d)
            + self.agent_pe[:N].view(1, N, 1, d)
        )
        x = x.reshape(B, N * T, d)

        kv = scene_latents.view(B, N, latents, d)
        kv_pad_r = kv_pad.view(B, N, latents) if kv_pad is not None else None

        sa_mask = build_block_staircase_mask(N, T, action_tokens.device)
        use_ckpt = self.cfg.grad_checkpoint and self.training and x.requires_grad
        for blk in self.blocks:
            if use_ckpt:
                x = checkpoint(blk, x, sa_mask, kv, kv_pad_r, use_reentrant=False)
            else:
                x = blk(x, sa_mask, kv, kv_pad_r)

        x = self.ln_out(x).view(B, N, T, d)
        return self.head(x)
