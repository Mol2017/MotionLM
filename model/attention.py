"""Shared attention helpers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSDPA(nn.Module):
    """Multi-head attention via ``F.scaled_dot_product_attention``.

    Dispatches to FlashAttention / memory-efficient kernels in fp16/bf16 when
    the mask configuration allows (no mask → Flash; bool mask → mem-efficient).
    Mask conventions follow ``nn.MultiheadAttention``: ``attn_mask`` bool
    ``True`` = blocked, ``key_padding_mask`` bool ``True`` = pad.
    """

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.d = d
        self.heads = heads
        self.d_h = d // heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.size(1)
        H, Dh = self.heads, self.d_h

        qh = self.q_proj(q).view(B, Lq, H, Dh).transpose(1, 2)
        kh = self.k_proj(k).view(B, Lk, H, Dh).transpose(1, 2)
        vh = self.v_proj(v).view(B, Lk, H, Dh).transpose(1, 2)

        sdpa_mask: torch.Tensor | None = None
        if attn_mask is not None or key_padding_mask is not None:
            # SDPA bool mask: True = attend. Invert MHA conventions.
            if attn_mask is not None:
                allow = ~attn_mask.view(1, 1, Lq, Lk)
            else:
                allow = torch.ones(1, 1, Lq, Lk, dtype=torch.bool, device=q.device)
            if key_padding_mask is not None:
                allow = allow & ~key_padding_mask.view(B, 1, 1, Lk)
            sdpa_mask = allow.expand(B, H, Lq, Lk)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=sdpa_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, Lq, self.d)
        return self.o_proj(out)
