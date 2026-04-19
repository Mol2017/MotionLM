"""Wayformer-aligned scene encoder (design doc §Stage 2).

Pipeline:
    per-modality MLP embed + PE + modality-type  (agents | roadgraph | tls)
    concat along token axis → [B', M, d]
    Perceiver: 192 learned latents cross-attend to the M scene tokens
    n_enc=6 self-attn blocks over the 192 latents
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.attention import MultiheadSDPA
from model.config import MotionLMConfig


def sinusoidal_pe(T: int, d: int, device: torch.device | None = None) -> torch.Tensor:
    pe = torch.zeros(T, d, device=device)
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverCrossAttn(nn.Module):
    """One cross-attn layer: latents (Q) ← scene tokens (K/V), with FFN + residuals."""

    def __init__(self, d: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = MultiheadSDPA(d, heads, dropout=dropout)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(
        self,
        latents: torch.Tensor,
        scene: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.ln_q(latents)
        kv = self.ln_kv(scene)
        attn = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        latents = latents + attn
        latents = latents + self.ff(self.ln_ff(latents))
        return latents


class EncoderSelfAttnBlock(nn.Module):
    """Pre-LN self-attn + FFN block over the 192 latents (SDPA; flash-path eligible)."""

    def __init__(self, d: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_sa = nn.LayerNorm(d)
        self.sa = MultiheadSDPA(d, heads, dropout=dropout)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_sa(x)
        x = x + self.sa(h, h, h)
        x = x + self.ff(self.ln_ff(x))
        return x


class SceneEncoder(nn.Module):
    def __init__(self, cfg: MotionLMConfig):
        super().__init__()
        self.cfg = cfg
        d, d_ff, heads = cfg.d, cfg.d_ff, cfg.heads

        # Per-modality embedders
        self.agent_mlp = MLP(cfg.D_a, d_ff, d)
        self.rg_mlp = MLP(cfg.D_r, d_ff, d)
        self.tl_mlp = MLP(cfg.D_tl, d_ff, d)

        # Positional encodings
        self.agent_pe = nn.Parameter(torch.randn(cfg.A, d) * 0.02)
        self.rg_pe = nn.Parameter(torch.randn(cfg.R, d) * 0.02)
        self.tl_slot_pe = nn.Parameter(torch.randn(cfg.L, d) * 0.02)
        self.register_buffer("time_pe", sinusoidal_pe(cfg.T_past, d), persistent=False)
        self.mod_embed = nn.Embedding(3, d)  # 0=agent, 1=roadgraph, 2=tl

        # Perceiver + self-attn stack
        self.latents = nn.Parameter(torch.randn(cfg.latents, d) * 0.02)
        self.perceiver = PerceiverCrossAttn(d, heads, d_ff, cfg.dropout)
        self.self_attn = nn.ModuleList(
            [EncoderSelfAttnBlock(d, heads, d_ff, cfg.dropout) for _ in range(cfg.n_enc)]
        )

    # --- per-modality helpers ---

    def _embed_agents(
        self, agents: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        Bp, A, Tp, _ = agents.shape
        x = self.agent_mlp(agents)
        x = (
            x
            + self.agent_pe.view(1, A, 1, -1)
            + self.time_pe.view(1, 1, Tp, -1)
            + self.mod_embed.weight[0].view(1, 1, 1, -1)
        )
        x = x.reshape(Bp, A * Tp, -1)
        kpad = (~mask).reshape(Bp, A * Tp) if mask is not None else None
        return x, kpad

    def _embed_roadgraph(
        self, rg: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        Bp, R, P, _ = rg.shape
        x = self.rg_mlp(rg)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = x.amax(dim=-2)  # [Bp, R, d]
        if mask is not None:
            chunk_valid = mask.any(dim=-1)
            pooled = torch.where(chunk_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
            kpad = ~chunk_valid
        else:
            kpad = None
        pooled = pooled + self.rg_pe.view(1, R, -1) + self.mod_embed.weight[1].view(1, 1, -1)
        return pooled, kpad

    def _embed_tl(
        self, tl: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        Bp, Tp, L, _ = tl.shape
        x = self.tl_mlp(tl)
        x = (
            x
            + self.time_pe.view(1, Tp, 1, -1)
            + self.tl_slot_pe.view(1, 1, L, -1)
            + self.mod_embed.weight[2].view(1, 1, 1, -1)
        )
        x = x.reshape(Bp, Tp * L, -1)
        kpad = (~mask).reshape(Bp, Tp * L) if mask is not None else None
        return x, kpad

    # --- forward ---

    def forward(
        self,
        agent_history: torch.Tensor,
        agent_mask: torch.Tensor | None,
        roadgraph: torch.Tensor,
        roadgraph_mask: torch.Tensor | None,
        traffic_lights: torch.Tensor,
        tl_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            agent_history:   [B', A, T_past, D_a]
            agent_mask:      [B', A, T_past] bool (True = valid)
            roadgraph:       [B', R, P, D_r]
            roadgraph_mask:  [B', R, P] bool
            traffic_lights:  [B', T_past, L, D_tl]
            tl_mask:         [B', T_past, L] bool
        Returns:
            scene_latents:   [B', latents, d]
        """
        use_ckpt = self.cfg.grad_checkpoint and self.training

        if use_ckpt and roadgraph.requires_grad:
            # Roadgraph MLP dominates activation memory (B·R·P·d_ff ≈ 4 GB at B=64).
            # Checkpoint the embedders so this is dropped between fwd and bwd.
            a_tok, a_pad = checkpoint(self._embed_agents, agent_history, agent_mask,
                                      use_reentrant=False)
            r_tok, r_pad = checkpoint(self._embed_roadgraph, roadgraph, roadgraph_mask,
                                      use_reentrant=False)
            t_tok, t_pad = checkpoint(self._embed_tl, traffic_lights, tl_mask,
                                      use_reentrant=False)
        else:
            a_tok, a_pad = self._embed_agents(agent_history, agent_mask)
            r_tok, r_pad = self._embed_roadgraph(roadgraph, roadgraph_mask)
            t_tok, t_pad = self._embed_tl(traffic_lights, tl_mask)

        scene = torch.cat([a_tok, r_tok, t_tok], dim=1)
        Bp = scene.size(0)

        if a_pad is None and r_pad is None and t_pad is None:
            kpad = None
        else:
            def _or_zeros(m: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
                if m is not None:
                    return m
                return torch.zeros(ref.size(0), ref.size(1), dtype=torch.bool, device=ref.device)

            kpad = torch.cat(
                [_or_zeros(a_pad, a_tok), _or_zeros(r_pad, r_tok), _or_zeros(t_pad, t_tok)], dim=1
            )

        latents = self.latents.view(1, self.cfg.latents, -1).expand(Bp, -1, -1).contiguous()

        if use_ckpt and scene.requires_grad:
            latents = checkpoint(self.perceiver, latents, scene, kpad, use_reentrant=False)
        else:
            latents = self.perceiver(latents, scene, key_padding_mask=kpad)

        for blk in self.self_attn:
            if use_ckpt and latents.requires_grad:
                latents = checkpoint(blk, latents, use_reentrant=False)
            else:
                latents = blk(latents)
        return latents
