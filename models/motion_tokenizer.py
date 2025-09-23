import torch
import torch.nn as nn

from models.config import MotionLMConfig


class MotionTokenizer(nn.Module):
    """
    Verlet-wrapped motion tokenizer.
    - Vocab id encodes a small correction (dx_corr, dy_corr) chosen from a BxB grid.
    - The actual step displacement follows: Δ_t = Δ_{t-1} + δ_t.
    """
    def __init__(self, cfg: MotionLMConfig):
        self.cfg = cfg
        B = cfg.bins_per_coord
        self.B = B
        self.V = B * B

        # Uniform bin centers in [-max_corr, +max_corr]
        self.centers = torch.linspace(-cfg.max_corr, cfg.max_corr, B)

    # ---------- packing / unpacking (ix, iy) <-> token id ----------
    def pack(self, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        """(ix, iy) in [0..B-1] → id in [0..V-1]"""
        return (ix * self.B + iy).long()

    def unpack(self, ids: torch.Tensor):
        """id in [0..V-1] → (ix, iy) in [0..B-1]"""
        ix = (ids // self.B).long()
        iy = (ids %  self.B).long()
        return ix, iy

    # ---------- quantize / dequantize a single 1D correction ----------
    def _quantize_1d(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -self.cfg.max_corr, self.cfg.max_corr)
        # [..., B]
        d = (x.unsqueeze(-1) - self.centers.view(1, -1))**2
        idx = torch.argmin(d, dim=-1)
        return idx

    def _dequant_1d(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centers[idx]

    # ---------- public API ----------
    @torch.no_grad()
    def xy_to_tokens(self, pos: torch.Tensor, start_prev_delta=None) -> torch.Tensor:
        """
        Encode XY trajectory → correction tokens.

        Inputs:
        pos: [..., T, 2]
        start_prev_delta: None or [..., 2]

        Returns:
        tokens: [..., T]
        """
        batch_dims = pos.shape[:-2]
        T = pos.shape[-2]
        
        # [B, T, 2]
        pos_flat = pos.reshape(-1, T, 2)
        B = pos_flat.size(0)
        
        # ----- (x, y) -> (Δ_x, Δ_y) -----
        # [B, T, 2]
        deltas = pos_flat.clone()
        # [B, T, 2]
        deltas[:, 1:, :] = pos_flat[:, 1:, :] - pos_flat[:, :-1, :]
        
        # ----- (Δ_x, Δ_y) -> tokens -----
        # [B, T]
        tokens = torch.empty(B, T, dtype=torch.long, device=pos.device)
        
        if start_prev_delta is None:
            # [B, 2]
            prev_delta = torch.zeros(B, 2, device=pos.device, dtype=pos.dtype)
        else:
            # [B, 2]
            prev_delta = start_prev_delta.view(-1, 2)  
        
        for t in range(T):
            # [B, 2]
            corr = deltas[:, t] - prev_delta
            
            # [B]
            ix = self._quantize_1d(corr[:, 0])
            iy = self._quantize_1d(corr[:, 1])

            # [B]
            token = self.pack(ix, iy)
            tokens[:, t] = token

            # [B]
            q_corr_x = self._dequant_1d(ix)
            q_corr_y = self._dequant_1d(iy)

            # [B, 2]
            q_corr = torch.stack([q_corr_x, q_corr_y], dim=-1)
            # [B, 2]
            prev_delta = prev_delta + q_corr

        # [..., T]
        return tokens.view(*batch_dims, T)

    @torch.no_grad()
    def tokens_to_deltas(self, tokens: torch.Tensor, start_prev_delta=None) -> torch.Tensor:
        """
        Decode correction tokens → displacements Δ_x, Δ_y.

        Inputs:
        tokens: [..., T]
        start_prev_delta: None or [..., 2]

        Returns:
        deltas [..., T, 2]
        """
        batch_dims = tokens.shape[:-1]
        T = tokens.shape[-1]
        
        # [B, T]
        tokens_flat = tokens.view(-1, T)
        B = tokens_flat.size(0)
        
        # [B, T, 2]
        deltas = torch.empty(B, T, 2, device=tokens.device)
        
        if start_prev_delta is None:
            # [B, 2]
            prev_delta = torch.zeros(B, 2, device=tokens.device)
        else:
            # [B, 2]
            prev_delta = start_prev_delta.view(-1, 2)  
        
        for t in range(T):
            # [B]
            ix, iy = self.unpack(tokens_flat[:, t])
            # [B]
            dx = self._dequant_1d(ix)
            dy = self._dequant_1d(iy)
            # [B, 2]
            q_corr = torch.stack([dx, dy], dim=-1)
            # [B, 2]
            prev_delta = prev_delta + q_corr  # Δ_t = Δ_{t-1} + δ̂_t
            deltas[:, t] = prev_delta
        
        # [..., T, 2]
        return deltas.view(*batch_dims, T, 2)

    @torch.no_grad()
    def deltas_to_xy(self, deltas: torch.Tensor, start_pos=None) -> torch.Tensor:
        """
        Decode displacements Δ_x, Δ_y → XY trajectory.
        
        Inputs:
        deltas: [..., T, 2]
        start_pos: [..., 2]

        Returns:
        pos [..., T, 2]
        """
        batch_dims = deltas.shape[:-2]
        T = deltas.shape[-2]
        
        # [B, T, 2]
        deltas_flat = deltas.view(-1, T, 2)
        B = deltas_flat.size(0)
        
        # [B, T, 2]
        pos = torch.empty_like(deltas_flat)
        
        # [B, 2]
        if start_pos is None:
            cur_pos = torch.zeros(B, 2, device=deltas.device, dtype=deltas.dtype)
        else:
            cur_pos = start_pos.view(-1, 2)  # [B, 2]

        for t in range(T):
            # [B, 2]
            cur_pos = cur_pos + deltas_flat[:, t]
            pos[:, t] = cur_pos
        
        # [..., T, 2]
        return pos.view(*batch_dims, T, 2)

    @torch.no_grad()
    def tokens_to_xy(self, tokens: torch.Tensor, start_pos=None, start_prev_delta=None) -> torch.Tensor:
        """
        End-to-end decode: tokens → Δ → XY.

        Inputs:
            tokens: [..., T]
            start_pos: [..., 2]
            start_prev_delta: [..., 2]

        Returns: 
            pos [..., T, 2]
        """
        deltas = self.tokens_to_deltas(tokens, start_prev_delta=start_prev_delta)
        return self.deltas_to_xy(deltas, start_pos=start_pos)
