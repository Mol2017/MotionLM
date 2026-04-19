"""Verlet-wrapped motion tokenizer (MotionLM paper Appendix A).

Two-layer quantization:

1. **Raw Δ-action space**: uniform bins over ``[-bin_range, +bin_range]`` per axis,
   ``raw_bins_per_coord`` bins → ``bin_step = 2·bin_range / raw_bins_per_coord``.
   Paper values: ±18 m, 128 bins → **0.28125 m per bin**.
2. **Verlet wrap**: the emitted token is the signed offset ``o_t = b_t − b_{t−1}``
   in bin units, clipped to ``|o| ≤ C = (bins_per_coord − 1) // 2``. Paper uses
   ``bins_per_coord = 13`` → ``C = 6`` → vocab = 169.

Decode recurrence::

    b_t   = b_{t-1} + offset(token_t)          # bin index (integer, can grow)
    pos_t = pos_{t-1} + b_t * bin_step          # physical Δ accumulates

Initial conditions: ``b_{-1} = 0`` and ``pos_{-1} = 0`` (agent-centric frame at t=0).
An optional ``init_bin`` argument lets callers seed the recurrence from the observed
velocity at t=0 so a fast-moving vehicle doesn't spend several steps ramping offsets
up to match its actual speed.

Token packing: ``id = (ox + C) * B + (oy + C)``, so id ∈ [0, B²).
Offset (0, 0) = "repeat previous Δ-bin" lives at id ``B² // 2``.
"""

from __future__ import annotations

import torch

from model.config import MotionLMConfig


class MotionTokenizer:
    """Pure-math Verlet tokenizer; not an ``nn.Module`` (no learnable parameters)."""

    def __init__(self, cfg: MotionLMConfig):
        self.cfg = cfg
        self.B = cfg.bins_per_coord
        assert self.B % 2 == 1, "bins_per_coord must be odd so offset=0 is centered"
        self.C = (self.B - 1) // 2                           # max signed Verlet offset (6 for B=13)
        self.V = self.B * self.B                             # vocab_size (169)
        # Raw Δ-bin granularity from the paper: 36 m / 128 = 0.28125 m at defaults.
        self.bin_step = 2.0 * cfg.bin_range / cfg.raw_bins_per_coord

    # --- packing ---

    def pack(self, ox: torch.Tensor, oy: torch.Tensor) -> torch.Tensor:
        return ((ox + self.C) * self.B + (oy + self.C)).long()

    def unpack(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ox = torch.div(ids, self.B, rounding_mode="floor") - self.C
        oy = (ids % self.B) - self.C
        return ox.long(), oy.long()

    # --- encode / decode ---

    @torch.no_grad()
    def delta_to_init_bin(
        self,
        dx: torch.Tensor | float,
        dy: torch.Tensor | float,
    ) -> torch.Tensor:
        """Quantize a per-step agent-frame displacement (metres) to the Δ-bin at t=−1.

        This is the canonical path for seeding the Verlet recurrence. ``(dx, dy)``
        should be the agent's displacement over **one tokenizer step** — at 2 Hz,
        that's the 0.5 s window. Best practice: compute from a backward finite
        difference of the past trajectory (``pos[cti] − pos[cti−5]`` at 10 Hz)
        since that directly matches the frequency of the emitted tokens.

        Args:
            dx, dy: scalars or broadcastable tensors, metres per step (agent frame).

        Returns:
            signed int64 tensor ``[..., 2]`` clipped to the raw bin half-range.
        """
        dx_t = torch.as_tensor(dx, dtype=torch.float32)
        dy_t = torch.as_tensor(dy, dtype=torch.float32)
        clip = self.cfg.raw_bins_per_coord // 2
        bx = torch.round(dx_t / self.bin_step).clamp(-clip, clip).long()
        by = torch.round(dy_t / self.bin_step).clamp(-clip, clip).long()
        return torch.stack([bx, by], dim=-1)

    @torch.no_grad()
    def velocity_to_init_bin(
        self,
        vx: torch.Tensor | float,
        vy: torch.Tensor | float,
        dt: float = 0.5,
    ) -> torch.Tensor:
        """Convenience wrapper — quantize a velocity (m/s) assuming constant motion over ``dt``.

        Equivalent to ``delta_to_init_bin(vx * dt, vy * dt)``. Prefer calling
        :meth:`delta_to_init_bin` with a backward-FD displacement when past
        positions are available (more accurate under non-zero acceleration).
        """
        return self.delta_to_init_bin(
            torch.as_tensor(vx, dtype=torch.float32) * dt,
            torch.as_tensor(vy, dtype=torch.float32) * dt,
        )

    @torch.no_grad()
    def encode(
        self,
        pos: torch.Tensor,
        init_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Trajectory → Verlet tokens (greedy, closed-loop).

        Args:
            pos: ``[..., T, 2]`` agent-frame waypoints at 2 Hz (``pos_{-1}`` assumed 0).
            init_bin: optional ``[..., 2]`` signed integer Δ-bin at t=−1 (i.e., observed
                velocity at t=0 expressed in bin units). ``None`` → zeros.

        Returns:
            ``[..., T]`` long tokens in ``[0, V)``.
        """
        *batch_shape, T, two = pos.shape
        assert two == 2
        flat = pos.reshape(-1, T, 2)
        Bf = flat.shape[0]

        if init_bin is None:
            prev_bin = torch.zeros(Bf, 2, dtype=torch.long, device=pos.device)
        else:
            prev_bin = init_bin.reshape(-1, 2).to(torch.long).to(pos.device)
        prev_pos = torch.zeros(Bf, 2, dtype=pos.dtype, device=pos.device)
        tokens = torch.empty(Bf, T, dtype=torch.long, device=pos.device)

        for t in range(T):
            target_delta = flat[:, t] - prev_pos
            target_bin = torch.round(target_delta / self.bin_step).long()
            offset = (target_bin - prev_bin).clamp(-self.C, self.C)
            actual_bin = prev_bin + offset
            actual_delta = actual_bin.to(pos.dtype) * self.bin_step
            prev_pos = prev_pos + actual_delta
            prev_bin = actual_bin
            tokens[:, t] = self.pack(offset[:, 0], offset[:, 1])

        return tokens.view(*batch_shape, T)

    @torch.no_grad()
    def decode(
        self,
        tokens: torch.Tensor,
        init_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Verlet tokens → trajectory (fully vectorized via cumsum).

        Args:
            tokens: ``[..., T]`` long in ``[0, V)``.
            init_bin: optional ``[..., 2]`` signed integer Δ-bin at t=−1; broadcasts
                over leading dims. ``None`` → zeros.

        Returns:
            ``[..., T, 2]`` agent-frame waypoints at 2 Hz.
        """
        ox, oy = self.unpack(tokens)
        offsets = torch.stack([ox, oy], dim=-1)              # [..., T, 2]
        bins = torch.cumsum(offsets, dim=-2)                 # [..., T, 2]
        if init_bin is not None:
            # Broadcast init_bin [..., 2] → [..., 1, 2] so it adds along the T axis.
            ib = init_bin.to(bins.dtype)
            while ib.dim() < bins.dim():
                ib = ib.unsqueeze(-2)
            bins = bins + ib
        pos = torch.cumsum(bins.to(torch.float32) * self.bin_step, dim=-2)
        return pos
