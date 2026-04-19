"""Small shared utilities used by the model + training code."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spline_to_10hz(waypoints_2hz: torch.Tensor, T_future: int) -> torch.Tensor:
    """Upsample per-rollout waypoints from ``T`` @ 2 Hz to ``T_future`` @ 10 Hz.

    Linear interpolation (TODO: cubic spline for better derivative continuity).
    Shape: ``[..., T, 2] → [..., T_future, 2]``.
    """
    *lead, T, two = waypoints_2hz.shape
    assert two == 2
    flat = waypoints_2hz.reshape(-1, T, 2).transpose(1, 2)
    up = F.interpolate(flat, size=T_future, mode="linear", align_corners=True)
    up = up.transpose(1, 2).reshape(*lead, T_future, 2)
    return up
