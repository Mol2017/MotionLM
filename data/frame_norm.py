"""Stage 0 — per-modeled-agent frame normalization + ROI clip.

All functions are pure-torch so the offline converter and online debug/eval paths
share the same implementation. See design doc §"Stage 0 — Frame normalization &
ROI clip".
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

# Design doc §"ROI clip": agent-centric 160 m × 100 m box.
ROI_BOX: tuple[tuple[float, float], tuple[float, float]] = ((-40.0, 120.0), (-50.0, 50.0))


class FrameParams(NamedTuple):
    x0: float
    y0: float
    h0: float


def wrap_angle(h: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(h), torch.cos(h))


def _rotate_by_neg_h(xy: torch.Tensor, h: float) -> torch.Tensor:
    c = math.cos(h)
    s = math.sin(h)
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack([x * c + y * s, -x * s + y * c], dim=-1)


def normalize_frame(
    agents_xy: torch.Tensor,
    agents_h: torch.Tensor,
    roadgraph_xy: torch.Tensor,
    tl_xy: torch.Tensor,
    x0: float,
    y0: float,
    h0: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FrameParams]:
    """Subtract ``(x0, y0)``, rotate by ``-h0``, wrap headings to ``[-π, π]``.

    Works on any xy-typed tensor shape (only the last dim =2 matters).
    """
    origin = torch.tensor([x0, y0], dtype=agents_xy.dtype, device=agents_xy.device)
    agents_xy_local = _rotate_by_neg_h(agents_xy - origin, h0)
    roadgraph_xy_local = _rotate_by_neg_h(roadgraph_xy - origin, h0)
    tl_xy_local = _rotate_by_neg_h(tl_xy - origin, h0)
    agents_h_local = wrap_angle(agents_h - h0)
    return (
        agents_xy_local,
        agents_h_local,
        roadgraph_xy_local,
        tl_xy_local,
        FrameParams(x0, y0, h0),
    )


def in_roi(
    xy: torch.Tensor,
    box: tuple[tuple[float, float], tuple[float, float]] = ROI_BOX,
) -> torch.Tensor:
    (xmin, xmax), (ymin, ymax) = box
    x = xy[..., 0]
    y = xy[..., 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def rank_context_agents(
    agents_xy_local: torch.Tensor,
    modeled_idx: int,
    A: int,
    current_time_index: int,
) -> torch.Tensor:
    """Return the ``A`` kept agent indices (modeled at slot 0, then nearest neighbors).

    Args:
        agents_xy_local: ``[A_raw, T_past, 2]``
    """
    A_raw = agents_xy_local.shape[0]
    device = agents_xy_local.device
    modeled_pos = agents_xy_local[modeled_idx, current_time_index]
    others = [i for i in range(A_raw) if i != modeled_idx]
    if not others:
        return torch.tensor([modeled_idx], device=device, dtype=torch.long)
    other_pos = agents_xy_local[others, current_time_index]
    d = torch.linalg.vector_norm(other_pos - modeled_pos, dim=-1)
    k = min(A - 1, d.numel())
    top = torch.topk(d, k, largest=False).indices
    kept_others = torch.as_tensor(
        [others[i] for i in top.tolist()], device=device, dtype=torch.long
    )
    return torch.cat([torch.tensor([modeled_idx], device=device, dtype=torch.long), kept_others])


def rank_tls_per_timestep(
    tl_xy_local: torch.Tensor,
    L: int,
    tl_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-timestep proximity-rank TLs to origin (modeled agent), keep top ``L``.

    Returns:
        ``[T_past, L]`` long indices (``-1`` where no TL available).
    """
    Tp, L_raw, _ = tl_xy_local.shape
    device = tl_xy_local.device
    out = torch.full((Tp, L), -1, device=device, dtype=torch.long)
    for t in range(Tp):
        d = torch.linalg.vector_norm(tl_xy_local[t], dim=-1)
        if tl_valid is not None:
            d = torch.where(tl_valid[t], d, torch.full_like(d, float("inf")))
        valid_count = int(torch.isfinite(d).sum().item())
        k = min(L, valid_count)
        if k == 0:
            continue
        top = torch.topk(d, k, largest=False).indices
        out[t, :k] = top
    return out


def chunk_roadgraph(
    rg_xy_local: torch.Tensor,
    rg_feature_id: torch.Tensor,
    P: int,
    R: int,
    roi_box: tuple[tuple[float, float], tuple[float, float]] = ROI_BOX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ROI-clip then split per-feature into chunks of ``P``; stratified-truncate to ``R``.

    Args:
        rg_xy_local:   ``[N_rg_raw, 2]`` agent-frame points
        rg_feature_id: ``[N_rg_raw]``    integer feature membership (points sharing an id
                                         belong to the same polyline / feature).

    Returns:
        chunk_slot: ``[N_kept]`` int in ``[0, R)`` — which chunk the point landed in
        point_slot: ``[N_kept]`` int in ``[0, P)`` — which point within that chunk
        orig_idx:   ``[N_kept]`` int — index back into ``rg_xy_local`` / feature vectors
    """
    device = rg_xy_local.device
    keep = in_roi(rg_xy_local, roi_box)
    orig = keep.nonzero(as_tuple=False).squeeze(-1)
    if orig.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, z

    feat = rg_feature_id[orig]

    # per-feature chunk lists (each chunk is a [<=P] tensor of original indices)
    chunks_by_feat: dict[int, list[torch.Tensor]] = {}
    for fid in torch.unique(feat).tolist():
        pts = orig[feat == fid]
        chunks_by_feat[fid] = list(torch.split(pts, P))

    # stratified truncation: 1 per feature, then proportional
    fids = list(chunks_by_feat.keys())
    total = sum(len(cs) for cs in chunks_by_feat.values())
    caps: dict[int, int] = {fid: 0 for fid in fids}
    if total <= R:
        caps = {fid: len(cs) for fid, cs in chunks_by_feat.items()}
    else:
        remaining = R
        for fid in fids:
            if remaining == 0:
                break
            caps[fid] = 1
            remaining -= 1
        while remaining > 0:
            progressed = False
            for fid in sorted(fids, key=lambda f: -len(chunks_by_feat[f])):
                if caps[fid] < len(chunks_by_feat[fid]):
                    caps[fid] += 1
                    remaining -= 1
                    progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break

    chunk_slot_parts: list[torch.Tensor] = []
    point_slot_parts: list[torch.Tensor] = []
    orig_parts: list[torch.Tensor] = []
    slot = 0
    for fid, chunks in chunks_by_feat.items():
        for c in chunks[: caps.get(fid, 0)]:
            if slot >= R:
                break
            n = c.numel()
            chunk_slot_parts.append(torch.full((n,), slot, dtype=torch.long, device=device))
            point_slot_parts.append(torch.arange(n, dtype=torch.long, device=device))
            orig_parts.append(c)
            slot += 1

    return (
        torch.cat(chunk_slot_parts),
        torch.cat(point_slot_parts),
        torch.cat(orig_parts),
    )
