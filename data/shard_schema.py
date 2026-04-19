"""On-disk sparse shard schema and sparse ↔ dense conversion.

Exactly mirrors the schema laid out in design doc §"Shard file schema".
Each shard is ``torch.save``-serialized + zstd-compressed:

    {"version": 1, "count": n_examples, "examples": [sparse_ex, ...]}

Rationale for the sparse representation: only non-padded slots are stored (type
fields are int8 indices rather than one-hot expansions), keeping each example
to ~65 KB after zstd-9.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F

from model.config import MotionLMConfig


class SparseExample(TypedDict, total=False):
    # roadgraph (sparse; one entry per valid point)
    rg_xyz: torch.Tensor        # fp16 [n_rg, 3]
    rg_dir: torch.Tensor        # fp16 [n_rg, 2]
    rg_type: torch.Tensor       # int8 [n_rg], ∈ 0..19
    rg_chunk_idx: torch.Tensor  # int16 [n_rg], ∈ 0..R-1
    rg_point_idx: torch.Tensor  # int8 [n_rg], ∈ 0..P-1

    # agents (sparse)
    ag_feats: torch.Tensor      # fp16 [n_ag, 10] (xyz + sin/cos h + vxy + LWH)
    ag_type: torch.Tensor       # int8 [n_ag], ∈ 0..2
    ag_slot: torch.Tensor       # int8 [n_ag], ∈ 0..A-1 (slot 0 = modeled)
    ag_time: torch.Tensor       # int8 [n_ag], ∈ 0..T_past-1

    # traffic lights (sparse)
    tl_feats: torch.Tensor      # fp16 [n_tl, 3]
    tl_state: torch.Tensor      # int8 [n_tl], ∈ 0..8
    tl_slot: torch.Tensor       # int8 [n_tl], ∈ 0..L-1
    tl_time: torch.Tensor       # int8 [n_tl], ∈ 0..T_past-1

    # training targets
    gt_tokens: torch.Tensor     # int16 [T=16]
    future_valid: torch.Tensor  # uint8 [2] — 16 bits packed
    init_bin: torch.Tensor      # int16 [2] — observed Δ-bin at t=0 (seeds Verlet decode)

    # metadata
    x0: float
    y0: float
    h0: float
    scenario_id: bytes
    track_id: int


def pack_future_valid(mask: torch.Tensor) -> torch.Tensor:
    """[T=16] bool → [2] uint8."""
    assert mask.numel() == 16 and mask.dtype == torch.bool
    bits = mask.to(torch.uint8)
    packed = torch.zeros(2, dtype=torch.uint8)
    for i in range(16):
        if bits[i]:
            packed[i // 8] |= 1 << (i % 8)
    return packed


def unpack_future_valid(packed: torch.Tensor) -> torch.Tensor:
    """[2] uint8 → [T=16] bool."""
    assert packed.numel() == 2 and packed.dtype == torch.uint8
    out = torch.zeros(16, dtype=torch.bool)
    for i in range(16):
        out[i] = bool(packed[i // 8].item() & (1 << (i % 8)))
    return out


def densify(
    ex: SparseExample,
    cfg: MotionLMConfig,
) -> dict[str, torch.Tensor]:
    """Sparse example (on-disk) → dense model-ready tensors for ONE modeled agent.

    Produces a dict with keys ``agent_history, agent_mask, roadgraph, roadgraph_mask,
    traffic_lights, tl_mask, gt_tokens, future_valid, x0, y0, h0, scenario_id, track_id``
    — matching the Stage 1 contract with shapes leading dim = 1 (single modeled agent).
    """
    A, Tp, D_a = cfg.A, cfg.T_past, cfg.D_a
    R, P, D_r = cfg.R, cfg.P, cfg.D_r
    L, D_tl = cfg.L, cfg.D_tl

    ah = torch.zeros(A, Tp, D_a)
    am = torch.zeros(A, Tp, dtype=torch.bool)
    rg = torch.zeros(R, P, D_r)
    rm = torch.zeros(R, P, dtype=torch.bool)
    tl = torch.zeros(Tp, L, D_tl)
    tm = torch.zeros(Tp, L, dtype=torch.bool)

    # --- agents: 10 continuous + 3 type = D_a=13 ---
    if ex["ag_slot"].numel() > 0:
        ag_slot = ex["ag_slot"].long()
        ag_time = ex["ag_time"].long()
        ag_type = ex["ag_type"].long()
        ah[ag_slot, ag_time, 0:10] = ex["ag_feats"].float()
        ah[ag_slot, ag_time, 10:13] = F.one_hot(ag_type, 3).float()
        am[ag_slot, ag_time] = True

    # --- roadgraph: 5 continuous + 20 type = D_r=25 ---
    if ex["rg_chunk_idx"].numel() > 0:
        rg_chunk = ex["rg_chunk_idx"].long()
        rg_point = ex["rg_point_idx"].long()
        rg_type = ex["rg_type"].long()
        rg[rg_chunk, rg_point, 0:3] = ex["rg_xyz"].float()
        rg[rg_chunk, rg_point, 3:5] = ex["rg_dir"].float()
        rg[rg_chunk, rg_point, 5:25] = F.one_hot(rg_type, 20).float()
        rm[rg_chunk, rg_point] = True

    # --- traffic lights: 3 continuous + 9 state = D_tl=12 ---
    if ex["tl_slot"].numel() > 0:
        tl_slot = ex["tl_slot"].long()
        tl_time = ex["tl_time"].long()
        tl_state = ex["tl_state"].long()
        tl[tl_time, tl_slot, 0:3] = ex["tl_feats"].float()
        tl[tl_time, tl_slot, 3:12] = F.one_hot(tl_state, 9).float()
        tm[tl_time, tl_slot] = True

    return {
        "agent_history": ah.unsqueeze(0),      # [1, A, T_past, D_a] — leading dim = N (=1 marginal)
        "agent_mask": am.unsqueeze(0),
        "roadgraph": rg.unsqueeze(0),
        "roadgraph_mask": rm.unsqueeze(0),
        "traffic_lights": tl.unsqueeze(0),
        "tl_mask": tm.unsqueeze(0),
        "gt_tokens": ex["gt_tokens"].long().unsqueeze(0),
        "future_valid": unpack_future_valid(ex["future_valid"]).unsqueeze(0),
        "init_bin": ex.get(
            "init_bin", torch.zeros(2, dtype=torch.int16)
        ).to(torch.long).unsqueeze(0),
        "x0": torch.tensor([ex["x0"]], dtype=torch.float32),
        "y0": torch.tensor([ex["y0"]], dtype=torch.float32),
        "h0": torch.tensor([ex["h0"]], dtype=torch.float32),
        "scenario_id": ex["scenario_id"],
        "track_id": ex["track_id"],
    }
