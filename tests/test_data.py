import math

import torch

from data.frame_norm import (
    ROI_BOX,
    chunk_roadgraph,
    normalize_frame,
    rank_context_agents,
    rank_tls_per_timestep,
    wrap_angle,
)
from data.shard_schema import densify, pack_future_valid, unpack_future_valid
from model.config import MotionLMConfig
from model.motion_tokenizer import MotionTokenizer


def test_normalize_frame_places_modeled_at_origin():
    torch.manual_seed(0)
    A_raw, Tp = 5, 11
    agents_xy = torch.randn(A_raw, Tp, 2) * 30.0
    agents_h = torch.randn(A_raw, Tp)
    rg_xyz = torch.randn(20, 2) * 30.0
    tl_xy = torch.randn(Tp, 3, 2) * 30.0

    modeled_idx = 2
    cti = Tp - 1
    x0, y0 = agents_xy[modeled_idx, cti].tolist()
    h0 = float(agents_h[modeled_idx, cti].item())

    a_xy_loc, a_h_loc, rg_loc, tl_loc, params = normalize_frame(
        agents_xy, agents_h, rg_xyz, tl_xy, x0, y0, h0
    )

    # modeled agent at cti sits at origin, heading 0
    assert a_xy_loc[modeled_idx, cti].abs().max() < 1e-4
    assert abs(float(a_h_loc[modeled_idx, cti].item())) < 1e-4
    # headings are wrapped to [-pi, pi]
    assert a_h_loc.abs().max() <= math.pi + 1e-4
    assert params == (x0, y0, h0)


def test_rank_context_agents_modeled_at_slot_0():
    torch.manual_seed(0)
    A_raw, Tp = 5, 11
    agents_xy = torch.randn(A_raw, Tp, 2) * 30.0
    kept = rank_context_agents(agents_xy, modeled_idx=2, A=4, current_time_index=Tp - 1)
    assert kept.numel() == 4
    assert int(kept[0].item()) == 2


def test_chunk_roadgraph_roi_clipping():
    torch.manual_seed(0)
    # 200 points across 5 features; half inside ROI, half outside.
    N = 200
    pts = torch.zeros(N, 2)
    pts[:100] = torch.rand(100, 2) * torch.tensor([80.0, 40.0]) + torch.tensor([-10.0, -20.0])   # inside
    pts[100:] = torch.rand(100, 2) * torch.tensor([200.0, 200.0]) + torch.tensor([200.0, 200.0])   # far out
    fid = torch.arange(N) // 40                                                                    # 5 features
    chunk_slot, point_slot, orig_idx = chunk_roadgraph(pts, fid, P=16, R=8, roi_box=ROI_BOX)
    # all kept points must be inside the ROI
    (xmin, xmax), (ymin, ymax) = ROI_BOX
    kept_xy = pts[orig_idx]
    assert (kept_xy[:, 0] >= xmin).all() and (kept_xy[:, 0] <= xmax).all()
    assert (kept_xy[:, 1] >= ymin).all() and (kept_xy[:, 1] <= ymax).all()
    # chunk count bounded by R
    assert chunk_slot.max() < 8 if chunk_slot.numel() > 0 else True
    # point slot bounded by P
    assert point_slot.max() < 16 if point_slot.numel() > 0 else True


def test_future_valid_pack_roundtrip():
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
    packed = pack_future_valid(mask)
    assert packed.dtype == torch.uint8 and packed.numel() == 2
    unpacked = unpack_future_valid(packed)
    assert torch.equal(mask, unpacked)


def test_densify_shape_contract():
    cfg = MotionLMConfig(
        A=4, T_past=3, D_a=13, R=4, P=4, D_r=25, L=2, D_tl=12, T=4,
        bins_per_coord=13, vocab_size=169, latents=4, n_enc=1, n_dec=1, N_max=1,
    )
    tokenizer = MotionTokenizer(cfg)
    # Build a minimal sparse example with 1 agent slot * T_past, 2 roadgraph points, 1 TL.
    from data.shard_schema import SparseExample

    ex: SparseExample = {
        "rg_xyz": torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float16),
        "rg_dir": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float16),
        "rg_type": torch.tensor([0, 1], dtype=torch.int8),
        "rg_chunk_idx": torch.tensor([0, 0], dtype=torch.int16),
        "rg_point_idx": torch.tensor([0, 1], dtype=torch.int8),
        "ag_feats": torch.zeros(3, 10, dtype=torch.float16),
        "ag_type": torch.tensor([0, 0, 0], dtype=torch.int8),
        "ag_slot": torch.tensor([0, 0, 0], dtype=torch.int8),
        "ag_time": torch.tensor([0, 1, 2], dtype=torch.int8),
        "tl_feats": torch.zeros(1, 3, dtype=torch.float16),
        "tl_state": torch.tensor([2], dtype=torch.int8),
        "tl_slot": torch.tensor([0], dtype=torch.int8),
        "tl_time": torch.tensor([2], dtype=torch.int8),
        "gt_tokens": tokenizer.encode(torch.randn(1, cfg.T, 2)).to(torch.int16).squeeze(0),
        "future_valid": pack_future_valid(torch.ones(16, dtype=torch.bool)),
        "x0": 1.0,
        "y0": 2.0,
        "h0": 0.3,
        "scenario_id": b"sid" + b"\0" * 13,
        "track_id": 42,
    }
    dense = densify(ex, cfg)
    assert dense["agent_history"].shape == (1, cfg.A, cfg.T_past, cfg.D_a)
    assert dense["roadgraph"].shape == (1, cfg.R, cfg.P, cfg.D_r)
    assert dense["traffic_lights"].shape == (1, cfg.T_past, cfg.L, cfg.D_tl)
    assert dense["gt_tokens"].shape == (1, cfg.T)
    assert dense["agent_mask"][0, 0, :3].all()    # modeled agent valid at first 3 times


if __name__ == "__main__":
    test_normalize_frame_places_modeled_at_origin()
    test_rank_context_agents_modeled_at_slot_0()
    test_chunk_roadgraph_roi_clipping()
    test_future_valid_pack_roundtrip()
    test_densify_shape_contract()
    print("Data tests passed.")
