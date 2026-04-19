"""End-to-end smoke test with a synthetic shard.

Generates a handful of fake sparse examples matching the real schema, runs them
through ``save_shard_zstd`` → ``MotionLMShardDataset`` → ``MotionLM.forward_train``
to prove the full pipeline holds together without requiring the WOMD tfrecord
dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from data import make_loader
from data.convert import save_shard_zstd
from data.shard_schema import SparseExample, pack_future_valid
from model import MotionLM, MotionLMConfig
from model.motion_tokenizer import MotionTokenizer


def _fake_example(cfg: MotionLMConfig, tokenizer: MotionTokenizer, seed: int) -> SparseExample:
    g = torch.Generator().manual_seed(seed)
    # Modeled agent's 10 Hz future for 8 s
    fut_10hz = torch.cumsum(torch.randn(80, 2, generator=g) * 0.3, dim=0) + torch.linspace(0, 40, 80).unsqueeze(-1) * torch.tensor([1.0, 0.0])
    fut_2hz = fut_10hz[4::5]
    tokens = tokenizer.encode(fut_2hz.unsqueeze(0)).squeeze(0)

    # Minimal agent / roadgraph / TL footprint
    n_ag = cfg.A * cfg.T_past // 2
    ag_slot = torch.randint(0, cfg.A, (n_ag,), generator=g, dtype=torch.int8)
    ag_time = torch.randint(0, cfg.T_past, (n_ag,), generator=g, dtype=torch.int8)
    ag_feats = torch.randn(n_ag, 10, generator=g).to(torch.float16)
    ag_type = torch.zeros(n_ag, dtype=torch.int8)  # all VEHICLE

    n_rg = 200
    rg_chunk = torch.randint(0, cfg.R, (n_rg,), generator=g, dtype=torch.int16)
    rg_point = torch.randint(0, cfg.P, (n_rg,), generator=g, dtype=torch.int8)
    rg_xyz = torch.randn(n_rg, 3, generator=g).to(torch.float16)
    rg_dir = torch.randn(n_rg, 2, generator=g).to(torch.float16)
    rg_type = torch.randint(0, 20, (n_rg,), generator=g, dtype=torch.int8)

    n_tl = 4
    tl_slot = torch.randint(0, cfg.L, (n_tl,), generator=g, dtype=torch.int8)
    tl_time = torch.randint(0, cfg.T_past, (n_tl,), generator=g, dtype=torch.int8)
    tl_feats = torch.randn(n_tl, 3, generator=g).to(torch.float16)
    tl_state = torch.randint(0, 9, (n_tl,), generator=g, dtype=torch.int8)

    return SparseExample(
        rg_xyz=rg_xyz,
        rg_dir=rg_dir,
        rg_type=rg_type,
        rg_chunk_idx=rg_chunk,
        rg_point_idx=rg_point,
        ag_feats=ag_feats,
        ag_type=ag_type,
        ag_slot=ag_slot,
        ag_time=ag_time,
        tl_feats=tl_feats,
        tl_state=tl_state,
        tl_slot=tl_slot,
        tl_time=tl_time,
        gt_tokens=tokens.to(torch.int16),
        future_valid=pack_future_valid(torch.ones(16, dtype=torch.bool)),
        x0=float(torch.randn(1, generator=g).item() * 10),
        y0=float(torch.randn(1, generator=g).item() * 10),
        h0=float(torch.randn(1, generator=g).item() * 0.5),
        scenario_id=b"synthetic" + b"\0" * 7,
        track_id=seed,
    )


def main() -> None:
    cfg = MotionLMConfig(
        d=64, d_ff=128, heads=4,
        A=16, T_past=11, R=32, P=16, L=4,
        T=16, T_future=80,
        latents=32, n_enc=2, n_dec=2, N_max=1,
        K=4, M_modes=3,
    )
    tokenizer = MotionTokenizer(cfg)

    with tempfile.TemporaryDirectory() as td:
        shard_path = Path(td) / "shard_00000.pt.zst"
        examples = [_fake_example(cfg, tokenizer, seed=i) for i in range(48)]
        save_shard_zstd({"version": 1, "count": len(examples), "examples": examples}, shard_path)
        sz = shard_path.stat().st_size
        print(f"shard: {sz / 1024:.1f} KB for {len(examples)} examples "
              f"({sz / len(examples):.0f} B/ex)")

        loader = make_loader([shard_path], cfg=cfg, batch_size=8, num_workers=0, pin_memory=False)
        it = iter(loader)
        batch = next(it)

        print("batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)}")

        # Train for a few steps
        model = MotionLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        model.train()
        losses = []
        for step in range(10):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            opt.zero_grad()
            loss, _ = model.forward_train(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"10 train steps: losses[0]={losses[0]:.3f} → losses[-1]={losses[-1]:.3f}")

        # One inference pass
        model.eval()
        out = model.forward_infer(batch)
        print(f"infer outputs: traj={tuple(out['trajectories_world'].shape)}  probs={tuple(out['probs'].shape)}")

    print("End-to-end synthetic pipeline test passed.")


if __name__ == "__main__":
    main()
