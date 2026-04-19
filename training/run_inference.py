"""Load a trained MotionLM checkpoint, run Stage-4 inference on a shard sample,
and render the predicted modes as a GIF.

Usage::

    uv run python -m training.run_inference \\
        --checkpoint checkpoints/motionlm_2k.pt \\
        --shard /home/wentao/Downloads/shard_00000.pt.zst \\
        --sample-idx 0 --K 64 --out img/inference.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import make_loader
from model import MotionLM, MotionLMConfig
from model.utils import spline_to_10hz
from model.motion_tokenizer import MotionTokenizer
from viz.plot_inference import plot_inference_gif, plot_inference_scene


def _batch_from_tfrecord(tfrecord: Path, scenario_idx: int, cfg: MotionLMConfig) -> dict:
    """Build a shard-compatible batch from a WOMD tfrecord (skips future filter).

    Useful for the test split: its scenarios have no GT future, so the normal
    converter drops them, but inference only needs past+map.

    Picks the ``scenario_idx``-th scenario whose first ``tracks_to_predict`` slot
    is a VEHICLE. Returns a single-sample batch dict with leading dim ``[B=1, N=1]``
    matching the shard loader's output.
    """
    import math
    import torch

    from data.convert import (
        OBJ_VEHICLE, read_scenarios, parse_scenario,
        _build_agents_sparse, _build_roadgraph_sparse, _build_tls_sparse,
        _build_targets,
    )
    from data.shard_schema import densify

    tokenizer = MotionTokenizer(cfg)
    seen = 0
    for blob in read_scenarios(tfrecord):
        sc = parse_scenario(blob)
        if not sc["tracks_to_predict"]:
            continue
        tidx = sc["tracks_to_predict"][0]
        if tidx >= len(sc["tracks"]) or sc["tracks"][tidx][1] != OBJ_VEHICLE:
            continue
        cti = sc["cti"]
        st0 = sc["tracks"][tidx][2][cti]
        if not st0["valid"]:
            continue
        if seen < scenario_idx:
            seen += 1
            continue

        x0, y0, h0 = st0["x"], st0["y"], st0["h"]
        cos_h, sin_h = math.cos(h0), math.sin(h0)

        ex: dict = {}
        ex.update(_build_agents_sparse(sc["tracks"], tidx, cti, x0, y0, h0, cos_h, sin_h))
        ex.update(_build_roadgraph_sparse(sc["map_features"], x0, y0, h0, cos_h, sin_h))
        ex.update(_build_tls_sparse(sc["dynamic_map_states"], cti, x0, y0, cos_h, sin_h))
        targets, _ = _build_targets(sc["tracks"][tidx], cti, x0, y0, cos_h, sin_h, tokenizer)
        ex.update(targets)
        ex["x0"] = float(x0); ex["y0"] = float(y0); ex["h0"] = float(h0)
        ex["scenario_id"] = sc["scenario_id"][:16].ljust(16, b"\x00")
        ex["track_id"] = int(sc["tracks"][tidx][0])

        dense = densify(ex, cfg)
        # Add a leading batch dim: densify returns [1, ...] for N, we prepend B=1.
        return {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                for k, v in dense.items()}

    raise IndexError(f"no vehicle scenario at index {scenario_idx} in {tfrecord}")


def load_checkpoint(
    path: Path, device: str = "cpu"
) -> tuple[MotionLM, MotionLMConfig, dict]:
    """Rebuild model + config from a checkpoint written by ``training.train``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = MotionLMConfig(**ckpt["cfg"])
    model = MotionLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def _nth_batch(loader, n: int) -> dict:
    it = iter(loader)
    batch = None
    for _ in range(n + 1):
        batch = next(it)
    assert batch is not None
    return batch


def _gt_trajectory_agent_frame(
    batch: dict, cfg: MotionLMConfig
) -> tuple[torch.Tensor, int]:
    """Decode GT Verlet tokens → agent-frame 10 Hz waypoints + valid prefix length."""
    tokenizer = MotionTokenizer(cfg)
    gt_2hz = tokenizer.decode(batch["gt_tokens"], init_bin=batch.get("init_bin"))
    gt_10hz = spline_to_10hz(gt_2hz, cfg.T_future)                         # [B, N, Tf, 2]

    fv = batch["future_valid"][0, 0].tolist()                              # [T] bool
    valid_2hz = 0
    for bit in fv:
        if bit:
            valid_2hz += 1
        else:
            break
    if valid_2hz == 0:
        return gt_10hz, 0
    # map last valid 2 Hz idx to 10 Hz idx (align_corners=True in spline)
    last_10hz = int(round((valid_2hz - 1) * (cfg.T_future - 1) / (cfg.T - 1)))
    return gt_10hz, min(last_10hz + 1, cfg.T_future)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard", type=Path, help="Shard (.pt.zst) to sample from.")
    src.add_argument("--tfrecord", type=Path,
                    help="Raw WOMD tfrecord — use when shard is empty (e.g. test split).")
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="For --shard: Nth batch. For --tfrecord: Nth VEHICLE scenario.")
    ap.add_argument("--K", type=int, default=64,
                    help="Rollouts for Stage-4 sampling (smaller = faster demo).")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--M-modes", type=int, default=6)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("img/inference.gif"))
    ap.add_argument("--scene-png", type=Path, default=None,
                    help="Optional static PNG overlay of all predicted modes.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, ckpt = load_checkpoint(args.checkpoint, device=args.device)
    print(f"loaded checkpoint: step={ckpt.get('step')}  "
          f"train_loss={ckpt.get('loss'):.4f}  params={sum(p.numel() for p in model.parameters()):,}")

    if args.shard is not None:
        loader = make_loader([args.shard], cfg=cfg, batch_size=1, num_workers=0,
                             shuffle_buffer=1, pin_memory=False)
        batch = _nth_batch(loader, args.sample_idx)
    else:
        batch = _batch_from_tfrecord(args.tfrecord, args.sample_idx, cfg)
    # Drop x0/y0/h0 so Stage-4 keeps trajectories in agent frame (matches our plot).
    batch_infer = {k: v for k, v in batch.items() if k not in ("x0", "y0", "h0")}
    batch_dev = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch_infer.items()}

    infer_out = model.forward_infer(
        batch_dev, K=args.K, tau=args.tau, top_k=args.top_k, M_modes=args.M_modes,
    )
    probs = infer_out["probs"][0, 0].cpu().tolist()
    print(f"predicted {args.M_modes} modes:  probs = "
          + ", ".join(f"{p:.2f}" for p in probs))

    # Move tensors back to CPU for matplotlib rendering.
    batch_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    infer_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in infer_out.items()}

    # Ground-truth agent-frame trajectory for overlay (from gt_tokens + init_bin).
    gt_10hz, gt_valid = _gt_trajectory_agent_frame(batch_cpu, cfg)
    gt_np = gt_10hz[0, 0].numpy()                                          # [Tf, 2]
    print(f"GT valid horizon: {gt_valid}/{cfg.T_future} frames "
          f"({gt_valid / args.fps:.1f}s)")

    plot_inference_gif(batch_cpu, infer_cpu, args.out, fps=args.fps,
                       gt_trajectory=gt_np, gt_valid_len=gt_valid)
    print(f"wrote {args.out}")
    if args.scene_png is not None:
        plot_inference_scene(batch_cpu, infer_cpu, args.scene_png,
                             gt_trajectory=gt_np, gt_valid_len=gt_valid)
        print(f"wrote {args.scene_png}")


if __name__ == "__main__":
    main()
