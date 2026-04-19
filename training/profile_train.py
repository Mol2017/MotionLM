"""Quick training profiler: break down per-step time into data / fwd / bwd / opt.

Runs N warmup + N measured steps and prints per-phase mean/std in ms, plus
throughput and a param/FLOP-adjacent sanity check.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from data import make_loader
from model import MotionLM, MotionLMConfig


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--compile", dest="compile_model", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MotionLMConfig(grad_checkpoint=args.grad_ckpt)
    model = MotionLM(cfg).to(device)
    if args.compile_model:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    amp_ctx = (
        torch.amp.autocast(device, dtype=torch.bfloat16)
        if args.amp else torch.amp.autocast(device, enabled=False)
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M   device={device}   "
          f"amp(bf16)={args.amp}   compile={args.compile_model}   grad_ckpt={args.grad_ckpt}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    loader = make_loader(
        args.shards, cfg=cfg, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    it = iter(loader)

    data_ms, h2d_ms, fwd_ms, bwd_ms, opt_ms, step_ms = [], [], [], [], [], []

    model.train()
    total = args.warmup + args.steps
    for i in range(total):
        record = i >= args.warmup
        t_step = time.perf_counter()

        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t_data = time.perf_counter() - t0

        t0 = time.perf_counter()
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        _sync(device)
        t_h2d = time.perf_counter() - t0

        opt.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        with amp_ctx:
            loss, _ = model.forward_train(batch)
        _sync(device)
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        t_bwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        _sync(device)
        t_opt = time.perf_counter() - t0

        t_step_total = time.perf_counter() - t_step

        if record:
            data_ms.append(t_data * 1000)
            h2d_ms.append(t_h2d * 1000)
            fwd_ms.append(t_fwd * 1000)
            bwd_ms.append(t_bwd * 1000)
            opt_ms.append(t_opt * 1000)
            step_ms.append(t_step_total * 1000)

        tag = "warm" if not record else "meas"
        print(f"[{tag}] step {i:3d}  loss={loss.item():.3f}  "
              f"data={t_data*1000:6.1f}  h2d={t_h2d*1000:5.1f}  "
              f"fwd={t_fwd*1000:6.1f}  bwd={t_bwd*1000:6.1f}  "
              f"opt={t_opt*1000:5.1f}  total={t_step_total*1000:6.1f} ms")

    def stat(xs):
        return f"{statistics.mean(xs):7.1f} ± {statistics.pstdev(xs):5.1f}"

    print("\n=== measured-step statistics (ms) ===")
    print(f"data   : {stat(data_ms)}")
    print(f"h2d    : {stat(h2d_ms)}")
    print(f"fwd    : {stat(fwd_ms)}")
    print(f"bwd    : {stat(bwd_ms)}")
    print(f"opt    : {stat(opt_ms)}")
    print(f"TOTAL  : {stat(step_ms)}")
    mean_step = statistics.mean(step_ms) / 1000
    print(f"\nthroughput: {1/mean_step:.2f} it/s  "
          f"(batch={args.batch_size}, samples/s={args.batch_size/mean_step:.1f})")
    if device == "cuda":
        print(f"peak gpu mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
