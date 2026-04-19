"""Minimal PyTorch train loop for MotionLM.

Reads from the shard-based dataloader (``data.dataset.make_loader``) and runs
teacher-forcing CE training. Gradient clipping and AMP are optional.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data import make_loader
from model import MotionLM, MotionLMConfig


def train(
    shard_paths: list[Path],
    cfg: MotionLMConfig,
    steps: int = 1000,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 4,
    log_every: int = 25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    amp: bool = False,
    compile_model: bool = False,
    save_path: Path | None = None,
    save_every: int | None = None,
) -> dict[str, float]:
    model = MotionLM(cfg).to(device)
    if compile_model:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = make_loader(shard_paths, cfg=cfg, batch_size=batch_size, num_workers=num_workers)
    loader_iter = iter(loader)

    # bf16 autocast — same speed as fp16 on Ampere+, no GradScaler needed
    # (bf16 matches fp32's exponent range so CE loss can't overflow).
    amp_ctx = (
        torch.amp.autocast(device, dtype=torch.bfloat16)
        if amp else torch.amp.autocast(device, enabled=False)
    )

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            loss, _ = model.forward_train(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        running += float(loss.item())
        if step % log_every == 0:
            avg = running / log_every
            running = 0.0
            dt = time.time() - t0
            print(f"step {step:>6d}  loss={avg:.4f}  ({step / dt:.2f} it/s)")

        if save_every and save_path is not None and step % save_every == 0:
            _save_checkpoint(save_path, _unwrap(model), cfg, step, loss.item())

    final_loss = avg if steps >= log_every else float(loss.item())
    if save_path is not None:
        _save_checkpoint(save_path, _unwrap(model), cfg, steps, final_loss)
    return {"final_loss": final_loss}


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying nn.Module, stripping torch.compile's wrapper."""
    return getattr(model, "_orig_mod", model)


def _save_checkpoint(
    path: Path, model: torch.nn.Module, cfg: MotionLMConfig,
    step: int, loss: float,
) -> None:
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cfg": asdict(cfg), "step": step, "loss": loss},
        path,
    )
    print(f"saved checkpoint → {path}  (step={step}, loss={loss:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="wrap model in torch.compile (pays ~1 min warmup for 20–40% backward speedup)")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="recompute decoder-block activations in bwd (fits larger batches; ~30% compute overhead per block)")
    parser.add_argument("--save-path", type=Path, default=None,
                        help="If given, save a checkpoint at the end (and at --save-every).")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Checkpoint every N steps in addition to the final save.")
    args = parser.parse_args()

    cfg = MotionLMConfig(grad_checkpoint=args.grad_ckpt)
    train(
        args.shards,
        cfg,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        amp=args.amp,
        compile_model=args.compile_model,
        save_path=args.save_path,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
