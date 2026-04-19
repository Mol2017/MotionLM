"""Train MotionLM across multiple epochs with end-of-epoch evaluation.

After each epoch:
  * saves checkpoint to ``<out_dir>/epoch_<NN>.pt``
  * runs ``training.evaluate.evaluate`` on a val subset
  * appends a JSONL line with train + val stats to ``<out_dir>/log.jsonl``

Designed to run unattended for several hours. Per-epoch checkpoints mean a
crash costs at most one epoch of progress.

Example::

    uv run python -m training.train_multi_epoch \\
        --train-shards /home/wentao/shards/training/train.*.pt.zst \\
        --val-shards /home/wentao/shards/validation/val.*.pt.zst \\
        --out-dir runs/run_$(date +%Y%m%d_%H%M%S) \\
        --epochs 6 --steps-per-epoch 46000 \\
        --batch-size 48 --eval-samples 10000 --K 64
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from data import make_loader
from model import MotionLM, MotionLMConfig
from training.evaluate import evaluate


def _save_ckpt(path: Path, model: torch.nn.Module, cfg: MotionLMConfig,
               step: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cfg": asdict(cfg), "step": step, "loss": loss},
        path,
    )


def make_warmup_cosine_scheduler(
    opt: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_frac: float,
    lr_min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over the first ``warmup_frac`` of steps, then cosine decay
    from peak LR down to ``lr_min_ratio × peak`` over the remainder."""
    warmup_steps = max(1, int(warmup_frac * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    *,
    steps: int,
    device: str,
    amp: bool,
    grad_clip: float,
    log_every: int,
    epoch_idx: int,
    global_step: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float, int, list[float]]:
    """Run ``steps`` training steps. Returns (mean_loss, last_loss, new_global_step, per_step_losses)."""
    model.train()
    amp_ctx = (torch.amp.autocast(device, dtype=torch.bfloat16)
               if amp else torch.amp.autocast(device, enabled=False))

    losses: list[float] = []
    it = iter(loader)
    t0 = time.time()
    running = 0.0
    last_report = t0
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            loss, _ = model.forward_train(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        fval = float(loss.item())
        losses.append(fval)
        running += fval
        global_step += 1
        if step % log_every == 0:
            now = time.time()
            avg = running / log_every
            running = 0.0
            its = log_every / (now - last_report)
            last_report = now
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch_idx}  step {step:>6d}/{steps}  "
                  f"loss={avg:.4f}  lr={cur_lr:.2e}  ({its:.2f} it/s, "
                  f"elapsed {now - t0:.0f}s)", flush=True)

    mean_loss = sum(losses) / max(len(losses), 1)
    return mean_loss, losses[-1], global_step, losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", type=Path, nargs="+", required=True)
    ap.add_argument("--val-shards", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=46000,
                    help="Approximates one full pass of the training set at batch=48.")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999,
                    help="AdamW beta2. Paper uses 0.999; 0.95 is common for LM plateau exit.")
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine",
                    help="cosine: linear warmup then cosine decay; constant: disable scheduling.")
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="Fraction of total steps for linear warmup (cosine only).")
    ap.add_argument("--lr-min-ratio", type=float, default=0.0,
                    help="Cosine decays from peak LR down to lr_min_ratio × peak.")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="bf16 autocast (default on).")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    # Eval knobs
    ap.add_argument("--eval-samples", type=int, default=10000,
                    help="Samples per end-of-epoch eval (~7 min at K=64).")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--eval-num-workers", type=int, default=2)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--M-modes", type=int, default=6)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = MotionLMConfig()
    device = args.device
    model = MotionLM(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=(device.startswith("cuda")),
    )

    total_steps = args.epochs * args.steps_per_epoch
    if args.lr_schedule == "cosine":
        scheduler = make_warmup_cosine_scheduler(
            opt,
            total_steps=total_steps,
            warmup_frac=args.warmup_frac,
            lr_min_ratio=args.lr_min_ratio,
        )
        warmup_steps = max(1, int(args.warmup_frac * total_steps))
        sched_desc = (f"cosine (warmup {warmup_steps} steps = {args.warmup_frac:.1%}, "
                      f"decay to {args.lr_min_ratio:.1%} of peak)")
    else:
        scheduler = None
        sched_desc = "constant"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M   device={device}   amp(bf16)={args.amp}")
    print(f"epochs={args.epochs}  steps/epoch={args.steps_per_epoch}  batch={args.batch_size}  "
          f"eval_samples={args.eval_samples}  K={args.K}")
    print(f"lr={args.lr:.1e}  schedule={sched_desc}")
    print(f"writing to: {out_dir}")

    # One long-lived loader; IterableDataset re-shuffles shards per __iter__().
    train_loader = make_loader(
        args.train_shards, cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    # Persist run-level metadata once.
    meta = {
        "type": "run_meta",
        "started": time.time(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                 if k not in ("train_shards", "val_shards")},
        "n_train_shards": len(args.train_shards),
        "n_val_shards": len(args.val_shards),
        "params_M": n_params / 1e6,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(meta) + "\n")

    global_step = 0
    run_t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n=========================  epoch {epoch}/{args.epochs - 1}  "
              f"(elapsed {(time.time() - run_t0) / 60:.1f} min)  "
              f"=========================", flush=True)
        ep_t0 = time.time()
        mean_loss, last_loss, global_step, _ = train_one_epoch(
            model, train_loader, opt,
            steps=args.steps_per_epoch, device=device, amp=args.amp,
            grad_clip=args.grad_clip, log_every=args.log_every,
            epoch_idx=epoch, global_step=global_step,
            scheduler=scheduler,
        )
        train_secs = time.time() - ep_t0

        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        _save_ckpt(ckpt_path, model, cfg, global_step, last_loss)
        print(f"  saved {ckpt_path}  (mean_train_loss={mean_loss:.4f}, "
              f"last={last_loss:.4f}, train_secs={train_secs:.0f})", flush=True)

        # --- end-of-epoch eval ---
        print(f"  evaluating on {args.eval_samples} val samples at K={args.K}...",
              flush=True)
        eval_t0 = time.time()
        eval_summary = evaluate(
            model, cfg, args.val_shards,
            max_samples=args.eval_samples,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            K=args.K, M_modes=args.M_modes, tau=args.tau,
            device=device,
        )
        eval_secs = time.time() - eval_t0
        print(
            f"  val CE={eval_summary['val_ce_loss']:.4f}  "
            f"tok_acc={eval_summary['token_top1_acc']:.3f}  "
            f"minADE@8s={eval_summary['minADE@8s']:.2f}  "
            f"minFDE@8s={eval_summary['minFDE@8s']:.2f}  "
            f"MR@8s={eval_summary['miss_rate@8s']:.3f}  "
            f"(eval_secs={eval_secs:.0f})",
            flush=True,
        )

        entry = {
            "type": "epoch",
            "epoch": epoch,
            "global_step": global_step,
            "train_secs": train_secs,
            "eval_secs": eval_secs,
            "mean_train_loss": mean_loss,
            "last_train_loss": last_loss,
            "checkpoint": str(ckpt_path),
            "elapsed_total_s": time.time() - run_t0,
            **eval_summary,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    print(f"\ndone. total wall: {(time.time() - run_t0) / 3600:.2f} h  "
          f"log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
