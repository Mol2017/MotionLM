"""Evaluate a MotionLM checkpoint on a validation set.

Computes:
  * Validation CE loss + perplexity (teacher-forced, same objective as training)
  * Token top-1 accuracy (teacher-forced)
  * WOMD interaction-prediction metrics (marginal, per-agent):
      - minADE @ {3, 5, 8} s
      - minFDE @ {3, 5, 8} s
      - Miss Rate @ {3, 5, 8} s  (speed-scaled thresholds, WOMD 2025 spec)

Two WOMD metrics are intentionally **not** computed here — see
`design.md` §Training performance notes for reasons:
  * mAP / Soft mAP — needs per-agent intent bucketing (straight / left /
    right / u-turn / stationary) that we haven't built. Stub documented below.
  * Overlap Rate — joint multi-agent collision metric; our shards are
    marginal (N=1, VEHICLE only). Revisit when the interactive split is wired.

The miss-rate lat/long projection is done in the **initial agent frame** (at
t=0 the modeled agent's heading is +x by construction of Stage 0). This is
exact for straight trajectories and an approximation for turns — WOMD's
official impl projects at each evaluation timestep's heading. Flagged below.

Usage::

    uv run python -m training.evaluate \\
        --checkpoint checkpoints/motionlm_2k.pt \\
        --shards /home/wentao/shards/validation/val.*.pt.zst \\
        --max-samples 1000 --batch-size 4 --K 64
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from data import make_loader
from model import MotionLM, MotionLMConfig
from model.motion_tokenizer import MotionTokenizer
from model.utils import spline_to_10hz


# --- WOMD 2025 miss-rate thresholds (lateral, longitudinal) in metres ---
# Source: waymo.com/open/challenges/2025/interaction-prediction
MR_THRESHOLDS = {
    3: (1.0, 2.0),
    5: (1.8, 3.6),
    8: (3.0, 6.0),
}


def _speed_scale(speed_mps: torch.Tensor) -> torch.Tensor:
    """WOMD speed scaling for miss-rate thresholds.

    <1.4 m/s → 0.5×, >11 m/s → 1.0×, linear in between. Returns a tensor
    broadcastable against ``speed_mps``.
    """
    s = speed_mps.clamp(1.4, 11.0)
    return 0.5 + 0.5 * (s - 1.4) / (11.0 - 1.4)


@dataclass
class Accum:
    """Running sums for per-sample metrics."""
    ce_sum: float = 0.0
    ce_count: int = 0              # token count (for weighted mean)
    tok_correct: int = 0
    tok_total: int = 0
    # Per-horizon accumulators keyed by seconds (3, 5, 8).
    ade_sum: dict = field(default_factory=lambda: {3: 0.0, 5: 0.0, 8: 0.0})
    fde_sum: dict = field(default_factory=lambda: {3: 0.0, 5: 0.0, 8: 0.0})
    miss_sum: dict = field(default_factory=lambda: {3: 0, 5: 0, 8: 0})
    horizon_count: dict = field(default_factory=lambda: {3: 0, 5: 0, 8: 0})


def _ce_and_token_acc(
    logits: torch.Tensor, gt_tokens: torch.Tensor, future_valid: torch.Tensor
) -> tuple[float, int, int, int]:
    """Returns (ce_sum_over_valid_tokens, n_valid_tokens, n_correct, n_total_tokens)."""
    B, N, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_tgt = gt_tokens.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_tgt, reduction="none").view(B, N, T)
    mask = future_valid.to(ce.dtype)
    ce_sum = float((ce * mask).sum().item())
    n_valid = int(mask.sum().item())

    pred = flat_logits.argmax(dim=-1).view(B, N, T)
    correct = ((pred == gt_tokens) & future_valid).sum().item()
    total = int(future_valid.sum().item())
    return ce_sum, n_valid, int(correct), total


def _gt_10hz_agent_frame(batch: dict, cfg: MotionLMConfig) -> torch.Tensor:
    """Decode shard GT tokens → [B, N, T_future, 2] agent-frame @ 10 Hz."""
    tokenizer = MotionTokenizer(cfg)
    gt_2hz = tokenizer.decode(batch["gt_tokens"], init_bin=batch.get("init_bin"))
    return spline_to_10hz(gt_2hz, cfg.T_future)


def _valid_10hz_steps(future_valid: torch.Tensor, T_future: int, T_2hz: int) -> torch.Tensor:
    """For each (B, N) sample, count the leading contiguous valid 2Hz tokens and
    map to a 10Hz frame count (align_corners=True convention; matches
    ``spline_to_10hz``).
    """
    # Count leading True values along last dim (future_valid: [B, N, T])
    # If sample is all valid, valid_2hz = T.
    valid_2hz = future_valid.int().cumprod(dim=-1).sum(dim=-1)  # [B, N]
    # Map last valid 2Hz idx to 10Hz idx (align_corners=True).
    # valid_2hz==0 → 0 frames. valid_2hz==T → T_future frames.
    step_ratio = (T_future - 1) / max(T_2hz - 1, 1)
    last_10hz = ((valid_2hz - 1).clamp(min=0).float() * step_ratio).round().long() + 1
    return torch.where(valid_2hz > 0, last_10hz, torch.zeros_like(last_10hz))


def _agent_speed_at_t0(batch: dict) -> torch.Tensor:
    """Modeled-agent speed at t=current from shard features. Returns [B, N]."""
    # agent_history: [B, N, A, T_past, D_a]; slot 0 = modeled agent, features
    # layout: 0:3 xyz, 3:5 sin/cos h, 5:7 vxy, 7:10 LWH, 10:13 type one-hot.
    ah = batch["agent_history"]
    T_past = ah.size(-2)
    t0 = T_past - 1
    vx = ah[..., 0, t0, 5]
    vy = ah[..., 0, t0, 6]
    return torch.sqrt(vx * vx + vy * vy)


def _update_horizon_metrics(
    acc: Accum,
    pred_world: torch.Tensor,   # [B, N, M_modes, Tf, 2] — agent-frame (x=fwd, y=lat)
    gt: torch.Tensor,           # [B, N, Tf, 2] agent-frame
    valid_10hz: torch.Tensor,   # [B, N] int
    speed_t0: torch.Tensor,     # [B, N]
    fps: int = 10,
) -> None:
    B, N, M, Tf, _ = pred_world.shape
    for seconds, (lat_thr, long_thr) in MR_THRESHOLDS.items():
        t_last_exclusive = seconds * fps                     # e.g. 30 frames for 3s
        # Only count samples whose GT is valid through `t_last_exclusive`.
        ok = valid_10hz >= t_last_exclusive                  # [B, N]
        if not ok.any():
            continue
        # per-timestep L2 over horizon
        pred_h = pred_world[..., :t_last_exclusive, :]       # [B, N, M, t, 2]
        gt_h = gt[..., :t_last_exclusive, :].unsqueeze(2)    # [B, N, 1, t, 2]
        err = pred_h - gt_h
        l2 = torch.linalg.norm(err, dim=-1)                  # [B, N, M, t]

        ade_per_mode = l2.mean(dim=-1)                       # [B, N, M]
        fde_per_mode = l2[..., -1]                           # [B, N, M]
        min_ade, best_mode = ade_per_mode.min(dim=-1)
        min_fde = fde_per_mode.min(dim=-1).values

        # Miss rate: any mode within (lat_thr*scale, long_thr*scale)?
        scale = _speed_scale(speed_t0).unsqueeze(-1)         # [B, N, 1]
        lat_ok = err[..., 1].abs() <= (lat_thr * scale).unsqueeze(-1)   # [B, N, M, t]
        long_ok = err[..., 0].abs() <= (long_thr * scale).unsqueeze(-1)
        all_ok_per_mode = (lat_ok & long_ok).all(dim=-1)     # [B, N, M] all-t within box
        any_mode_ok = all_ok_per_mode.any(dim=-1)            # [B, N]

        for b in range(B):
            for n in range(N):
                if not bool(ok[b, n]):
                    continue
                acc.horizon_count[seconds] += 1
                acc.ade_sum[seconds] += float(min_ade[b, n])
                acc.fde_sum[seconds] += float(min_fde[b, n])
                if not bool(any_mode_ok[b, n]):
                    acc.miss_sum[seconds] += 1


def evaluate(
    model: MotionLM,
    cfg: MotionLMConfig,
    shard_paths: list[Path],
    *,
    max_samples: int = 1000,
    batch_size: int = 4,
    num_workers: int = 2,
    K: int = 64,
    M_modes: int = 6,
    tau: float = 1.0,
    top_k: int | None = None,
    device: str = "cuda",
    log_every: int | None = None,
) -> dict:
    # Rate-limit progress prints. Default: ~20 lines total regardless of size.
    log_every = log_every if log_every is not None else max(1, max_samples // 20)
    loader = make_loader(
        shard_paths,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=1024,
        pin_memory=(device.startswith("cuda")),
    )

    acc = Accum()
    seen = 0
    t_start = time.time()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            B = batch["gt_tokens"].size(0)
            if seen + B > max_samples:
                # Trim final batch to respect max_samples exactly.
                keep = max_samples - seen
                batch = {k: (v[:keep] if isinstance(v, torch.Tensor) else v[:keep])
                         for k, v in batch.items()}
                B = keep

            dev_batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            # 1) Teacher-forced CE + token accuracy
            loss, logits = model.forward_train(dev_batch)
            ce_sum, n_valid, n_correct, n_total = _ce_and_token_acc(
                logits.float(), dev_batch["gt_tokens"], dev_batch["future_valid"]
            )
            acc.ce_sum += ce_sum
            acc.ce_count += n_valid
            acc.tok_correct += n_correct
            acc.tok_total += n_total

            # 2) Stage-4 rollout in agent frame (drop x0/y0/h0 so inverse_frame is skipped).
            infer_batch = {k: v for k, v in dev_batch.items() if k not in ("x0", "y0", "h0")}
            out = model.forward_infer(
                infer_batch, K=K, tau=tau, top_k=top_k, M_modes=M_modes,
            )
            pred = out["trajectories_world"].float().cpu()   # agent-frame here
            gt_10hz = _gt_10hz_agent_frame(batch, cfg).float()  # cpu
            valid_10hz = _valid_10hz_steps(batch["future_valid"], cfg.T_future, cfg.T)
            speed_t0 = _agent_speed_at_t0(batch)

            _update_horizon_metrics(acc, pred, gt_10hz, valid_10hz, speed_t0)

            seen += B
            elapsed = time.time() - t_start
            done = seen >= max_samples
            if seen // log_every > (seen - B) // log_every or done:
                print(f"[{seen:>6d}/{max_samples}]  "
                      f"running ce={acc.ce_sum/max(acc.ce_count,1):.4f}  "
                      f"tok_acc={acc.tok_correct/max(acc.tok_total,1):.3f}  "
                      f"{seen/elapsed:.1f} samples/s",
                      flush=True)

            if done:
                break

    # --- finalize ---
    ce = acc.ce_sum / max(acc.ce_count, 1)
    summary = {
        "samples_evaluated": seen,
        "val_ce_loss": ce,
        "val_perplexity": math.exp(ce),
        "token_top1_acc": acc.tok_correct / max(acc.tok_total, 1),
        "elapsed_seconds": time.time() - t_start,
    }
    for s in (3, 5, 8):
        n = max(acc.horizon_count[s], 1)
        summary[f"minADE@{s}s"] = acc.ade_sum[s] / n
        summary[f"minFDE@{s}s"] = acc.fde_sum[s] / n
        summary[f"miss_rate@{s}s"] = acc.miss_sum[s] / n
        summary[f"n_samples@{s}s"] = acc.horizon_count[s]
    return summary


def _load_checkpoint(path: Path, device: str) -> tuple[MotionLM, MotionLMConfig, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = MotionLMConfig(**ckpt["cfg"])
    model = MotionLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--K", type=int, default=64,
                    help="Stage-4 rollout count (paper uses 512; 64 is a fast estimate).")
    ap.add_argument("--M-modes", type=int, default=6)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=None,
                    help="Print progress every N samples; default ~20 lines total.")
    args = ap.parse_args()

    model, cfg, ckpt = _load_checkpoint(args.checkpoint, args.device)
    print(f"loaded checkpoint: step={ckpt.get('step')}  "
          f"train_loss={ckpt.get('loss'):.4f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    summary = evaluate(
        model, cfg, args.shards,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        K=args.K, M_modes=args.M_modes,
        tau=args.tau, top_k=args.top_k,
        device=args.device,
        log_every=args.log_every,
    )

    print("\n=== validation summary ===")
    print(f"samples evaluated : {summary['samples_evaluated']}")
    print(f"elapsed           : {summary['elapsed_seconds']:.1f} s")
    print(f"val CE loss       : {summary['val_ce_loss']:.4f}   "
          f"(perplexity {summary['val_perplexity']:.2f})")
    print(f"token top-1 acc   : {summary['token_top1_acc']:.3f}")
    print(f"\n{'horizon':>8s}  {'minADE':>8s}  {'minFDE':>8s}  {'MissRate':>10s}  {'N':>6s}")
    for s in (3, 5, 8):
        print(f"{s}s       "
              f"{summary[f'minADE@{s}s']:>8.3f}  "
              f"{summary[f'minFDE@{s}s']:>8.3f}  "
              f"{summary[f'miss_rate@{s}s']:>10.3f}  "
              f"{summary[f'n_samples@{s}s']:>6d}")
    print("\n(skipped: Soft mAP needs intent bucketing; overlap rate needs N=2)")


if __name__ == "__main__":
    main()
