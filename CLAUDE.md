# MotionLM — Claude context

Reproduction of **MotionLM: Multi-Agent Motion Forecasting as Language Modeling**
(Seff et al., ICCV 2023). Verlet-tokenized trajectories, Wayformer scene encoder,
joint multi-agent decoder, K-rollout Stage-4 inference with NMS aggregation.

## Design doc

`design.md` — the canonical spec. Stages 0–4, exact tensor shapes, vocab
math, encoder/decoder layer counts. Code is written to match this doc; when the
two disagree, the doc wins and the code is the bug.

## Repo layout

```
model/              # network + tokenizer
  config.py         # MotionLMConfig dataclass (single source of truth for dims)
  motion_tokenizer.py   # Verlet Δ-bin tokenizer (169-token vocab, pure math)
  scene_encoder.py  # Wayformer: per-modality embed → Perceiver(192) → self-attn×6
  motion_decoder.py # (SA staircase → CA routed → FFN) × 4 joint decoder
  motionlm.py       # orchestrator: forward_train (CE) + forward_infer (K rollouts → decode → spline → inverse_frame → NMS)
  utils.py          # shared helpers (e.g. spline_to_10hz)
data/               # data pipeline
  frame_norm.py     # Stage 0: agent-centric frame + ROI clip
  shard_schema.py   # sparse dict ↔ dense tensor (densify)
  convert.py        # offline WOMD tfrecord → shard_NNNNN.pt.zst
  dataset.py        # IterableDataset + make_loader
training/
  train.py          # train() library + CLI
  evaluate.py       # val CE + minADE/minFDE/MissRate @ 3/5/8s (WOMD 2025)
  run_inference.py  # load checkpoint → Stage-4 → render GIF/PNG
viz/
  plot_scenario.py      # scene viz (from shard or tfrecord); PNG + animated GIF
  plot_inference.py     # M-mode overlay + ground-truth dashed line
  plot_tokenizer_error.py   # Verlet round-trip error sweep
tests/              # pytest (editable install; no sys.path hacks)
checkpoints/        # saved .pt files
img/                # rendered outputs (GIFs + PNGs)
```

## Commands

All commands assume `uv` is set up; the project is installed editable.

```bash
# tests
uv run pytest tests/

# convert one WOMD tfrecord → shard
uv run python -m data.convert <input.tfrecord> <output.pt.zst>

# train (writes a checkpoint at end when --save-path is given)
uv run python -m training.train \
    --shards path/to/shard_00000.pt.zst \
    --steps 2000 --batch-size 16 --amp \
    --save-path checkpoints/motionlm_2k.pt --save-every 500

# multi-epoch train + per-epoch eval, runs unattended
# writes <out_dir>/checkpoints/epoch_NN.pt + <out_dir>/log.jsonl
uv run python -m training.train_multi_epoch \
    --train-shards /home/wentao/shards/training/train.*.pt.zst \
    --val-shards /home/wentao/shards/validation/val.*.pt.zst \
    --out-dir runs/run_$(date +%Y%m%d_%H%M%S) \
    --epochs 6 --steps-per-epoch 46000 \
    --batch-size 48 --eval-samples 10000 --K 64 --amp

# smoke variant (one shard each side, ~3 min):
uv run python -m training.train_multi_epoch \
    --train-shards /home/wentao/shards/training/train.0of1000.pt.zst \
    --val-shards /home/wentao/shards/validation/val.0of150.pt.zst \
    --out-dir runs/smoke --epochs 2 --steps-per-epoch 500 \
    --batch-size 48 --eval-samples 64 --K 32 --amp

# inference on one shard sample → GIF + optional static PNG (with GT overlay)
uv run python -m training.run_inference \
    --checkpoint checkpoints/motionlm_2k.pt \
    --shard path/to/shard_00000.pt.zst \
    --sample-idx 0 --K 64 --M-modes 6 \
    --out img/inference.gif --scene-png img/inference.png

# evaluate — val CE + minADE/minFDE/MissRate @ 3/5/8s (quick: ~20s for 512 samples)
uv run python -m training.evaluate \
    --checkpoint checkpoints/motionlm_2k.pt \
    --shards /home/wentao/shards/validation/val.0of150.pt.zst \
    --max-samples 512 --batch-size 8 --K 64

# scenario viz (from shard; optional tfrecord overlay)
uv run python -m viz.plot_scenario --shard path/to/shard_00000.pt.zst --gif

# tokenizer reconstruction-error sweep
uv run python -m viz.plot_tokenizer_error --n 64
```

## Output images

`img/` holds all rendered artifacts:

- `inference.gif` / `inference.png` — trained model's M predicted modes animated
  over 11 past + 80 future frames (@ 10 Hz), with a dashed black ground-truth
  overlay. Coordinates are agent-frame (modeled agent at origin, +x = heading).
- `scenario_whole_shard.gif` / `.png`, `scenario_frame_shard.png` — dense
  roadgraph + agents + TL for one shard sample, from `plot_scenario`.
- `motion_tokenizer_reconstruction_error.png` — Verlet round-trip error curve.
- `train_loss.png` — training loss over steps.

## Conventions

- **Frame**: everything inside the model is agent-centric (Stage 0 output).
  World frame is reconstructed in Stage 4 via `inverse_frame(x0, y0, h0)`. If
  you want predictions in agent frame (e.g. for visualization alongside the
  shard's ROI-clipped scene), drop `x0/y0/h0` from the batch before
  `forward_infer` — that's what `training/run_inference.py` does.
- **Shapes**: `A=64` agents, `T_past=11`, `T=16` (2 Hz tokens), `T_future=80`
  (10 Hz), `R=256` roadgraph chunks × `P=128` points, `L=16` TL slots,
  vocab=169 + BOS (170 total), `N` = modeled agents per pass (1 or 2).
- **Imports**: all packages are installed editable via hatchling; tests import
  `from model.X import ...` directly — no sys.path manipulation.
- **CE baseline**: random-init loss is `log(169) ≈ 5.13` nats. A 2k-step run on
  one shard converges to ~1.6 at batch=16 on an RTX 5080.
