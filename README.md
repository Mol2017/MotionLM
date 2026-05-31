# MotionLM: Multi-Agent Motion Forecasting as Language Modeling

A reproduction of **MotionLM** (Seff et al., ICCV 2023). Continuous trajectories
are tokenized into discrete motion tokens, and multi-agent forecasting is cast
as a language-modeling task: a Wayformer scene encoder conditions an
autoregressive decoder that predicts future motion tokens.

> Ari Seff, Brian Cera, Dian Chen, Mason Ng, Aurick Zhou, Nigamaa Nayakanti,
> Khaled S. Refaat, Rami Al-Rfou, Benjamin Sapp. ICCV 2023.
> [Paper](https://arxiv.org/abs/2309.16534)

## Results

A 10.8M-parameter model trained for 3 epochs (≈138k steps, batch=48) on the
WOMD `VEHICLE` marginal split. End-of-epoch eval on 100k val samples at K=64:

| val CE | minADE@8s | minFDE@8s | MR@8s |
|:--:|:--:|:--:|:--:|
| 1.70 | 1.96 m | 4.39 m | 0.39 |

Animations show 11 past + 80 future frames @ 10 Hz in agent frame (modeled agent
at origin, +x = heading). The 6 predicted modes are color-coded by confidence;
the dashed black line is ground truth.

| Working — highway, minFDE@8s ≈ 0.3 m | Working — urban, minFDE@8s ≈ 0.3 m |
|:--:|:--:|
| ![good1](img/example/good1.gif) | ![good2](img/example/good2.gif) |

| Failure — minFDE@8s ≈ 41 m | Failure — mode cloud drifts off GT |
|:--:|:--:|
| ![bad1](img/example/bad1.gif) | ![bad2](img/example/bad2.gif) |

## Components

- **Motion tokenizer** (`model/motion_tokenizer.py`) — Verlet-wrapped tokenizer
  that quantizes trajectory corrections into a 13×13 grid (169-token vocab).
- **Scene encoder** (`model/scene_encoder.py`) — Wayformer: per-modality embed
  (agents, roadgraph, traffic lights) → Perceiver → self-attention, producing
  scene memory.
- **Motion decoder** (`model/motion_decoder.py`) — autoregressive transformer
  that generates motion tokens conditioned on scene memory.
- **MotionLM** (`model/motionlm.py`) — end-to-end model with training
  (teacher-forced CE) and Stage-4 inference (K rollouts → decode → NMS).

Key dims live in `model/config.py`: `d_model=256`, `vocab_size=169`,
`bins_per_coord=13`, `max_corr=1.5`.

## Quick start

Requires Python 3.7+, PyTorch, TensorFlow. The project installs editable via `uv`.

```bash
uv run pytest tests/
```

### Data

Training reads `.pt.zst` shards (~115 MB each) produced from WOMD v1.3.1
scenario tfrecords.

**Pre-converted shards are on Hugging Face:**
[`wentao023/motionlm`](https://huggingface.co/datasets/wentao023/motionlm) —
download these to skip conversion.

To convert tfrecords yourself:

```bash
# single file
uv run python -m data.convert <input.tfrecord> <output.pt.zst>

# bulk: streams gs://, resume-safe, deletes tfrecords after conversion
SHARD_ROOT=$HOME/shards bash scripts/prepare_shards.sh training validation
```

### Train

```bash
# full run — 1 epoch ≈ 2.21M samples ≈ 82 min at batch=48 on an RTX 5080
uv run python -m training.train_multi_epoch \
    --train-shards /path/to/shards/training/train.*.pt.zst \
    --val-shards   /path/to/shards/validation/val.*.pt.zst \
    --out-dir runs/run_$(date +%Y%m%d_%H%M%S) \
    --epochs 6 --steps-per-epoch 46000 \
    --batch-size 48 --eval-samples 10000 --K 64 --amp
```

Writes `<out_dir>/checkpoints/epoch_NN.pt` + `<out_dir>/log.jsonl` (train/val
loss and minADE/minFDE/MR @ 3/5/8 s per epoch). Per-epoch checkpoints cap a
crash at one epoch.

Performance flags: `--amp` (bf16 autocast, ~10% faster than fp32),
`--grad-ckpt` (needed only for batch ≥ 64), `--compile` (neutral at this size).
SDPA/FlashAttention is on by default. Best throughput on a 16 GB GPU is
batch=48 (~453 samples/s) without checkpointing.

### Evaluate

```bash
uv run python -m training.evaluate \
    --checkpoint checkpoints/my_model.pt \
    --shards /home/wentao/shards/validation/val.*.pt.zst \
    --max-samples 1000000 --batch-size 8 --num-workers 4 --K 64
```

Reports val CE + token top-1 accuracy + minADE/minFDE/MissRate @ 3/5/8 s
(WOMD 2025 marginal spec). Throughput is ~25 samples/s at K=64; use `--K 64`
for dev loops and `--K 512` (paper value) only to reproduce paper numbers.
Drop `--max-samples` and `--shards val.0of150.pt.zst` for a ~20 s sanity check.

For reference, the paper's single-replica numbers after 600k steps at K=512:
minADE@8s ≈ 1.03 m, minFDE@8s ≈ 2.39 m.

## Diagnostics

**Training loss** (3-epoch, 138k steps, constant LR 2e-4) — the ~1.69 plateau
motivated the warmup + cosine-decay schedule:

![Training loss curve](img/long_loss.png)

**Verlet tokenizer reconstruction** — the quantization floor for the 13×13 grid;
a finer grid would lower the minADE floor:

![Tokenizer reconstruction error](img/motion_tokenizer_reconstruction_error.png)

## Roadmap

Throughput is at the hardware ceiling (~458 samples/s at 11M params on a 16 GB
RTX 5080); the real headroom is in prediction quality.

- **Training recipe** — LR warmup + cosine decay (landed via `--lr-schedule
  cosine`), longer training, neighbor-aware soft Verlet targets, scheduled
  sampling, horizon-weighted CE, label smoothing / EMA.
- **Inference** — K 64 → 512, temperature / top-k sweeps.
- **Tokenizer / data** — finer Verlet grid (13×13 → 17×17), longer past context.
- **Missing paper features** — masked-LM pretraining, joint N=2 interactive
  training (unlocks Overlap Rate), intent bucketing for Soft-mAP, heading-frame
  miss-rate projection for strict WOMD parity.

## Citation

```bibtex
@article{seff2023motionlm,
  title={MotionLM: Multi-Agent Motion Forecasting as Language Modeling},
  author={Seff, Ari and Cera, Brian and Chen, Dian and Ng, Mason and Zhou, Aurick and Nayakanti, Nigamaa and Refaat, Khaled S. and Al-Rfou, Rami and Sapp, Benjamin},
  journal={arXiv preprint arXiv:2309.16534},
  year={2023}
}
```
