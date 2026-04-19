"""Streaming ``IterableDataset`` + ``DataLoader`` wiring (design doc §"Dataloader design").

Usage::

    from data import MotionLMShardDataset, make_loader
    loader = make_loader(shard_paths, batch_size=32, num_workers=8, cfg=cfg)
    for batch in loader:
        loss, _ = model.forward_train(batch)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset

from data.convert import load_shard_zstd
from data.shard_schema import densify
from model.config import MotionLMConfig


class MotionLMShardDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: Sequence[Path],
        cfg: MotionLMConfig,
        shuffle_buffer: int = 8192,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shards = list(shard_paths)
        self.cfg = cfg
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_shards = shuffle_shards

    def _shards_for_worker(self) -> list[Path]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return list(self.shards)
        return list(self.shards[info.id :: info.num_workers])

    def __iter__(self) -> Iterator[dict]:
        shards = self._shards_for_worker()
        if self.shuffle_shards:
            random.shuffle(shards)

        # Buffer holds SPARSE examples (~65 KB each) — densify on yield so
        # the dense ~3.4 MB roadgraph tensor never sits in the shuffle buffer.
        buf: list[dict] = []
        for shard_path in shards:
            shard = load_shard_zstd(shard_path)
            examples = shard["examples"]
            random.shuffle(examples)
            for ex in examples:
                buf.append(ex)
                if len(buf) >= self.shuffle_buffer:
                    random.shuffle(buf)
                    drain = len(buf) // 2
                    for _ in range(drain):
                        yield densify(buf.pop(), self.cfg)
        random.shuffle(buf)
        while buf:
            yield densify(buf.pop(), self.cfg)


def _collate(batch: list[dict]) -> dict:
    """Default collate: stack tensors on dim 0; pass-through metadata."""
    keys = batch[0].keys()
    out: dict = {}
    for k in keys:
        first = batch[0][k]
        if isinstance(first, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def make_loader(
    shard_paths: Sequence[Path],
    cfg: MotionLMConfig,
    batch_size: int = 32,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    shuffle_buffer: int = 8192,
    pin_memory: bool = True,
) -> DataLoader:
    ds = MotionLMShardDataset(shard_paths, cfg=cfg, shuffle_buffer=shuffle_buffer)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
