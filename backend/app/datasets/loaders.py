"""数据集加载（契约见 docs/02 §4.1）：idx 解析、子集切片、DataLoader 构造与归一化。"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.datasets import registry

DEFAULT_TRAIN_SIZE = 20000
DEFAULT_VAL_SIZE = 5000


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise ValueError(f"idx 文件损坏：{path.name}")
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    expected = 4 + 4 * ndim + int(np.prod(dims))
    if expected != len(raw):
        raise ValueError(f"idx 长度不符：{path.name} expected={expected} got={len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims).copy()


def load_split(spec: registry.DatasetSpec, split_key: str, limit: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    split = spec.split(split_key)
    if split is None:
        raise KeyError(split_key)
    images_file = spec.file(split.images)
    labels_file = spec.file(split.labels)
    if images_file is None or labels_file is None:
        raise KeyError(split.images)
    images_path = registry.locate(spec, images_file)
    labels_path = registry.locate(spec, labels_file)
    if images_path is None or labels_path is None:
        raise FileNotFoundError(f"数据集未缓存：{spec.id}/{split_key}")

    images = read_idx(images_path)
    labels = read_idx(labels_path)
    if limit > 0:
        images = images[:limit]
        labels = labels[:limit]
    # (N, H, W) → (N, *input_shape)（MNIST 为 (N,1,28,28)）
    shape = tuple(spec.input_shape)
    if len(shape) == 3 and shape[0] == 1 and images.ndim == 3:
        images = images.reshape(len(images), 1, *images.shape[1:])
    x = torch.from_numpy(np.ascontiguousarray(images))
    y = torch.from_numpy(np.ascontiguousarray(labels)).to(torch.int64)
    return x, y


def build_loaders(
    spec: registry.DatasetSpec,
    *,
    batch_size: int,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_x, train_y = load_split(spec, "train", max(0, train_size))
    val_x, val_y = load_split(spec, "val", max(0, val_size))
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    info = {
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "train_batches": len(train_loader),
        "steps_per_epoch": len(train_loader),
    }
    return train_loader, val_loader, info


def normalize(batch: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """uint8 → float32 并做 (x/255 - mean) / std（图像数据集）。"""
    x = batch.to(torch.float32).div_(255.0)
    mean = float(meta.get("mean") or 0.0)
    std = float(meta.get("std") or 1.0)
    if mean != 0.0 or std != 1.0:
        x.sub_(mean).div_(std)
    return x
