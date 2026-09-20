"""数据集加载（契约见 docs/02 §4.1）：idx 解析、字符级语料分词、子集切片、DataLoader 构造与归一化。"""

from __future__ import annotations

import gzip
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.datasets import registry

DEFAULT_TRAIN_SIZE = 20000
DEFAULT_VAL_SIZE = 5000

# 字符级语料：验证集取末尾 10%，未知字符统一落到 <unk>（idx 0）
TEXT_VAL_FRACTION = 0.1
TEXT_UNK = "\x00"

_text_lock = threading.Lock()
_text_memo: dict[tuple[str, int, int, int], tuple[torch.Tensor, tuple[str, ...]]] = {}


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
    if spec.loader == "text_char":
        return _build_text_loaders(
            spec,
            batch_size=batch_size,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            shuffle=shuffle,
        )
    train_x, train_y = load_split(spec, "train", max(0, train_size))
    val_x, val_y = load_split(spec, "val", max(0, val_size))
    return _make_loaders(
        train_x, train_y, val_x, val_y, batch_size=batch_size, seed=seed, shuffle=shuffle
    )


def _make_loaders(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    extra_info: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
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
    info: dict[str, Any] = {
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "train_batches": len(train_loader),
        "steps_per_epoch": len(train_loader),
    }
    if extra_info:
        info.update(extra_info)
    return train_loader, val_loader, info


def load_text_corpus(spec: registry.DatasetSpec) -> tuple[torch.Tensor, tuple[str, ...]]:
    """读取并缓存字符级语料：返回 (token ids, idx → 字符)；idx 0 固定为 <unk>。"""
    if not spec.corpus:
        raise FileNotFoundError(f"数据集 {spec.id} 未声明语料文件")
    item = spec.file(spec.corpus)
    path = registry.locate(spec, item) if item is not None else None
    if path is None:
        raise FileNotFoundError(f"语料未找到：{spec.corpus}")
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, spec.vocab_cap)
    with _text_lock:
        cached = _text_memo.get(key)
    if cached is not None:
        return cached

    text = path.read_text(encoding="utf-8")
    counts = Counter(text)
    if spec.vocab_cap and len(counts) > spec.vocab_cap:
        keep = [char for char, _ in counts.most_common(spec.vocab_cap - 1)]
    else:
        keep = sorted(counts)
    itos = (TEXT_UNK, *keep)
    stoi = {char: index for index, char in enumerate(itos)}
    ids = torch.tensor([stoi.get(char, 0) for char in text], dtype=torch.int64)
    result = (ids, itos)
    with _text_lock:
        _text_memo[key] = result
    return result


def text_vocab_size(spec: registry.DatasetSpec) -> int:
    return len(load_text_corpus(spec)[1])


def _windows(ids: torch.Tensor, seq_len: int, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    """滑窗切分：每个窗口 x = ids[i:i+T]、y = ids[i+1:i+T+1]（stride = T，不重叠）。"""
    empty = torch.empty((0, seq_len), dtype=torch.int64)
    total = max(0, (ids.numel() - 1) // seq_len)
    if limit > 0:
        total = min(total, limit)
    if total <= 0:
        return empty, empty
    starts = torch.arange(total, dtype=torch.int64).unsqueeze(1) * seq_len
    offsets = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    return ids[starts + offsets], ids[starts + offsets + 1]


def _build_text_loaders(
    spec: registry.DatasetSpec,
    *,
    batch_size: int,
    train_size: int,
    val_size: int,
    seed: int,
    shuffle: bool,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    ids, itos = load_text_corpus(spec)
    seq_len = int(spec.input_shape[0])
    cut = int(ids.numel() * (1.0 - TEXT_VAL_FRACTION))
    train_x, train_y = _windows(ids[:cut], seq_len, train_size)
    val_x, val_y = _windows(ids[cut:], seq_len, val_size)
    return _make_loaders(
        train_x,
        train_y,
        val_x,
        val_y,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        extra_info={"seq_len": seq_len, "vocab_size": len(itos)},
    )


def normalize(batch: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """图像：uint8 → float32 并做 (x/255 - mean) / std；文本：词 id 保持 int64。"""
    if meta.get("loader") == "text_char":
        return batch.to(torch.long)
    x = batch.to(torch.float32).div_(255.0)
    mean = float(meta.get("mean") or 0.0)
    std = float(meta.get("std") or 1.0)
    if mean != 0.0 or std != 1.0:
        x.sub_(mean).div_(std)
    return x
