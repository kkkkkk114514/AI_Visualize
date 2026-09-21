"""上传点集（`csv2d`，契约见 docs/02 §7.5）：3 列 x,y,label 的解析、切分与缓存。"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from app.datasets import registry

TRAIN_FRACTION = 0.8
SPLIT_SEED = 42
MIN_ROWS = 20

_lock = threading.Lock()
_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}


def parse_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    """解析 3 列 `x,y,label`；首行非数字自动当表头跳过；label ∈ {0,1}、坐标必须是有限值。"""
    xs: list[float] = []
    ys: list[float] = []
    labels: list[int] = []
    first = True
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"line {lineno}: expected 3 columns, got {len(parts)}")
        try:
            x, y, label = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            if first:
                first = False
                continue
            raise ValueError(f"line {lineno}: not numeric") from None
        first = False
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"line {lineno}: non-finite coordinate")
        if label not in (0.0, 1.0):
            raise ValueError(f"line {lineno}: label must be 0 or 1, got {parts[2]}")
        xs.append(x)
        ys.append(y)
        labels.append(int(label))
    if len(xs) < MIN_ROWS:
        raise ValueError(f"need >= {MIN_ROWS} rows, got {len(xs)}")
    points = np.stack(
        [np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)], axis=1
    )
    return points, np.asarray(labels, dtype=np.int64)


def load_points(spec: registry.DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    item = spec.file(spec.corpus or "") if spec.corpus else None
    path = registry.locate(spec, item) if item is not None else None
    if path is None:
        raise FileNotFoundError(f"数据集未缓存：{spec.id}")
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    points, labels = parse_text(path.read_text(encoding="utf-8-sig"))
    with _lock:
        _cache[key] = (points, labels)
    return points, labels


def split(
    points: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """seed 42 打乱后 80/20 切分（同一文件逐位可复现）。"""
    order = np.random.default_rng(SPLIT_SEED).permutation(points.shape[0])
    n_train = min(max(1, int(round(points.shape[0] * TRAIN_FRACTION))), points.shape[0] - 1)
    train, val = order[:n_train], order[n_train:]
    return points[train], labels[train], points[val], labels[val]


def split_arrays(spec: registry.DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points, labels = load_points(spec)
    return split(points, labels)


def bounds(spec: registry.DatasetSpec, margin: float = 0.05) -> tuple[float, float, float, float]:
    """训练集包围盒外扩 margin（决策边界网格用，docs/02 §13.5）。"""
    train_x, _, _, _ = split_arrays(spec)
    low = train_x.min(axis=0)
    high = train_x.max(axis=0)
    pad = float(np.max(high - low)) * margin
    return (
        float(low[0] - pad),
        float(high[0] + pad),
        float(low[1] - pad),
        float(high[1] + pad),
    )


def points_payload(spec: registry.DatasetSpec, split_key: str) -> dict[str, Any]:
    train_x, train_y, val_x, val_y = split_arrays(spec)
    points, labels = (val_x, val_y) if split_key == "val" else (train_x, train_y)
    return {
        "split": split_key,
        "points": np.round(points.astype(np.float64), 4).tolist(),
        "labels": labels.tolist(),
        "count": int(points.shape[0]),
    }
