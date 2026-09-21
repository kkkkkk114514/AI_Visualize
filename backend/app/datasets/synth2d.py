"""二维合成数据集的程序生成（契约见 docs/02 §13.3）：固定种子，逐位可复现。"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

POINTS_TRAIN = 300
POINTS_VAL = 200
BOX = 0.85


@dataclass(frozen=True)
class SynthParams:
    noise: float
    seed: int


PARAMS: dict[str, SynthParams] = {
    "moons": SynthParams(0.18, 7),
    "circles": SynthParams(0.12, 11),
    "blobs": SynthParams(0.40, 13),
    "spiral": SynthParams(0.06, 17),
}

_lock = threading.Lock()
_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _moons(half: int) -> np.ndarray:
    t = np.linspace(0.0, math.pi, half)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    return np.concatenate([upper, lower], axis=0)


def _circles(half: int, rng: np.random.Generator) -> np.ndarray:
    outer = rng.uniform(0.0, 2.0 * math.pi, half)
    inner = rng.uniform(0.0, 2.0 * math.pi, half)
    return np.concatenate(
        [
            np.stack([np.cos(outer), np.sin(outer)], axis=1),
            0.45 * np.stack([np.cos(inner), np.sin(inner)], axis=1),
        ],
        axis=0,
    )


def _blobs(half: int) -> np.ndarray:
    return np.concatenate(
        [np.tile([-0.5, -0.5], (half, 1)), np.tile([0.5, 0.5], (half, 1))], axis=0
    )


def _spiral(half: int) -> np.ndarray:
    # 起点半径拉开一点：两条臂从中心起步时相距仅 2r，噪声一压就混在一起
    radius = np.linspace(0.16, 1.0, half) ** 0.85
    arms = []
    for arm in range(2):
        theta = radius * 3.5 * math.pi + arm * math.pi
        arms.append(np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1))
    return np.concatenate(arms, axis=0)


def _finalize(raw: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """先按原坐标系叠加噪声，再居中缩放到最远点 ~BOX：噪声与几何等比缩放。"""
    noisy = raw + noise * rng.standard_normal(raw.shape)
    centered = noisy - 0.5 * (noisy.min(axis=0) + noisy.max(axis=0))
    scale = BOX / float(np.abs(centered).max())
    return (centered * scale).astype(np.float32)


def _generate(dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    params = PARAMS[dataset_id]
    rng = np.random.default_rng(params.seed)
    half = (POINTS_TRAIN + POINTS_VAL) // 2
    if dataset_id == "moons":
        raw = _moons(half)
    elif dataset_id == "circles":
        raw = _circles(half, rng)
    elif dataset_id == "blobs":
        raw = _blobs(half)
    else:
        raw = _spiral(half)
    points = _finalize(raw, params.noise, rng)
    # 每类各自打乱后逐点交错：切分后类别均衡，且训练 / 验证都覆盖整个空间
    ordered = np.empty_like(points)
    ordered[0::2] = points[:half][rng.permutation(half)]
    ordered[1::2] = points[half:][rng.permutation(half)]
    labels = np.zeros(ordered.shape[0], dtype=np.int64)
    labels[1::2] = 1
    return ordered, labels


def data(dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    """全部点集：(N,2) float32 坐标 + (N,) int64 标签；同一 id 结果恒定。"""
    with _lock:
        cached = _cache.get(dataset_id)
    if cached is not None:
        return cached
    result = _generate(dataset_id)
    with _lock:
        _cache[dataset_id] = result
    return result


def make_points(dataset_id: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    points, labels = data(dataset_id)
    if split == "val":
        return points[POINTS_TRAIN:], labels[POINTS_TRAIN:]
    return points[:POINTS_TRAIN], labels[:POINTS_TRAIN]


def split_arrays(
    dataset_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(train_x, train_y, val_x, val_y)，供训练侧直接使用。"""
    train_x, train_y = make_points(dataset_id, "train")
    val_x, val_y = make_points(dataset_id, "val")
    return train_x, train_y, val_x, val_y


def bounds(dataset_id: str, margin: float = 0.05) -> tuple[float, float, float, float]:
    """训练集包围盒外扩 margin（决策边界网格用，docs/02 §13.5）。"""
    points, _ = data(dataset_id)
    low = points.min(axis=0)
    high = points.max(axis=0)
    pad = float(np.max(high - low)) * margin
    return (
        float(low[0] - pad),
        float(high[0] + pad),
        float(low[1] - pad),
        float(high[1] + pad),
    )


def points_payload(dataset_id: str, split: str) -> dict[str, Any]:
    points, labels = make_points(dataset_id, split)
    return {
        "split": split,
        "points": np.round(points.astype(np.float64), 4).tolist(),
        "labels": labels.tolist(),
        "count": int(points.shape[0]),
    }
