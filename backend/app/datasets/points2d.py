"""二维点集的分派层（契约见 docs/02 §13.3）：`synth2d`（程序生成）与 `csv2d`（上传）统一出口。

刻意不 import torch：ML 子进程只依赖本模块取点集与包围盒（docs/02 §13.4「子进程不 import torch」）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets import csv2d, registry, synth2d

POINT_LOADERS = ("synth2d", "csv2d")


def split_arrays(spec: registry.DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spec.loader == "csv2d":
        return csv2d.split_arrays(spec)
    train_x, train_y = synth2d.make_points(spec.id, "train")
    val_x, val_y = synth2d.make_points(spec.id, "val")
    return train_x, train_y, val_x, val_y


def bounds(spec: registry.DatasetSpec, margin: float = 0.05) -> tuple[float, float, float, float]:
    if spec.loader == "csv2d":
        return csv2d.bounds(spec, margin)
    return synth2d.bounds(spec.id, margin)


def points_payload(spec: registry.DatasetSpec, split: str) -> dict[str, Any]:
    if spec.loader == "csv2d":
        return csv2d.points_payload(spec, split)
    return synth2d.points_payload(spec.id, split)
