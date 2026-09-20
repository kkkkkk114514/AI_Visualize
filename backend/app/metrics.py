"""指标常量与 LTTB 降采样（契约见 docs/02 §5.4 / §7.2）。"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

METRIC_NAMES: tuple[str, ...] = (
    "loss",
    "acc",
    "val_loss",
    "val_acc",
    "lr",
    "grad_norm",
    "throughput",
    "vram_mb",
)

# 曲线默认勾选的主指标（其余在 UI 里可选显示）
PRIMARY_METRICS: tuple[str, ...] = ("loss", "acc")


def lttb(xs: Sequence[float], ys: Sequence[float], max_points: int) -> tuple[list[float], list[float]]:
    """Largest-Triangle-Three-Buckets：保形降采样，首尾点必留、下标严格递增。"""
    n = len(xs)
    if max_points < 3 or n <= max_points:
        return list(xs), list(ys)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    sampled = [0]
    every = (n - 2) / (max_points - 2)
    for i in range(max_points - 2):
        avg_start = min(int(math.floor((i + 1) * every)) + 1, n - 1)
        avg_end = max(min(int(math.floor((i + 2) * every)) + 1, n), avg_start + 1)
        avg_x = float(x[avg_start:avg_end].mean())
        avg_y = float(y[avg_start:avg_end].mean())

        off_start = min(int(math.floor(i * every)) + 1, n - 1)
        off_end = max(min(int(math.floor((i + 1) * every)) + 1, n - 1), off_start + 1)
        span = slice(off_start, off_end)
        anchor = sampled[-1]
        areas = np.abs(
            (x[span] - x[anchor]) * (avg_y - y[anchor]) - (avg_x - x[anchor]) * (y[span] - y[anchor])
        )
        sampled.append(min(off_start + int(np.argmax(areas)), n - 2))
    sampled.append(n - 1)

    return [xs[i] for i in sampled], [ys[i] for i in sampled]


def downsample_series(
    steps: Sequence[int], values: Sequence[float], max_points: int
) -> tuple[list[int], list[float]]:
    if max_points <= 0 or len(steps) <= max_points:
        return list(steps), list(values)
    out_steps, out_values = lttb(list(steps), list(values), max_points)
    return [int(s) for s in out_steps], [float(v) for v in out_values]


def series_payload(series: dict[str, dict[str, list[float]]], max_points: int) -> dict[str, Any]:
    """把 {name: {steps, values}} 逐条降采样成前端直绘的格式。"""
    out: dict[str, Any] = {}
    for name in METRIC_NAMES:
        entry = series.get(name)
        if not entry or not entry["steps"]:
            continue
        steps, values = downsample_series(entry["steps"], entry["values"], max_points)
        out[name] = {"steps": steps, "values": values, "total": len(entry["steps"])}
    return out


def values_row(values: dict[str, Any]) -> Iterable[tuple[str, float]]:
    for name in METRIC_NAMES:
        value = values.get(name)
        if value is None:
            continue
        try:
            yield name, float(value)
        except (TypeError, ValueError):
            continue
