"""探针快照编码（契约见 docs/02 §6.2）：降采样 + 量化 + 统一 payload 容器。

纯函数层：输入 torch 张量，输出可直接 JSON 序列化的 payload（含 base64 数据）。
"""

from __future__ import annotations

import base64
import secrets
from typing import Any

import numpy as np
import torch

KINDS = ("feature_grid", "attention", "hidden", "histogram", "weights")

# kind 默认按节点类型推导（docs/02 §6.1）
DERIVED_KIND: dict[str, str] = {
    "Conv2d": "feature_grid",
    "MultiHeadAttention": "attention",
    "LSTM": "hidden",
}
DEFAULT_KIND = "histogram"
DEFAULT_MAX_ITEMS = 32

HISTOGRAM_BINS = 32
MAX_CHANNELS = 32
MAX_ATTENTION_HEADS = 8
MAX_SIDE = 32          # feature_grid / weights 单通道边长上限
MAX_HIDDEN_SIDE = 64   # hidden / attention 矩阵边长上限


class ProbeShapeError(Exception):
    """张量形状与 kind 不匹配：调用方停用该流并发 log.probe.disabled。"""

    def __init__(self, message_key: str, error_args: dict[str, Any] | None = None):
        super().__init__(message_key)
        self.message_key = message_key
        self.error_args = error_args or {}


def derived_kind(node_type: str) -> str:
    return DERIVED_KIND.get(node_type, DEFAULT_KIND)


def new_snapshot_id() -> str:
    return f"s_{secrets.token_hex(4)}"


# ---------------------------------------------------------------- 基础操作


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _quantize_uint8(array: np.ndarray, low: float, high: float) -> np.ndarray:
    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array.astype(np.float32) - low) / span, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _pool_to(array: np.ndarray, max_side: int) -> np.ndarray:
    """二维平均池化到 ≤ max_side × max_side（已足够小则原样返回）。"""
    rows, cols = int(array.shape[0]), int(array.shape[1])
    if rows <= max_side and cols <= max_side:
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        tensor[None, None], (min(rows, max_side), min(cols, max_side))
    )
    return pooled[0, 0].numpy()


def _stride_sample(array: np.ndarray, max_side: int) -> np.ndarray:
    """按步长抽取到 ≤ max_side × max_side（保留极值观感，不做平均）。"""
    rows, cols = int(array.shape[0]), int(array.shape[1])
    step = max(1, -(-rows // max_side), -(-cols // max_side))
    return array[::step, ::step]


def _payload(
    array: np.ndarray,
    *,
    dtype: str,
    layout: str,
    low: float,
    high: float,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": dtype,
        "layout": layout,
        "min": float(low),
        "max": float(high),
        "data_b64": _b64(array),
    }
    if meta:
        data["meta"] = meta
    return data


def _pick_sample(tensor: torch.Tensor, sample_index: int, *, batch_dim: bool) -> torch.Tensor:
    if batch_dim:
        if tensor.dim() == 0:
            raise ProbeShapeError("errors.probe.kindMismatch")
        index = min(max(sample_index, 0), int(tensor.shape[0]) - 1)
        return tensor[index]
    return tensor


# ---------------------------------------------------------------- 各 kind 编码


def encode_feature_grid(
    tensor: torch.Tensor, *, max_items: int = DEFAULT_MAX_ITEMS, sample_index: int = 0
) -> dict[str, Any]:
    """(B,C,H,W) → 前 min(C, max_items, 32) 通道，每通道池化到 ≤32×32。

    只接受四维张量：三维既可能是 (C,H,W)，也可能是文本模型里 Linear 的 (B,T,C)，
    无法区分，按契约对非线性层取 feature_grid 应停用该流（docs/02 §6.1）。
    """
    x = tensor.detach().to(torch.float32)
    if x.dim() != 4:
        raise ProbeShapeError("errors.probe.kindMismatch", {"shape": list(tensor.shape), "kind": "feature_grid"})
    sample = _pick_sample(x, sample_index, batch_dim=True)
    channels = min(int(sample.shape[0]), max(max_items, 1), MAX_CHANNELS)
    if channels < 1:
        raise ProbeShapeError("errors.probe.kindMismatch", {"shape": list(tensor.shape), "kind": "feature_grid"})
    maps = np.stack([_pool_to(sample[index].numpy(), MAX_SIDE) for index in range(channels)])
    return _payload(
        _quantize_uint8(maps, float(maps.min()), float(maps.max())),
        dtype="uint8",
        layout="CHW",
        low=float(maps.min()),
        high=float(maps.max()),
    )


def encode_attention(
    weights: torch.Tensor,
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    sample_index: int = 0,
    causal: bool = False,
) -> dict[str, Any]:
    """(B,heads,T,T) → 前 min(heads, max_items, 8) 头；值域固定 [0,1]。"""
    w = weights.detach().to(torch.float32)
    if w.dim() == 3:
        w = w.unsqueeze(0)
    if w.dim() != 4 or w.shape[-1] != w.shape[-2]:
        raise ProbeShapeError("errors.probe.kindMismatch", {"shape": list(weights.shape), "kind": "attention"})
    sample = _pick_sample(w, sample_index, batch_dim=True)
    heads = min(int(sample.shape[0]), max(max_items, 1), MAX_ATTENTION_HEADS)
    matrices = np.stack(
        [_stride_sample(sample[index].numpy(), MAX_HIDDEN_SIDE) for index in range(heads)]
    )
    return _payload(
        _quantize_uint8(matrices, 0.0, 1.0),
        dtype="uint8",
        layout="HTT",
        low=0.0,
        high=1.0,
        meta={"heads": heads, "tokens": int(matrices.shape[-1]), "causal": bool(causal)},
    )


def encode_hidden(
    tensor: torch.Tensor, *, max_items: int = DEFAULT_MAX_ITEMS, sample_index: int = 0
) -> dict[str, Any]:
    """(B,T,H) → 取一个样本的 (T,H) 热力图；二维输入按 (T,H) 解释。"""
    x = tensor.detach().to(torch.float32)
    if x.dim() == 3:
        x = _pick_sample(x, sample_index, batch_dim=True)
    elif x.dim() != 2:
        raise ProbeShapeError("errors.probe.kindMismatch", {"shape": list(tensor.shape), "kind": "hidden"})
    matrix = _stride_sample(x.numpy(), MAX_HIDDEN_SIDE)
    return _payload(
        _quantize_uint8(matrix, float(matrix.min()), float(matrix.max())),
        dtype="uint8",
        layout="TH",
        low=float(matrix.min()),
        high=float(matrix.max()),
    )


def encode_histogram(tensor: torch.Tensor, **_ignored: Any) -> dict[str, Any]:
    """32 个等宽 bin 的计数（uint32 不归一）；min == max 时范围取 ±0.5。"""
    values = tensor.detach().to(torch.float32).reshape(-1).numpy()
    if values.size == 0:
        raise ProbeShapeError("errors.probe.kindMismatch", {"shape": list(tensor.shape), "kind": "histogram"})
    low, high = float(values.min()), float(values.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    counts, _ = np.histogram(values, bins=HISTOGRAM_BINS, range=(low, high))
    return _payload(
        counts.astype("<u4"),
        dtype="uint32",
        layout="bins",
        low=low,
        high=high,
        meta={"bins": HISTOGRAM_BINS},
    )


def encode_weights(
    param: torch.Tensor, *, max_items: int = DEFAULT_MAX_ITEMS, sample_index: int = 0
) -> dict[str, Any]:
    """参数 (out, ...) → [min(out,32), 1, W]：每行展平后池化到 ≤32 宽。"""
    w = param.detach().to(torch.float32)
    if w.dim() < 2:
        raise ProbeShapeError("errors.probe.kindMismatch", {"shape": list(param.shape), "kind": "weights"})
    rows_total = int(w.shape[0])
    flat = w.reshape(rows_total, -1)
    rows = min(rows_total, max(max_items, 1), MAX_CHANNELS)
    maps = np.stack(
        [_pool_to(flat[index].reshape(1, -1).numpy(), MAX_SIDE).reshape(-1) for index in range(rows)]
    )
    stacked = maps[:, None, :]
    return _payload(
        _quantize_uint8(stacked, float(stacked.min()), float(stacked.max())),
        dtype="uint8",
        layout="CHW",
        low=float(stacked.min()),
        high=float(stacked.max()),
    )


def encode(kind: str, tensor: torch.Tensor, **options: Any) -> dict[str, Any]:
    if kind == "feature_grid":
        return encode_feature_grid(tensor, **options)
    if kind == "attention":
        return encode_attention(tensor, **options)
    if kind == "hidden":
        return encode_hidden(tensor, **options)
    if kind == "histogram":
        return encode_histogram(tensor, **options)
    if kind == "weights":
        return encode_weights(tensor, **options)
    raise ProbeShapeError("errors.probe.badKind", {"kind": kind})
