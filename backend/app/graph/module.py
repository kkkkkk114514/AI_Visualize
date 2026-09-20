"""通用图执行器：IR → 动态构建的 nn.Module（拓扑执行，不做源码生成）。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from app.graph.ir import Edge, GraphIR, Node, fan_in_map, topological_order


class NodeExecutionError(Exception):
    """节点级执行错误：定位到具体节点，携带 i18n key 与参数。"""

    def __init__(
        self,
        node_id: str,
        code: str,
        message_key: str,
        error_args: dict[str, Any] | None = None,
        detail: str = "",
    ):
        super().__init__(detail or message_key)
        self.node_id = node_id
        self.code = code
        self.message_key = message_key
        # 注意：不要用 self.args —— BaseException.args 会把任意可迭代对象转成元组
        self.error_args = error_args or {}
        self.detail = detail


def _shape_of(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def _dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", type(value).__name__))


class _NodeBase(nn.Module):
    """节点包装层：子模块在首次前向拿到输入形状时才构建（D3：形状来自真实前向）。"""

    def __init__(self, node: Node, params: dict[str, Any]):
        super().__init__()
        self.node_id = node.id
        self.params = params
        self.inner: nn.Module | None = None

    def _require_tensor(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise NodeExecutionError(
                self.node_id,
                "bad_input",
                "errors.node.notTensor",
                {"got": type(value).__name__},
            )
        return value

    def _require_float(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise NodeExecutionError(
                self.node_id,
                "bad_dtype",
                "errors.node.expectFloat",
                {"got": _dtype_name(x)},
            )
        return x

    def _require_ndim(self, x: torch.Tensor, ndim: int) -> torch.Tensor:
        if x.dim() != ndim:
            raise NodeExecutionError(
                self.node_id,
                "bad_ndim",
                "errors.node.expectNdim",
                {"expected": ndim, "got": x.dim(), "shape": list(x.shape)},
            )
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class InputNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


class LinearNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        if self.inner is None:
            self.inner = nn.Linear(int(x.shape[-1]), self.params["out_features"], self.params["bias"])
        return self.inner(x)


class Conv2dNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, 4)
        if self.inner is None:
            groups = self.params["groups"]
            in_channels = int(x.shape[1])
            out_channels = self.params["out_channels"]
            for name, value in (("out_channels", out_channels), ("in_channels", in_channels)):
                if value % groups != 0:
                    raise NodeExecutionError(
                        self.node_id,
                        "bad_groups",
                        "errors.node.groupsDivisible",
                        {"param": name, "value": value, "groups": groups},
                    )
            self.inner = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.params["kernel_size"],
                stride=self.params["stride"],
                padding=self.params["padding"],
                dilation=self.params["dilation"],
                groups=groups,
            )
        return self.inner(x)


class _PoolNode(_NodeBase):
    pool_cls: type[nn.Module]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, 4)
        if self.inner is None:
            stride = self.params["stride"] or None
            self.inner = self.pool_cls(
                kernel_size=self.params["kernel_size"],
                stride=stride,
                padding=self.params["padding"],
            )
        return self.inner(x)


class MaxPool2dNode(_PoolNode):
    pool_cls = nn.MaxPool2d


class AvgPool2dNode(_PoolNode):
    pool_cls = nn.AvgPool2d


class AdaptiveAvgPool2dNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, 4)
        if self.inner is None:
            self.inner = nn.AdaptiveAvgPool2d(self.params["output_size"])
        return self.inner(x)


class FlattenNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_tensor(x)
        if self.inner is None:
            self.inner = nn.Flatten(self.params["start_dim"])
        return self.inner(x)


class DropoutNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_tensor(x)
        if self.inner is None:
            self.inner = nn.Dropout(self.params["p"])
        return self.inner(x)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class ActivationNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_tensor(x)
        if self.inner is None:
            self.inner = _ACTIVATIONS[self.params["name"]]()
        return self.inner(x)


class BatchNormNode(_NodeBase):
    expected_ndim = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, self.expected_ndim)
        if self.inner is None:
            self.inner = self.bn_cls(int(x.shape[1]), eps=self.params["eps"], momentum=self.params["momentum"])
        return self.inner(x)


class BatchNorm2dNode(BatchNormNode):
    expected_ndim = 4
    bn_cls = nn.BatchNorm2d


class BatchNorm1dNode(BatchNormNode):
    expected_ndim = 3
    bn_cls = nn.BatchNorm1d


class LayerNormNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        if self.inner is None:
            size = self.params["normalized_shape"]
            normalized_shape = [int(size)] if size else [int(x.shape[-1])]
            self.inner = nn.LayerNorm(normalized_shape)
        return self.inner(x)


class EmbeddingNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_tensor(x)
        if x.is_floating_point():
            raise NodeExecutionError(
                self.node_id,
                "bad_dtype",
                "errors.node.expectInt",
                {"got": _dtype_name(x)},
            )
        if self.inner is None:
            self.inner = nn.Embedding(self.params["num_embeddings"], self.params["embedding_dim"])
        if x.numel() > 0:
            max_index = int(x.max())
            if max_index >= self.params["num_embeddings"]:
                raise NodeExecutionError(
                    self.node_id,
                    "index_out_of_range",
                    "errors.node.embeddingRange",
                    {"max_index": max_index, "num_embeddings": self.params["num_embeddings"]},
                )
        return self.inner(x)


class PositionalEncodingNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, 3)
        seq_len, dim = int(x.shape[1]), int(x.shape[2])
        max_len = self.params["max_len"]
        if seq_len > max_len:
            raise NodeExecutionError(
                self.node_id,
                "seq_too_long",
                "errors.node.seqTooLong",
                {"length": seq_len, "max_len": max_len},
            )
        if self.params["mode"] == "learned":
            if self.inner is None:
                self.inner = nn.Embedding(max_len, dim)
            positions = torch.arange(seq_len, device=x.device)
            return x + self.inner(positions).unsqueeze(0)
        if self.inner is None:
            self.inner = _SinusoidalEncoding(max_len, dim)
        return x + self.inner(positions=None)[:seq_len].unsqueeze(0)  # type: ignore[call-arg]


class _SinusoidalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor | None = None) -> torch.Tensor:
        return self.pe


class LSTMNode(_NodeBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, 3)
        if self.inner is None:
            self.inner = nn.LSTM(
                input_size=int(x.shape[2]),
                hidden_size=self.params["hidden_size"],
                num_layers=self.params["num_layers"],
                bidirectional=self.params["bidirectional"],
                batch_first=True,
            )
        out, _ = self.inner(x)
        if self.params["return_sequences"]:
            return out
        return out[:, -1, :]


class MultiHeadAttentionNode(_NodeBase):
    def __init__(self, node: Node, params: dict[str, Any]):
        super().__init__(node, params)
        # 采样步由探针置位：本步返回逐头注意力权重（供 attention 快照读取）
        self.capture_attention = False
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_float(self._require_tensor(x))
        self._require_ndim(x, 3)
        dim, heads = int(x.shape[2]), self.params["num_heads"]
        if dim % heads != 0:
            raise NodeExecutionError(
                self.node_id,
                "heads_mismatch",
                "errors.node.headsMismatch",
                {"dim": dim, "heads": heads},
            )
        if self.inner is None:
            self.inner = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=self.params["dropout"],
                batch_first=True,
            )
        mask = None
        if self.params["causal"]:
            seq_len = int(x.shape[1])
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
        if self.capture_attention:
            out, weights = self.inner(
                x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False
            )
            self.last_attn_weights = weights.detach()
            return out
        self.last_attn_weights = None
        out, _ = self.inner(x, x, x, attn_mask=mask, need_weights=False)
        return out


class ResidualAddNode(_NodeBase):
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tensors = [self._require_tensor(value) for value in inputs]
        if len(tensors) < 2:
            raise NodeExecutionError(
                self.node_id,
                "fan_in_missing",
                "errors.node.addNeedsTwo",
                {"count": len(tensors)},
            )
        shapes = [_shape_of(t) for t in tensors]
        if len({tuple(s or ()) for s in shapes}) != 1:
            raise NodeExecutionError(
                self.node_id,
                "shape_mismatch",
                "errors.node.addShape",
                {"shapes": shapes},
            )
        total = tensors[0]
        for extra in tensors[1:]:
            total = total + extra
        return total


class ConcatNode(_NodeBase):
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tensors = [self._require_tensor(value) for value in inputs]
        if len(tensors) < 2:
            raise NodeExecutionError(
                self.node_id,
                "fan_in_missing",
                "errors.node.catNeedsTwo",
                {"count": len(tensors)},
            )
        shapes = [list(t.shape) for t in tensors]
        dim = self.params["dim"]
        ndim = tensors[0].dim()
        norm_dim = dim if dim >= 0 else dim + ndim
        base = list(shapes[0])
        for shape in shapes[1:]:
            if len(shape) != len(base) or any(
                shape[i] != base[i] for i in range(len(base)) if i != norm_dim
            ):
                raise NodeExecutionError(
                    self.node_id,
                    "shape_mismatch",
                    "errors.node.catShape",
                    {"shapes": shapes, "dim": dim},
                )
        return torch.cat(tensors, dim=norm_dim)


class OutputNode(_NodeBase):
    """图终点：声明式校验入端最后一维（不做投影）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._require_tensor(x)
        out_dim = int(self.params.get("out_dim") or 0)
        expected = out_dim if out_dim > 0 else int(self.params["classes"])
        got = int(x.shape[-1]) if x.dim() >= 1 else -1
        if got != expected:
            raise NodeExecutionError(
                self.node_id,
                "shape_mismatch",
                "errors.node.outputDim",
                {
                    "expected": expected,
                    "got": got,
                    "kind": "regression" if out_dim > 0 else "classification",
                },
            )
        return x


_NODE_CLASSES: dict[str, type[_NodeBase]] = {
    "Input": InputNode,
    "Linear": LinearNode,
    "Conv2d": Conv2dNode,
    "MaxPool2d": MaxPool2dNode,
    "AvgPool2d": AvgPool2dNode,
    "AdaptiveAvgPool2d": AdaptiveAvgPool2dNode,
    "Flatten": FlattenNode,
    "Dropout": DropoutNode,
    "Activation": ActivationNode,
    "BatchNorm2d": BatchNorm2dNode,
    "BatchNorm1d": BatchNorm1dNode,
    "LayerNorm": LayerNormNode,
    "Embedding": EmbeddingNode,
    "PositionalEncoding": PositionalEncodingNode,
    "LSTM": LSTMNode,
    "MultiHeadAttention": MultiHeadAttentionNode,
    "ResidualAdd": ResidualAddNode,
    "Concat": ConcatNode,
    "Output": OutputNode,
}


class GraphModule(nn.Module):
    """通用拓扑执行器：nodes 按 n_<id> 注册，前向按拓扑序执行，入边多者按端口序传参。"""

    def __init__(self, graph: GraphIR, params_by_node: dict[str, dict[str, Any]]):
        super().__init__()
        order = topological_order(graph, params_by_node)
        if order is None:
            raise ValueError("graph is not a valid DAG")
        self.order = order
        input_nodes = [n for n in graph.nodes if n.type == "Input"]
        output_nodes = [n for n in graph.nodes if n.type == "Output"]
        if len(input_nodes) != 1 or len(output_nodes) != 1:
            raise ValueError("graph must have exactly one Input and one Output")
        self.input_id = input_nodes[0].id
        self.output_id = output_nodes[0].id

        modules: dict[str, nn.Module] = {}
        for node in graph.nodes:
            cls = _NODE_CLASSES[node.type]
            modules[f"n_{node.id}"] = cls(node, params_by_node[node.id])
        self.nodes_map = nn.ModuleDict(modules)

        fan_in = fan_in_map(graph)
        self.fan_in: dict[str, list[Edge]] = fan_in
        self.sources: dict[str, list[str]] = {
            node_id: [edge.source for edge in edges] for node_id, edges in fan_in.items()
        }

    def node_module(self, node_id: str) -> _NodeBase:
        return self.nodes_map[f"n_{node_id}"]  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outs: dict[str, torch.Tensor] = {}
        for node_id in self.order:
            module = self.node_module(node_id)
            try:
                if node_id == self.input_id:
                    outs[node_id] = module(x)
                    continue
                sources = self.sources.get(node_id, [])
                inputs = [outs[src] for src in sources]
                outs[node_id] = module(*inputs) if len(inputs) > 1 else module(inputs[0])
            except NodeExecutionError:
                raise
            except Exception as exc:  # noqa: BLE001 - 统一转成节点级错误
                raise NodeExecutionError(
                    node_id,
                    "forward_failed",
                    "errors.node.forward",
                    {"type": type(module).__name__},
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
        return outs[self.output_id]


def build_graph_module(graph: GraphIR, params_by_node: dict[str, dict[str, Any]]) -> GraphModule:
    return GraphModule(graph, params_by_node)
