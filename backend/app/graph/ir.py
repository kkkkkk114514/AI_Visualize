"""图 IR：节点白名单、解析与结构校验（契约见 docs/02 §3）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from app import config

Severity = Literal["error", "warning"]

ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh", "sigmoid")
POS_ENCODING_MODES = ("learned", "sinusoidal")
MAX_FAN_IN = 4


@dataclass(frozen=True)
class ParamSpec:
    kind: Literal["int", "float", "bool", "enum", "int_pair"]
    default: Any
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    # 该参数由数据集元数据决定（UI 只读）
    dataset_bound: bool = False


# 节点白名单与参数规格：白名单之外的类型一律拒绝（docs/02 §3.2）
NODE_PARAMS: dict[str, dict[str, ParamSpec]] = {
    "Input": {"shape": ParamSpec("int_pair", default=[1, 28, 28], dataset_bound=True)},
    "Linear": {
        "out_features": ParamSpec("int", default=128, minimum=1, maximum=1 << 20),
        "bias": ParamSpec("bool", default=True),
    },
    "Conv2d": {
        "out_channels": ParamSpec("int", default=16, minimum=1, maximum=4096),
        "kernel_size": ParamSpec("int", default=3, minimum=1, maximum=32),
        "stride": ParamSpec("int", default=1, minimum=1, maximum=32),
        "padding": ParamSpec("int", default=1, minimum=0, maximum=32),
        "dilation": ParamSpec("int", default=1, minimum=1, maximum=8),
        "groups": ParamSpec("int", default=1, minimum=1, maximum=4096),
    },
    "MaxPool2d": {
        "kernel_size": ParamSpec("int", default=2, minimum=1, maximum=32),
        "stride": ParamSpec("int", default=0, minimum=0, maximum=32),
        "padding": ParamSpec("int", default=0, minimum=0, maximum=32),
    },
    "AvgPool2d": {
        "kernel_size": ParamSpec("int", default=2, minimum=1, maximum=32),
        "stride": ParamSpec("int", default=0, minimum=0, maximum=32),
        "padding": ParamSpec("int", default=0, minimum=0, maximum=32),
    },
    "AdaptiveAvgPool2d": {"output_size": ParamSpec("int", default=1, minimum=1, maximum=256)},
    "Flatten": {"start_dim": ParamSpec("int", default=1, minimum=0, maximum=4)},
    "Dropout": {"p": ParamSpec("float", default=0.5, minimum=0.0, maximum=0.99)},
    "Activation": {"name": ParamSpec("enum", default="relu", choices=ACTIVATIONS)},
    "BatchNorm2d": {
        "eps": ParamSpec("float", default=1e-5, minimum=0.0, maximum=1.0),
        "momentum": ParamSpec("float", default=0.1, minimum=0.0, maximum=1.0),
    },
    "BatchNorm1d": {
        "eps": ParamSpec("float", default=1e-5, minimum=0.0, maximum=1.0),
        "momentum": ParamSpec("float", default=0.1, minimum=0.0, maximum=1.0),
    },
    "LayerNorm": {"normalized_shape": ParamSpec("int", default=0, minimum=0, maximum=1 << 20)},
    "Embedding": {
        "num_embeddings": ParamSpec("int", default=5000, minimum=2, maximum=1 << 20, dataset_bound=True),
        "embedding_dim": ParamSpec("int", default=64, minimum=1, maximum=4096),
    },
    "PositionalEncoding": {
        "max_len": ParamSpec("int", default=128, minimum=1, maximum=8192),
        "mode": ParamSpec("enum", default="learned", choices=POS_ENCODING_MODES),
    },
    "LSTM": {
        "hidden_size": ParamSpec("int", default=128, minimum=1, maximum=4096),
        "num_layers": ParamSpec("int", default=1, minimum=1, maximum=8),
        "bidirectional": ParamSpec("bool", default=False),
        "return_sequences": ParamSpec("bool", default=True),
    },
    "MultiHeadAttention": {
        "num_heads": ParamSpec("int", default=4, minimum=1, maximum=64),
        "causal": ParamSpec("bool", default=True),
        "dropout": ParamSpec("float", default=0.0, minimum=0.0, maximum=0.9),
    },
    "ResidualAdd": {},
    "Concat": {"dim": ParamSpec("int", default=-1, minimum=-4, maximum=3)},
    "Output": {
        "classes": ParamSpec("int", default=10, minimum=1, maximum=1 << 20, dataset_bound=True),
        "out_dim": ParamSpec("int", default=0, minimum=0, maximum=1 << 20, dataset_bound=True),
    },
}

# 入度上限 >1 的节点（多输入端口 in1..inN）
MULTI_INPUT_TYPES = ("ResidualAdd", "Concat")

# 入度必须 ≥2 的节点
MIN_FAN_IN = {"ResidualAdd": 2, "Concat": 2}

REQUIRED_INPUT_TYPES = ("Input",)
REQUIRED_OUTPUT_TYPES = ("Output",)


@dataclass
class Issue:
    code: str
    message_key: str
    severity: Severity = "error"
    node_id: str | None = None
    edge_id: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "message_key": self.message_key,
            "severity": self.severity,
            "args": self.args,
        }
        if self.node_id is not None:
            out["node_id"] = self.node_id
        if self.edge_id is not None:
            out["edge_id"] = self.edge_id
        return out


@dataclass
class Node:
    id: str
    type: str
    params: dict[str, Any]
    ui: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    id: str
    source: str
    source_port: str
    target: str
    target_port: str


@dataclass
class GraphIR:
    id: str
    kind: str
    task: str
    nodes: list[Node]
    edges: list[Edge]
    name: Any = None
    desc: Any = None
    ir_version: int = 1
    hyper_defaults: dict[str, Any] = field(default_factory=dict)
    probe_defaults: list[str] = field(default_factory=list)

    def node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "id": self.id,
            "kind": self.kind,
            "task": self.task,
            "name": self.name,
            "desc": self.desc,
            "hyper_defaults": self.hyper_defaults,
            "probe_defaults": self.probe_defaults,
            "nodes": [
                {"id": n.id, "type": n.type, "params": n.params, "ui": n.ui} for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "source_port": e.source_port,
                    "target": e.target,
                    "target_port": e.target_port,
                }
                for e in self.edges
            ],
        }

    def canonical(self) -> str:
        """缓存 key 用的规范 JSON：忽略 ui 坐标与展示性字段。"""
        payload = {
            "nodes": [
                {"id": n.id, "type": n.type, "params": n.params}
                for n in sorted(self.nodes, key=lambda n: n.id)
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "source_port": e.source_port,
                    "target": e.target,
                    "target_port": e.target_port,
                }
                for e in sorted(self.edges, key=lambda e: e.id)
            ],
            "task": self.task,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class IRParseError(ValueError):
    def __init__(self, message_key: str, detail: str = "", error_args: dict[str, Any] | None = None):
        super().__init__(detail or message_key)
        self.message_key = message_key
        self.detail = detail
        self.error_args = error_args or {}


def default_params(node_type: str) -> dict[str, Any]:
    return {name: spec.default for name, spec in NODE_PARAMS.get(node_type, {}).items()}


def parse_graph(data: dict[str, Any]) -> GraphIR:
    if not isinstance(data, dict):
        raise IRParseError("errors.graph.notObject", detail=type(data).__name__)

    nodes: list[Node] = []
    for raw in data.get("nodes") or []:
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("type"):
            raise IRParseError("errors.graph.badNode", detail=json.dumps(raw, ensure_ascii=False)[:200])
        nodes.append(
            Node(
                id=str(raw["id"]),
                type=str(raw["type"]),
                params=dict(raw.get("params") or {}),
                ui=dict(raw.get("ui") or {}),
            )
        )

    edges: list[Edge] = []
    for raw in data.get("edges") or []:
        if not isinstance(raw, dict) or not raw.get("source") or not raw.get("target"):
            raise IRParseError("errors.graph.badEdge", detail=json.dumps(raw, ensure_ascii=False)[:200])
        edges.append(
            Edge(
                id=str(raw.get("id") or f"{raw['source']}->{raw['target']}"),
                source=str(raw["source"]),
                source_port=str(raw.get("source_port") or "out"),
                target=str(raw["target"]),
                target_port=str(raw.get("target_port") or "in"),
            )
        )

    if not nodes:
        raise IRParseError("errors.graph.empty", detail="nodes is empty")

    return GraphIR(
        id=str(data.get("id") or "graph"),
        kind=str(data.get("kind") or "dl"),
        task=str(data.get("task") or "image_classification"),
        name=data.get("name"),
        desc=data.get("desc"),
        ir_version=int(data.get("ir_version") or 1),
        hyper_defaults=dict(data.get("hyper_defaults") or {}),
        probe_defaults=list(data.get("probe_defaults") or []),
        nodes=nodes,
        edges=edges,
    )


def coerce_params(node: Node, issues: list[Issue]) -> dict[str, Any]:
    """按规格表补默认值、做范围/枚举检查，返回可直接交给构建器的参数字典。"""
    specs = NODE_PARAMS[node.type]
    params: dict[str, Any] = {}
    for name, spec in specs.items():
        value = node.params.get(name, spec.default)
        if spec.kind == "bool":
            if not isinstance(value, bool):
                issues.append(
                    Issue(
                        code="bad_param",
                        message_key="errors.param.type",
                        node_id=node.id,
                        args={"param": name, "expected": "bool"},
                    )
                )
                value = spec.default
            params[name] = value
        elif spec.kind in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(
                    Issue(
                        code="bad_param",
                        message_key="errors.param.type",
                        node_id=node.id,
                        args={"param": name, "expected": spec.kind},
                    )
                )
                value = spec.default
            if spec.kind == "int":
                value = int(round(float(value)))
            else:
                value = float(value)
            if spec.minimum is not None and value < spec.minimum:
                issues.append(
                    Issue(
                        code="param_out_of_range",
                        message_key="errors.param.range",
                        node_id=node.id,
                        args={"param": name, "min": spec.minimum, "max": spec.maximum},
                    )
                )
                value = spec.minimum
            if spec.maximum is not None and value > spec.maximum:
                issues.append(
                    Issue(
                        code="param_out_of_range",
                        message_key="errors.param.range",
                        node_id=node.id,
                        args={"param": name, "min": spec.minimum, "max": spec.maximum},
                    )
                )
                value = spec.maximum
            params[name] = value
        elif spec.kind == "enum":
            if value not in spec.choices:
                issues.append(
                    Issue(
                        code="bad_param",
                        message_key="errors.param.choice",
                        node_id=node.id,
                        args={"param": name, "choices": list(spec.choices)},
                    )
                )
                value = spec.default
            params[name] = value
        elif spec.kind == "int_pair":
            value = _coerce_int_pair(value, node, name, issues)
            params[name] = value
        else:  # pragma: no cover - 规格表未使用其他 kind
            params[name] = value
    return params


def _coerce_int_pair(value: Any, node: Node, name: str, issues: list[Issue]) -> list[int]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [int(value)]
    if not isinstance(value, (list, tuple)) or not value:
        issues.append(
            Issue(
                code="bad_param",
                message_key="errors.param.type",
                node_id=node.id,
                args={"param": name, "expected": "int_pair"},
            )
        )
        return [1]
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            issues.append(
                Issue(
                    code="bad_param",
                    message_key="errors.param.type",
                    node_id=node.id,
                    args={"param": name, "expected": "int_pair"},
                )
            )
            return [1]
        out.append(int(round(float(item))))
    return out


def validate_graph(graph: GraphIR) -> tuple[list[Issue], dict[str, dict[str, Any]]]:
    """结构校验（不跑前向）。返回 (issues, 逐节点补全后的 params)。"""
    issues: list[Issue] = []
    params_by_node: dict[str, dict[str, Any]] = {}

    seen: set[str] = set()
    for node in graph.nodes:
        if node.id in seen:
            issues.append(Issue(code="duplicate_node_id", message_key="errors.graph.duplicateNode", node_id=node.id))
            continue
        seen.add(node.id)
        if node.type not in NODE_PARAMS:
            issues.append(
                Issue(
                    code="unknown_node_type",
                    message_key="errors.graph.unknownType",
                    node_id=node.id,
                    args={"type": node.type},
                )
            )
            continue
        params_by_node[node.id] = coerce_params(node, issues)

    inputs = [n for n in graph.nodes if n.type in REQUIRED_INPUT_TYPES]
    outputs = [n for n in graph.nodes if n.type in REQUIRED_OUTPUT_TYPES]
    if not inputs:
        issues.append(Issue(code="missing_input", message_key="errors.graph.missingInput"))
    elif len(inputs) > 1:
        issues.append(
            Issue(
                code="multiple_inputs",
                message_key="errors.graph.multipleInput",
                node_id=inputs[1].id,
                args={"count": len(inputs)},
            )
        )
    if not outputs:
        issues.append(Issue(code="missing_output", message_key="errors.graph.missingOutput"))
    elif len(outputs) > 1:
        issues.append(
            Issue(
                code="multiple_outputs",
                message_key="errors.graph.multipleOutput",
                node_id=outputs[1].id,
                args={"count": len(outputs)},
            )
        )

    fan_in: dict[str, list[Edge]] = {n.id: [] for n in graph.nodes}
    edge_ids: set[str] = set()
    for edge in graph.edges:
        if edge.id in edge_ids:
            issues.append(Issue(code="duplicate_edge_id", message_key="errors.graph.duplicateEdge", edge_id=edge.id))
            continue
        edge_ids.add(edge.id)
        if edge.source not in seen or edge.target not in seen:
            issues.append(
                Issue(
                    code="dangling_edge",
                    message_key="errors.graph.danglingEdge",
                    edge_id=edge.id,
                    args={"source": edge.source, "target": edge.target},
                )
            )
            continue
        if edge.source_port != "out":
            issues.append(
                Issue(
                    code="bad_port",
                    message_key="errors.graph.badPort",
                    edge_id=edge.id,
                    args={"port": edge.source_port},
                )
            )
            continue
        fan_in[edge.target].append(edge)

    for node in graph.nodes:
        if node.type not in NODE_PARAMS:
            continue
        incoming = fan_in[node.id]
        limit = MAX_FAN_IN if node.type in MULTI_INPUT_TYPES else 1
        if len(incoming) > limit:
            issues.append(
                Issue(
                    code="fan_in_overflow",
                    message_key="errors.graph.fanInOverflow",
                    node_id=node.id,
                    args={"count": len(incoming), "limit": limit},
                )
            )
        if node.type in MIN_FAN_IN and len(incoming) < MIN_FAN_IN[node.type]:
            issues.append(
                Issue(
                    code="fan_in_missing",
                    message_key="errors.graph.fanInMissing",
                    node_id=node.id,
                    args={"need": MIN_FAN_IN[node.type], "count": len(incoming)},
                )
            )
        if node.type == "Output" and not incoming:
            issues.append(Issue(code="output_no_input", message_key="errors.graph.outputNoInput", node_id=node.id))
        if node.type in MULTI_INPUT_TYPES:
            expected = [f"in{i + 1}" for i in range(len(incoming))]
            got = sorted(e.target_port for e in incoming)
            if got != sorted(expected):
                issues.append(
                    Issue(
                        code="bad_port",
                        message_key="errors.graph.multiInPort",
                        node_id=node.id,
                        args={"ports": got},
                    )
                )

    if inputs and outputs and not _has_errors(issues):
        order = topological_order(graph, params_by_node)
        if order is None:
            issues.append(Issue(code="cycle", message_key="errors.graph.cycle"))
        else:
            reachable = _reachable(graph, order, inputs[0].id)
            if outputs[0].id not in reachable:
                issues.append(
                    Issue(
                        code="output_unreachable",
                        message_key="errors.graph.outputUnreachable",
                        node_id=outputs[0].id,
                    )
                )
            for node in graph.nodes:
                if node.id not in reachable:
                    issues.append(
                        Issue(
                            code="unreachable_node",
                            message_key="errors.graph.unreachable",
                            severity="warning",
                            node_id=node.id,
                        )
                    )
    return issues, params_by_node


def _has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)


def topological_order(graph: GraphIR, params_by_node: dict[str, dict[str, Any]] | None = None) -> list[str] | None:
    """Kahn 拓扑排序；有环或图非法（缺 Input/Output）时返回 None。"""
    if params_by_node is None:
        params_by_node = {n.id: dict(n.params) for n in graph.nodes}
    if not params_by_node:
        return None
    indegree = {node_id: 0 for node_id in params_by_node}
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in params_by_node}
    for edge in graph.edges:
        if edge.source in indegree and edge.target in indegree:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1
    ready = sorted([node_id for node_id, deg in indegree.items() if deg == 0])
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for nxt in sorted(adjacency[node_id]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    if len(order) != len(indegree):
        return None
    return order


def fan_in_map(graph: GraphIR) -> dict[str, list[Edge]]:
    """target 节点 → 按端口序排序的入边列表。"""
    fan_in: dict[str, list[Edge]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.target in fan_in:
            fan_in[edge.target].append(edge)

    def port_key(edge: Edge) -> tuple[int, str]:
        port = edge.target_port
        if port.startswith("in") and port[2:].isdigit():
            return (int(port[2:]), port)
        return (0, port)

    for node_id in fan_in:
        fan_in[node_id].sort(key=port_key)
    return fan_in


def _reachable(graph: GraphIR, order: list[str], start: str) -> set[str]:
    adjacency: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        if edge.source in adjacency:
            adjacency[edge.source].append(edge.target)
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in adjacency.get(current, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def bind_dataset(graph: GraphIR, params_by_node: dict[str, dict[str, Any]], meta: dict[str, Any]) -> None:
    """把数据集元数据写回 dataset_bound 参数（docs/02 §4.1）：Input.shape / Output.classes / Embedding 词表。"""
    shape = meta.get("input_shape")
    num_classes = meta.get("num_classes")
    vocab_size = meta.get("vocab_size")
    for node in graph.nodes:
        params = params_by_node.get(node.id)
        if params is None:
            continue
        if node.type == "Input" and shape:
            params["shape"] = [int(value) for value in shape]
        elif node.type == "Output":
            if not params.get("out_dim") and num_classes:
                params["classes"] = int(num_classes)
        elif node.type == "Embedding" and vocab_size:
            params["num_embeddings"] = int(vocab_size)


def load_preset(model_id: str) -> GraphIR | None:
    path = config.PRESETS_DIR / f"{model_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IRParseError("errors.graph.presetUnreadable", detail=f"{path.name}: {exc}") from exc
    return parse_graph(data)
