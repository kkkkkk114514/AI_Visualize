"""dry-run：真实前向推导逐节点形状与参数量（带 LRU 缓存）。"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

import torch

from app.graph.ir import GraphIR, Issue, bind_dataset, validate_graph
from app.graph.module import GraphModule, NodeExecutionError, build_graph_module

DRY_RUN_BATCH = 2
CACHE_SIZE = 64

_TEXT_TASK_HINTS = ("text", "language", "seq")

_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_cache_lock = threading.Lock()


def is_text_task(task: str) -> bool:
    lowered = task.lower()
    return any(hint in lowered for hint in _TEXT_TASK_HINTS)


def make_dummy_input(graph: GraphIR, input_params: dict[str, Any], batch: int = DRY_RUN_BATCH) -> torch.Tensor:
    shape = [int(v) for v in input_params.get("shape") or [1]]
    if is_text_task(graph.task):
        return torch.zeros([batch, *shape], dtype=torch.int64)
    return torch.zeros([batch, *shape], dtype=torch.float32)


def _shape_list(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def _run(graph: GraphIR, batch: int, dataset_id: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    issues, params_by_node = validate_graph(graph)
    errors = [i.to_dict() for i in issues if i.severity == "error"]
    warnings = [i.to_dict() for i in issues if i.severity == "warning"]
    result: dict[str, Any] = {
        "ok": False,
        "nodes": {},
        "total_params": 0,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    meta = dataset_meta(dataset_id)
    if meta is not None:
        bind_dataset(graph, params_by_node, meta)

    input_node = next(n for n in graph.nodes if n.type == "Input")
    dummy = make_dummy_input(graph, params_by_node[input_node.id], batch)

    captured: dict[str, dict[str, Any]] = {}

    def make_hook(node_id: str):
        def hook(module: GraphModule, args: tuple[Any, ...], output: Any, _node_id: str = node_id) -> None:
            in_shapes = [s for s in (_shape_list(a) for a in args) if s is not None]
            entry: dict[str, Any] = {"out_shape": _shape_list(output)}
            if len(in_shapes) == 1:
                entry["in_shape"] = in_shapes[0]
            elif in_shapes:
                entry["in_shapes"] = in_shapes
            captured[_node_id] = entry

        return hook

    model: GraphModule | None = None
    try:
        model = build_graph_module(graph, params_by_node)
        for node in graph.nodes:
            model.node_module(node.id).register_forward_hook(make_hook(node.id))
        model.eval()
        with torch.no_grad():
            model(dummy)
    except NodeExecutionError as exc:
        result["errors"] = [
            {
                "code": exc.code,
                "message_key": exc.message_key,
                "severity": "error",
                "node_id": exc.node_id,
                "args": exc.error_args,
                "detail": exc.detail,
            }
        ]
    except Exception as exc:  # noqa: BLE001 - 构建阶段失败（理论上已被结构校验拦住）
        result["errors"] = [
            {
                "code": "dry_run_failed",
                "message_key": "errors.graph.dryRunFailed",
                "severity": "error",
                "args": {},
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]

    if model is not None:
        nodes: dict[str, dict[str, Any]] = {}
        total = 0
        for node in graph.nodes:
            entry: dict[str, Any] = {"out_shape": None, **captured.get(node.id, {})}
            module = model.node_module(node.id)
            executed = node.id in captured
            entry["params"] = module.param_count() if executed else None
            if entry["params"] is not None:
                total += entry["params"]
            nodes[node.id] = entry
        result["nodes"] = nodes
        result["total_params"] = total

    result["ok"] = not result["errors"]
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def dataset_meta(dataset_id: str | None) -> dict[str, Any] | None:
    """已缓存数据集的元数据；未指定或未缓存时返回 None（退回 IR 里的 dataset_bound 参数）。"""
    if not dataset_id:
        return None
    from app.datasets import registry

    spec = registry.get_spec(dataset_id)
    if spec is None or not registry.is_cached(spec):
        return None
    return registry.meta(spec)


def analyze(graph: GraphIR, batch: int = DRY_RUN_BATCH, dataset_id: str | None = None) -> dict[str, Any]:
    """结构校验 + dry-run；结果按（忽略 ui 的规范 IR, batch, dataset）缓存。"""
    key = hashlib.sha256(
        f"{graph.canonical()}|batch={batch}|dataset={dataset_id or '-'}".encode("utf-8")
    ).hexdigest()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return copy.deepcopy(cached)

    result = _run(graph, batch, dataset_id)

    with _cache_lock:
        _cache[key] = copy.deepcopy(result)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def structural_issues(graph: GraphIR) -> list[Issue]:
    issues, _ = validate_graph(graph)
    return issues
