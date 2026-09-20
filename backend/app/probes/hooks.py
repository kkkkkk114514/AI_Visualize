"""探针采样（契约见 docs/02 §6.1）：配置解析、采样步挂/摘 hook、每流上限、编码落盘与事件上报。

子进程侧运行：hook 只在采样步注册、前向返回后立即摘除，非采样步完全不介入前向。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.hooks import RemovableHandle

from app import config
from app.graph.ir import GraphIR
from app.probes import encode
from app.runners import base

log = logging.getLogger(__name__)


class ProbeError(Exception):
    """探针配置非法（主进程在建 run 时转 422）。"""

    def __init__(self, message_key: str, error_args: dict[str, Any] | None = None, detail: str = ""):
        super().__init__(detail or message_key)
        self.message_key = message_key
        self.error_args = error_args or {}
        self.detail = detail


@dataclass(frozen=True)
class ProbeSpec:
    node_id: str
    kind: str
    every_n_steps: int
    sample_index: int
    max_items: int

    @property
    def stream(self) -> str:
        return f"{self.node_id}:{self.kind}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "every_n_steps": self.every_n_steps,
            "sample_index": self.sample_index,
            "max_items": self.max_items,
        }


def _int_field(value: Any, name: str, minimum: int, maximum: int) -> int:
    if value is None:
        raise ProbeError("errors.probe.badValue", {"param": name})
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError("errors.probe.badValue", {"param": name})
    number = int(round(float(value)))
    if number < minimum or number > maximum:
        raise ProbeError(
            "errors.probe.badValue", {"param": name, "min": minimum, "max": maximum}
        )
    return number


def resolve_probes(raw: Any, graph: GraphIR) -> list[ProbeSpec]:
    """请求里的 `probes`（缺省用图的 `probe_defaults`）→ 规范化条目；非法即抛 ProbeError。"""
    if raw is None:
        entries: list[Any] = [{"node_id": node_id} for node_id in graph.probe_defaults]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ProbeError("errors.probe.badConfig", detail=type(raw).__name__)

    node_types = {node.id: node.type for node in graph.nodes}
    specs: list[ProbeSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProbeError("errors.probe.badConfig", detail=type(entry).__name__)
        node_id = entry.get("node_id")
        if not isinstance(node_id, str) or node_id not in node_types:
            raise ProbeError("errors.probe.unknownNode", {"node_id": node_id})
        kind = entry.get("kind") or encode.derived_kind(node_types[node_id])
        if kind not in encode.KINDS:
            raise ProbeError("errors.probe.badKind", {"kind": kind, "node_id": node_id})
        spec = ProbeSpec(
            node_id=node_id,
            kind=str(kind),
            every_n_steps=_int_field(
                entry.get("every_n_steps", config.PROBE_DEFAULT_EVERY_N),
                "every_n_steps",
                1,
                100_000,
            ),
            sample_index=_int_field(entry.get("sample_index", 0), "sample_index", 0, 4096),
            max_items=_int_field(
                entry.get("max_items", config.PROBE_MAX_ITEMS), "max_items", 1, 256
            ),
        )
        if spec.stream in seen:
            continue
        seen.add(spec.stream)
        specs.append(spec)
    return specs


def find_weight_tensor(module: torch.nn.Module) -> torch.Tensor | None:
    """weights 探针取该节点第一个 ≥2 维的参数（Linear/Conv2d/MHA 的权重矩阵）。"""
    for _, param in module.named_parameters():
        if param.dim() >= 2:
            return param
    return None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ProbeEngine:
    """采样调度：`begin_step` 挂 hook → `end_step` 摘 hook、编码、落盘、发事件。"""

    def __init__(
        self,
        specs: list[ProbeSpec],
        model: torch.nn.Module,
        node_types: dict[str, str],
        *,
        run_id: str,
        snapshots_dir: Path,
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        self._specs = list(specs)
        self._model = model
        self._node_types = node_types
        self._run_id = run_id
        self._dir = Path(snapshots_dir)
        self._emit = emit
        self._counts: dict[str, int] = {}
        self._disabled: dict[str, str] = {}
        self._capped: set[str] = set()
        self._due: list[ProbeSpec] = []
        self._captured: dict[str, torch.Tensor] = {}
        self._hooks: list[RemovableHandle] = []
        self._step = 0
        self._epoch = 0

    def active(self) -> bool:
        return bool(self._specs)

    def spec_payload(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs]

    # ------------------------------------------------------------ 采样步

    def begin_step(self, step: int, epoch: int) -> bool:
        self._step, self._epoch = step, epoch
        self._captured = {}
        self._due = [
            spec
            for spec in self._specs
            if spec.stream not in self._disabled and step % spec.every_n_steps == 0
        ]
        if not self._due:
            return False
        for spec in self._due:
            module = self._model.node_module(spec.node_id)
            if spec.kind == "attention":
                if hasattr(module, "capture_attention"):
                    module.capture_attention = True
                else:
                    self._disable(spec, "errors.probe.kindMismatch")
                    continue
            if spec.kind == "weights":
                continue
            self._hooks.append(
                module.register_forward_hook(self._make_hook(spec))
            )
        return True

    def _make_hook(self, spec: ProbeSpec) -> Callable[..., None]:
        def hook(module: torch.nn.Module, _args: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self._captured[spec.stream] = output.detach()

        return hook

    def end_step(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if not self._due:
            return
        for spec in self._due:
            module = self._model.node_module(spec.node_id)
            if spec.kind == "attention":
                weights = getattr(module, "last_attn_weights", None)
                module.capture_attention = False
                module.last_attn_weights = None
                tensor = weights
            elif spec.kind == "weights":
                tensor = find_weight_tensor(module)
            else:
                tensor = self._captured.get(spec.stream)
            if tensor is None:
                self._disable(spec, "errors.probe.kindMismatch")
                continue
            self._write(spec, tensor, module)
        self._due = []
        self._captured = {}

    def _write(self, spec: ProbeSpec, tensor: torch.Tensor, module: torch.nn.Module) -> None:
        options: dict[str, Any] = {"max_items": spec.max_items, "sample_index": spec.sample_index}
        if spec.kind == "attention":
            options["causal"] = bool(getattr(module, "params", {}).get("causal", False))
        try:
            payload = encode.encode(spec.kind, tensor, **options)
        except encode.ProbeShapeError as exc:
            self._disable(spec, exc.message_key, exc.error_args)
            return

        snapshot_id = encode.new_snapshot_id()
        created_at = _now_iso()
        record = {
            "id": snapshot_id,
            "run_id": self._run_id,
            "step": self._step,
            "epoch": self._epoch,
            "node_id": spec.node_id,
            "kind": spec.kind,
            **payload,
            "created_at": created_at,
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{snapshot_id}.json"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        self._counts[spec.stream] = self._counts.get(spec.stream, 0) + 1
        self._emit(
            {
                "type": base.EVENT_PROBE,
                "run_id": self._run_id,
                "snapshot_id": snapshot_id,
                "step": self._step,
                "epoch": self._epoch,
                "node_id": spec.node_id,
                "kind": spec.kind,
                "shape": payload["shape"],
                "min": payload["min"],
                "max": payload["max"],
                "file_path": str(path),
            }
        )
        if self._counts[spec.stream] >= config.PROBE_MAX_PER_STREAM:
            self._capped.add(spec.stream)
            self._disabled[spec.stream] = "capped"
            self._log_warning(
                "log.probe.capped",
                {"node_id": spec.node_id, "kind": spec.kind, "limit": config.PROBE_MAX_PER_STREAM},
            )

    def _disable(self, spec: ProbeSpec, message_key: str, args: dict[str, Any] | None = None) -> None:
        if spec.stream in self._disabled:
            return
        self._disabled[spec.stream] = message_key
        self._log_warning(
            "log.probe.disabled", {"node_id": spec.node_id, "kind": spec.kind, **(args or {})}
        )

    def _log_warning(self, key: str, args: dict[str, Any]) -> None:
        self._emit({"type": base.EVENT_LOG, "level": "warning", "key": key, "args": args})
