"""快照落盘 + 事件上报（契约见 docs/02 §6.2 / §6.3）：DL / ML / RL 共用同一写入器。

刻意不 import torch：ML / RL 子进程要求冷启动不加载 torch（docs/02 §13.4）。
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app import config
from app.runners import base


def new_snapshot_id() -> str:
    return f"s_{secrets.token_hex(4)}"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SnapshotWriter:
    """每个流累加到 `max_per_stream` 条即停发并警告一次（`log.probe.capped`）；

    `disable` 用于形状不符等一次性停用（`log.probe.disabled`），两者都只影响该流。
    """

    def __init__(
        self,
        *,
        run_id: str,
        snapshots_dir: Path,
        emit: Callable[[dict[str, Any]], None],
        max_per_stream: int | None = None,
    ) -> None:
        self._run_id = run_id
        self._dir = Path(snapshots_dir)
        self._emit = emit
        self._max = config.PROBE_MAX_PER_STREAM if max_per_stream is None else int(max_per_stream)
        self._counts: dict[str, int] = {}
        self._disabled: dict[str, str] = {}

    def is_disabled(self, stream: str) -> bool:
        return stream in self._disabled

    def count(self, stream: str) -> int:
        return self._counts.get(stream, 0)

    def log_attached(self, n: int) -> None:
        self._emit(
            {"type": base.EVENT_LOG, "level": "info", "key": "log.probe.attached", "args": {"n": n}}
        )

    def disable(self, stream: str, message_key: str, args: dict[str, Any] | None = None) -> None:
        if stream in self._disabled:
            return
        self._disabled[stream] = message_key
        self._emit(
            {
                "type": base.EVENT_LOG,
                "level": "warning",
                "key": "log.probe.disabled",
                "args": {"stream": stream, **(args or {})},
            }
        )

    def write(
        self, *, node_id: str, kind: str, step: int, epoch: int, payload: dict[str, Any]
    ) -> str | None:
        """写一个快照文件并广播元数据；该流已停用 / 到上限时返回 None。"""
        stream = f"{node_id}:{kind}"
        if self.is_disabled(stream):
            return None
        snapshot_id = new_snapshot_id()
        record = {
            "id": snapshot_id,
            "run_id": self._run_id,
            "step": step,
            "epoch": epoch,
            "node_id": node_id,
            "kind": kind,
            **payload,
            "created_at": _now_iso(),
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{snapshot_id}.json"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        self._counts[stream] = self.count(stream) + 1
        self._emit(
            {
                "type": base.EVENT_PROBE,
                "run_id": self._run_id,
                "snapshot_id": snapshot_id,
                "step": step,
                "epoch": epoch,
                "node_id": node_id,
                "kind": kind,
                "shape": payload["shape"],
                "min": payload["min"],
                "max": payload["max"],
                "file_path": str(path),
            }
        )
        if self._counts[stream] >= self._max:
            self._disabled[stream] = "capped"
            self._emit(
                {
                    "type": base.EVENT_LOG,
                    "level": "warning",
                    "key": "log.probe.capped",
                    "args": {"node_id": node_id, "kind": kind, "limit": self._max},
                }
            )
        return snapshot_id
