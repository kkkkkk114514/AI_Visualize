"""runner 协议（契约见 docs/02 §5.4 / §5.5）：事件与控制命令常量、超参解析、子进程公共设施。"""

from __future__ import annotations

import logging
import math
import queue
import time
import traceback
from typing import Any, Callable

log = logging.getLogger(__name__)

# 状态机（docs/02 §5.1）
STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_FINISHED = "finished"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

ACTIVE_STATUSES = (STATUS_CREATED, STATUS_RUNNING, STATUS_PAUSED)

# 子进程 → 主进程事件（docs/02 §5.5）
EVENT_STATUS = "status"
EVENT_METRICS = "metrics"
EVENT_PROBE = "probe"
EVENT_LOG = "log"
EVENT_ERROR = "error"
EVENT_BYE = "bye"

# WS 线上事件名（docs/02 §8.2）：子进程的 status 事件对外发成 run.status
WIRE_STATUS = "run.status"
WIRE_PROBE = "probe.snapshot"
WIRE_DELETED = "run.deleted"

# 主进程 → 子进程控制命令（docs/02 §5.2）
CTRL_PAUSE = "pause"
CTRL_RESUME = "resume"
CTRL_STOP = "stop"
CTRL_KILL = "kill"
CTRL_SET_LR = "set_lr"
CTRL_SET_BATCH_SIZE = "set_batch_size"
CONTROL_ACTIONS = (CTRL_PAUSE, CTRL_RESUME, CTRL_STOP, CTRL_KILL, CTRL_SET_LR, CTRL_SET_BATCH_SIZE)

OPTIMIZERS = ("adam", "adamw", "sgd")
LOSSES = ("cross_entropy", "mse")

HYPER_DEFAULTS: dict[str, Any] = {
    "optimizer": "adam",
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 3,
    "loss": "cross_entropy",
    "grad_clip": 5.0,
    "train_size": 20000,
    "val_size": 5000,
}

_LIMITS: dict[str, tuple[float, float]] = {
    "lr": (1e-6, 1.0),
    "batch_size": (1, 4096),
    "epochs": (1, 50),
    "grad_clip": (0.0, 1000.0),
    "train_size": (0, 10_000_000),
    "val_size": (0, 1_000_000),
}

# 指标上报节流（docs/02 §5.4）
REPORT_STEPS = 20
REPORT_INTERVAL_S = 1.0
PAUSE_POLL_S = 0.1
THROUGHPUT_WINDOW = 20


class HyperError(ValueError):
    """超参非法：带 i18n key 与字段名，API 层转 422。"""

    def __init__(self, param: str, detail: str = "", message_key: str = "errors.run.invalidHyper"):
        super().__init__(detail or param)
        self.param = param
        self.message_key = message_key
        self.detail = detail
        self.error_args = {"param": param}


def resolve_hyperparams(model_defaults: dict[str, Any] | None, requested: dict[str, Any] | None) -> dict[str, Any]:
    """模型 hyper_defaults → 请求值 → 全局默认，逐项类型归一与范围校验。"""
    merged: dict[str, Any] = dict(HYPER_DEFAULTS)
    for source in (model_defaults or {}, requested or {}):
        for key, value in source.items():
            if key in HYPER_DEFAULTS and value is not None:
                merged[key] = value

    resolved: dict[str, Any] = {}
    for key, value in merged.items():
        if key in ("optimizer", "loss"):
            if not isinstance(value, str) or value not in (OPTIMIZERS if key == "optimizer" else LOSSES):
                raise HyperError(key)
            resolved[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HyperError(key, detail=f"{key}={value!r}")
        low, high = _LIMITS[key]
        number = float(value)
        if number < low or number > high:
            raise HyperError(key, detail=f"{key}={number} not in [{low}, {high}]")
        resolved[key] = number if key in ("lr", "grad_clip") else int(round(number))
    return resolved


def status_event(status: str, *, step: int = 0, epoch: int = 0, elapsed_s: float = 0.0, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": EVENT_STATUS,
        "status": status,
        "step": step,
        "epoch": epoch,
        "elapsed_s": round(elapsed_s, 3),
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------- 子进程公共设施


class RunFailed(Exception):
    """训练前置失败（数据集 / 图 / 超参 / algo spec）：带定位信息，转 error 事件。"""

    def __init__(
        self,
        code: str,
        message_key: str,
        error_args: dict[str, Any] | None = None,
        detail: str = "",
        node_id: str | None = None,
    ):
        super().__init__(detail or message_key)
        self.code = code
        self.message_key = message_key
        self.error_args = error_args or {}
        self.detail = detail
        self.node_id = node_id


class RunStopped(Exception):
    """用户停止：走正常收尾路径（flush + stopped 状态）。"""


class ControlState:
    """控制队列的当前意图（docs/02 §5.2）：pause / stop / lr / batch_size。"""

    def __init__(self, lr: float) -> None:
        self.pause = False
        self.stop = False
        self.lr = lr
        self.lr_dirty = False
        self.batch_size: int | None = None


def drain_control(control_queue: Any, state: ControlState) -> None:
    """非阻塞 drain 控制队列；`kill` 直接退出进程（主进程随后 terminate 兜底）。"""
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return
        action = command.get("action")
        if action == CTRL_PAUSE:
            state.pause = True
        elif action == CTRL_RESUME:
            state.pause = False
        elif action == CTRL_STOP:
            state.stop = True
        elif action == CTRL_KILL:
            raise SystemExit(0)
        elif action == CTRL_SET_LR:
            state.lr = float(command.get("value"))
            state.lr_dirty = True
        elif action == CTRL_SET_BATCH_SIZE:
            state.batch_size = int(command.get("value"))
        else:
            log.warning("未知控制命令：%r", command)


def handle_pause(
    control_queue: Any,
    state: ControlState,
    *,
    reporter: "Reporter",
    emit: Callable[[dict[str, Any]], None],
    step: int,
    epoch: int,
    elapsed: Callable[[], float],
) -> None:
    """暂停时不占 CPU：轮询控制队列，恢复 / 停止都由主进程命令驱动。"""
    if not state.pause:
        return
    reporter.maybe_flush(force=True)
    emit(status_event(STATUS_PAUSED, step=step, epoch=epoch, elapsed_s=elapsed()))
    while state.pause and not state.stop:
        time.sleep(PAUSE_POLL_S)
        drain_control(control_queue, state)
    if state.stop:
        raise RunStopped
    emit(status_event(STATUS_RUNNING, step=step, epoch=epoch, elapsed_s=elapsed()))


class Reporter:
    """指标聚合与节流上报（docs/02 §5.4）；非有限值一律丢弃（NaN 会污染库与 JSON）。"""

    def __init__(self, event_queue: Any) -> None:
        self.event_queue = event_queue
        self.buffer: list[dict[str, Any]] = []
        self.last_emit = time.monotonic()

    def add(self, step: int, epoch: int, values: dict[str, Any]) -> None:
        clean: dict[str, float] = {}
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean[name] = number
        self.buffer.append({"step": step, "epoch": epoch, "values": clean})

    def maybe_flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        if not force and len(self.buffer) < REPORT_STEPS:
            if time.monotonic() - self.last_emit < REPORT_INTERVAL_S:
                return
        self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.event_queue.put({"type": EVENT_METRICS, "points": self.buffer})
        self.buffer = []
        self.last_emit = time.monotonic()


def failure_payload(
    exc: BaseException, *, step: int, epoch: int, elapsed_s: float
) -> list[dict[str, Any]]:
    """异常 → `error` + `status(failed)` 两条事件（docs/02 §5.5）。"""
    if isinstance(exc, RunFailed) or (
        hasattr(exc, "message_key") and hasattr(exc, "code")
    ):
        payload = {
            "type": EVENT_ERROR,
            "code": getattr(exc, "code", "runner_failed"),
            "message_key": exc.message_key,
            "args": getattr(exc, "error_args", {}) or {},
            "node_id": getattr(exc, "node_id", None),
            "detail": getattr(exc, "detail", ""),
        }
    else:  # 兜底：任何异常都要变成可读错误事件
        payload = {
            "type": EVENT_ERROR,
            "code": "runner_failed",
            "message_key": "errors.run.failed",
            "args": {"type": type(exc).__name__},
            "detail": traceback.format_exc(limit=6)[-2000:],
        }
    return [payload, status_event(STATUS_FAILED, step=step, epoch=epoch, elapsed_s=elapsed_s)]
