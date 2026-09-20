"""runner 协议（契约见 docs/02 §5.4 / §5.5）：事件与控制命令常量、超参解析。"""

from __future__ import annotations

from typing import Any

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
