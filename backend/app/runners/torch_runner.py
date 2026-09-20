"""DL 训练子进程（契约见 docs/02 §5.2 / §5.4 / §5.5）：真实前向 + step 级指标 + 控制队列 drain。"""

from __future__ import annotations

import logging
import queue
import sys
import time
import traceback
from collections import deque
from typing import Any

import torch
import torch.nn as nn

from app import config as app_config
from app.datasets import loaders, registry
from app.graph.ir import bind_dataset, parse_graph, validate_graph
from app.graph.module import NodeExecutionError, build_graph_module
from app.probes import hooks as probe_hooks
from app.runners import base
from app.store import files

log = logging.getLogger(__name__)


class RunFailed(Exception):
    """训练前置失败（数据集/图/超参）：带定位信息，转 error 事件。"""

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


class _Stopped(Exception):
    """用户停止：走正常收尾路径（flush + stopped 状态）。"""


class _Control:
    def __init__(self, lr: float) -> None:
        self.pause = False
        self.stop = False
        self.lr = lr
        self.lr_dirty = False
        self.batch_size: int | None = None


class _Reporter:
    """指标聚合与节流上报（docs/02 §5.4）。"""

    def __init__(self, event_queue: Any) -> None:
        self.event_queue = event_queue
        self.buffer: list[dict[str, Any]] = []
        self.last_emit = time.monotonic()

    def add(self, step: int, epoch: int, values: dict[str, Any]) -> None:
        self.buffer.append({"step": step, "epoch": epoch, "values": values})

    def maybe_flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        if not force and len(self.buffer) < base.REPORT_STEPS:
            if time.monotonic() - self.last_emit < base.REPORT_INTERVAL_S:
                return
        self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.event_queue.put({"type": base.EVENT_METRICS, "points": self.buffer})
        self.buffer = []
        self.last_emit = time.monotonic()


def _drain(control_queue: Any, state: _Control) -> None:
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return
        action = command.get("action")
        if action == base.CTRL_PAUSE:
            state.pause = True
        elif action == base.CTRL_RESUME:
            state.pause = False
        elif action == base.CTRL_STOP:
            state.stop = True
        elif action == base.CTRL_KILL:
            raise SystemExit(0)
        elif action == base.CTRL_SET_LR:
            state.lr = float(command.get("value"))
            state.lr_dirty = True
        elif action == base.CTRL_SET_BATCH_SIZE:
            state.batch_size = int(command.get("value"))
        else:
            log.warning("未知控制命令：%r", command)


def _build_optimizer(model: nn.Module, hyper: dict[str, Any]) -> torch.optim.Optimizer:
    if hyper["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=hyper["lr"])
    if hyper["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=hyper["lr"])
    return torch.optim.SGD(model.parameters(), lr=hyper["lr"], momentum=0.9)


def _batch_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    return float((output.argmax(dim=-1) == target).to(torch.float32).mean().item())


def _is_sequence(meta: dict[str, Any]) -> bool:
    """字符级语言模型：logits (B,T,V) / target (B,T)，loss 按 token 展平。"""
    return meta.get("loader") == "text_char"


def _loss_inputs(
    output: torch.Tensor, target: torch.Tensor, sequence: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # CrossEntropyLoss 的类别维固定在第 1 维，序列 logits 需从 (B,T,V) 展平成 (B*T,V)
    if not sequence:
        return output, target
    return output.reshape(-1, output.shape[-1]), target.reshape(-1)


def _evaluate(
    model: nn.Module,
    loader: Any,
    loss_fn: nn.Module,
    device: torch.device,
    meta: dict[str, Any],
    classification: bool,
) -> tuple[float, float | None]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    sequence = _is_sequence(meta)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = loaders.normalize(batch_x.to(device), meta)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            target = batch_y if classification else batch_y.to(torch.float32)
            loss = loss_fn(*_loss_inputs(output, target, sequence))
            # 序列任务（语言模型）的 target 是 (B,T)，loss 与准确率按 token 数加权
            elements = int(batch_y.numel()) if classification else int(batch_x.shape[0])
            total_loss += float(loss.item()) * elements
            if classification:
                correct += int((output.argmax(dim=-1) == batch_y).sum().item())
            count += elements
    model.train()
    if count == 0:
        return 0.0, (0.0 if classification else None)
    return total_loss / count, (correct / count if classification else None)


def run_training(config: dict[str, Any], event_queue: Any, control_queue: Any) -> int:
    """子进程入口主体；返回退出码（0 正常/停止，1 失败）。"""
    run_id = config["run_id"]
    hyper = config["hyperparams"]
    start = time.monotonic()
    device = torch.device(
        config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    reporter = _Reporter(event_queue)
    step = 0
    epoch = 0
    best_metric: float | None = None

    def elapsed() -> float:
        return time.monotonic() - start

    # 冷启动（torch 导入 + 首次 optimizer 构建）在 CPU 上约 2~5s，先报一声让界面确认子进程已拉起
    event_queue.put(
        {
            "type": base.EVENT_LOG,
            "level": "info",
            "key": "log.run.starting",
            "args": {"device": device.type},
        }
    )

    try:
        spec = registry.get_spec(config["dataset_id"])
        if spec is None:
            raise RunFailed(
                "dataset_not_found",
                "errors.dataset.notFound",
                {"id": config["dataset_id"]},
            )
        if not registry.is_cached(spec):
            raise RunFailed(
                "dataset_not_cached",
                "errors.dataset.notCached",
                {"id": spec.id, "path": str(registry.raw_dir(spec))},
            )
        meta = registry.meta(spec)

        graph = parse_graph(config["graph"])
        issues, params_by_node = validate_graph(graph)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            first = errors[0].to_dict()
            raise RunFailed(
                "invalid_ir",
                first["message_key"],
                first.get("args") or {},
                detail=f"{first['code']} on {first.get('node_id')}",
                node_id=first.get("node_id"),
            )
        bind_dataset(graph, params_by_node, meta)

        torch.manual_seed(config.get("seed") or 0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.get("seed") or 0)

        model = build_graph_module(graph, params_by_node).to(device)
        classification = hyper["loss"] == "cross_entropy"
        loss_fn: nn.Module = nn.CrossEntropyLoss() if classification else nn.MSELoss()

        train_loader, val_loader, info = loaders.build_loaders(
            spec,
            batch_size=hyper["batch_size"],
            train_size=hyper["train_size"],
            val_size=hyper["val_size"],
            seed=config.get("seed") or 0,
        )
        if int(info["steps_per_epoch"]) == 0:
            raise RunFailed("empty_dataset", "errors.run.emptyDataset", {"id": spec.id})
        planned_steps = int(hyper["epochs"]) * int(info["steps_per_epoch"])

        # 节点子模块在首次前向时才构建（D3），先用一个真实批次物化参数，optimizer 才有参数可优化
        warmup_x, _ = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            model(loaders.normalize(warmup_x.to(device), meta))
        model.train()
        total_params = sum(parameter.numel() for parameter in model.parameters())
        optimizer = _build_optimizer(model, hyper)

        event_queue.put(
            base.status_event(
                base.STATUS_RUNNING,
                step=0,
                epoch=0,
                elapsed_s=elapsed(),
                device=device.type,
                total_params=total_params,
                planned_steps=planned_steps,
                steps_per_epoch=info["steps_per_epoch"],
            )
        )
        event_queue.put(
            {
                "type": base.EVENT_LOG,
                "level": "info",
                "key": "log.run.dataset",
                "args": {"train": info["train_samples"], "val": info["val_samples"]},
            }
        )
        if info.get("vocab_size"):
            event_queue.put(
                {
                    "type": base.EVENT_LOG,
                    "level": "info",
                    "key": "log.run.textCorpus",
                    "args": {"vocab_size": info["vocab_size"], "seq_len": info.get("seq_len")},
                }
            )

        state = _Control(lr=float(hyper["lr"]))
        batch_size = int(hyper["batch_size"])
        step_times: deque[float] = deque(maxlen=base.THROUGHPUT_WINDOW)
        sequence = _is_sequence(meta)
        probe_engine = probe_hooks.ProbeEngine(
            probe_hooks.resolve_probes(config.get("probes"), graph),
            model,
            {node.id: node.type for node in graph.nodes},
            run_id=run_id,
            snapshots_dir=config.get("snapshots_dir") or files.snapshot_dir(run_id),
            emit=event_queue.put,
        )
        if probe_engine.active():
            event_queue.put(
                {
                    "type": base.EVENT_LOG,
                    "level": "info",
                    "key": "log.probe.attached",
                    "args": {"n": len(probe_engine.spec_payload())},
                }
            )
        model.train()

        for epoch in range(1, int(hyper["epochs"]) + 1):
            if state.batch_size and state.batch_size != batch_size:
                batch_size = state.batch_size
                state.batch_size = None
                train_loader, _, info = loaders.build_loaders(
                    spec,
                    batch_size=batch_size,
                    train_size=hyper["train_size"],
                    val_size=hyper["val_size"],
                    seed=config.get("seed") or 0,
                )
                event_queue.put(
                    {
                        "type": base.EVENT_LOG,
                        "level": "info",
                        "key": "log.run.batchSize",
                        "args": {"batch_size": batch_size},
                    }
                )

            for batch_x, batch_y in train_loader:
                _drain(control_queue, state)
                if state.stop:
                    raise _Stopped
                if state.pause:
                    reporter.maybe_flush(force=True)
                    event_queue.put(
                        base.status_event(
                            base.STATUS_PAUSED, step=step, epoch=epoch, elapsed_s=elapsed()
                        )
                    )
                    while state.pause and not state.stop:
                        time.sleep(base.PAUSE_POLL_S)
                        _drain(control_queue, state)
                    if state.stop:
                        raise _Stopped
                    event_queue.put(
                        base.status_event(
                            base.STATUS_RUNNING, step=step, epoch=epoch, elapsed_s=elapsed()
                        )
                    )

                if state.lr_dirty:
                    for group in optimizer.param_groups:
                        group["lr"] = state.lr
                    state.lr_dirty = False

                tick = time.perf_counter()
                batch_x = loaders.normalize(batch_x.to(device), meta)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                # 探针：只在采样步挂 hook，前向一结束立即摘除（docs/02 §6.1）
                probe_engine.begin_step(step + 1, epoch)
                try:
                    output = model(batch_x)
                finally:
                    probe_engine.end_step()
                loss = loss_fn(*_loss_inputs(output, batch_y, sequence))
                loss.backward()
                if hyper["grad_clip"] > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float(hyper["grad_clip"])
                        ).item()
                    )
                else:
                    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                    grad_norm = (
                        float(torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()) if grads else 0.0
                    )
                optimizer.step()
                step_times.append(max(time.perf_counter() - tick, 1e-6))
                step += 1

                values: dict[str, Any] = {
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": grad_norm,
                    "throughput": round(batch_size * len(step_times) / sum(step_times), 2),
                }
                if classification:
                    values["acc"] = _batch_accuracy(output, batch_y)
                if device.type == "cuda" and step % 50 == 0:
                    values["vram_mb"] = round(torch.cuda.max_memory_allocated() / (1 << 20), 1)
                reporter.add(step, epoch, values)
                reporter.maybe_flush()

            val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device, meta, classification)
            val_values: dict[str, Any] = {"val_loss": val_loss}
            if val_acc is not None:
                val_values["val_acc"] = val_acc
            reporter.add(step, epoch, val_values)
            reporter.flush()

            improved = best_metric is None or (
                val_acc > best_metric if val_acc is not None else val_loss < best_metric
            )
            if improved:
                best_metric = val_acc if val_acc is not None else val_loss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "best_metric": best_metric,
                    },
                    config["checkpoint_path"],
                )
                event_queue.put(
                    {
                        "type": base.EVENT_LOG,
                        "level": "info",
                        "key": "log.run.bestSaved",
                        "args": {"epoch": epoch, "value": round(best_metric, 4)},
                    }
                )
            event_queue.put(
                base.status_event(
                    base.STATUS_RUNNING,
                    step=step,
                    epoch=epoch,
                    elapsed_s=elapsed(),
                    best_metric=best_metric,
                )
            )

        reporter.flush()
        event_queue.put(
            {
                "type": base.EVENT_LOG,
                "level": "info",
                "key": "log.run.finished",
                "args": {"epochs": epoch, "elapsed_s": round(elapsed(), 1)},
            }
        )
        event_queue.put(
            base.status_event(
                base.STATUS_FINISHED,
                step=step,
                epoch=epoch,
                elapsed_s=elapsed(),
                best_metric=best_metric,
            )
        )
        return 0

    except _Stopped:
        reporter.flush()
        event_queue.put(
            base.status_event(
                base.STATUS_STOPPED,
                step=step,
                epoch=epoch,
                elapsed_s=elapsed(),
                best_metric=best_metric,
            )
        )
        return 0
    except RunFailed as exc:
        reporter.flush()
        event_queue.put(
            {
                "type": base.EVENT_ERROR,
                "code": exc.code,
                "message_key": exc.message_key,
                "args": exc.error_args,
                "node_id": exc.node_id,
                "detail": exc.detail,
            }
        )
        event_queue.put(
            base.status_event(base.STATUS_FAILED, step=step, epoch=epoch, elapsed_s=elapsed())
        )
        return 1
    except NodeExecutionError as exc:
        reporter.flush()
        event_queue.put(
            {
                "type": base.EVENT_ERROR,
                "code": exc.code,
                "message_key": exc.message_key,
                "args": exc.error_args,
                "node_id": exc.node_id,
                "detail": exc.detail,
            }
        )
        event_queue.put(
            base.status_event(base.STATUS_FAILED, step=step, epoch=epoch, elapsed_s=elapsed())
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - 兜底：任何异常都要变成可读错误事件
        reporter.flush()
        event_queue.put(
            {
                "type": base.EVENT_ERROR,
                "code": "runner_failed",
                "message_key": "errors.run.failed",
                "args": {"type": type(exc).__name__},
                "detail": traceback.format_exc(limit=6)[-2000:],
            }
        )
        event_queue.put(
            base.status_event(base.STATUS_FAILED, step=step, epoch=epoch, elapsed_s=elapsed())
        )
        return 1


def entry(config: dict[str, Any], event_queue: Any, control_queue: Any) -> None:
    """multiprocessing 入口（spawn 要求模块级函数）。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # 形参 config（run 配置）遮蔽了同名模块，这里用别名取进程级设置
    torch.set_num_threads(app_config.TORCH_THREADS)
    exit_code = run_training(config, event_queue, control_queue)
    event_queue.put({"type": base.EVENT_BYE, "exit_code": exit_code})
    sys.exit(exit_code)
