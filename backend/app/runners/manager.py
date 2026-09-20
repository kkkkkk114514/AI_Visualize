"""run 生命周期（契约见 docs/02 §2 / §5 / §8.5）：子进程 + reader 线程 + 状态机 + 批量落库 + WS 广播。"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import queue
import threading
import time
from datetime import datetime
from typing import Any

from app import config
from app.api.ws import hub
from app.datasets import registry
from app.graph.ir import IRParseError, GraphIR, parse_graph, validate_graph
from app.metrics import METRIC_NAMES
from app.probes import hooks as probe_hooks
from app.runners import base, torch_runner
from app.store import db, files

log = logging.getLogger(__name__)


class RunError(Exception):
    """REST 层可翻译的错误：HTTP 状态 + code + i18n key。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message_key: str,
        error_args: dict[str, Any] | None = None,
        detail: str = "",
    ):
        super().__init__(detail or message_key)
        self.status_code = status_code
        self.code = code
        self.message_key = message_key
        self.error_args = error_args or {}
        self.detail = detail


class RunProcess:
    """活动 run 的运行时视图（仅事件循环线程读写）。"""

    def __init__(self, run_id: str, process: Any, event_queue: Any, control_queue: Any):
        self.run_id = run_id
        self.process = process
        self.event_queue = event_queue
        self.control_queue = control_queue
        self.status = base.STATUS_CREATED
        self.step = 0
        self.epoch = 0
        self.device: str | None = None
        self.planned_steps = 0
        self.best_metric: float | None = None
        self.error: str | None = None
        self.exit_code: int | None = None
        self.finished_at: str | None = None
        # 终态事件已处理：此后不再接受该 run 的事件，活动槽位也已释放
        self.finalized = False
        self.stop_requested = False
        self.watchdog_warned = False
        self.started_monotonic = time.monotonic()
        self.last_event_at = time.monotonic()
        self.metrics_buffer: list[tuple[int, int | None, str, float]] = []
        self.tasks: list[asyncio.Task] = []

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_monotonic

    def alive(self) -> bool:
        return bool(self.process.is_alive())


def _default_name(graph: GraphIR) -> str:
    text: str | None = None
    if isinstance(graph.name, dict):
        text = graph.name.get("zh") or graph.name.get("en")
    elif isinstance(graph.name, str):
        text = graph.name
    stamp = datetime.now().strftime("%H:%M:%S")
    return f"{text or graph.id} · {stamp}"


class RunnerManager:
    def __init__(self) -> None:
        self._current: RunProcess | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def current(self) -> RunProcess | None:
        return self._current

    def is_busy(self) -> bool:
        return self._current is not None

    def live_info(self, run_id: str) -> dict[str, Any] | None:
        run = self._current
        if run is None or run.run_id != run_id:
            return None
        return {
            "status": run.status,
            "step": run.step,
            "epoch": run.epoch,
            "device": run.device,
            "elapsed_s": round(run.elapsed_s, 3),
            "planned_steps": run.planned_steps,
            "best_metric": run.best_metric,
        }

    def pending_metrics(self, run_id: str) -> list[tuple[int, int | None, str, float]]:
        run = self._current
        if run is None or run.run_id != run_id:
            return []
        return list(run.metrics_buffer)

    # ------------------------------------------------------------- 启动

    async def start(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._current is not None:
            raise RunError(
                409, "errors.run.busy", "errors.run.busy", {"run_id": self._current.run_id}
            )

        model_id = body.get("model_id")
        graph_payload = body.get("graph")
        if model_id:
            from app.api.models import read_preset

            preset = read_preset(str(model_id))
            if preset is None:
                raise RunError(404, "errors.model.notFound", "errors.model.notFound", {"id": model_id})
            graph_payload = preset
        if not isinstance(graph_payload, dict):
            raise RunError(400, "errors.run.needModel", "errors.run.needModel")

        dataset_id = body.get("dataset_id")
        spec = registry.get_spec(dataset_id) if isinstance(dataset_id, str) else None
        if spec is None:
            raise RunError(
                404, "errors.dataset.notFound", "errors.dataset.notFound", {"id": dataset_id}
            )
        if not registry.is_cached(spec):
            raise RunError(
                409,
                "errors.dataset.notCached",
                "errors.dataset.notCached",
                {"id": spec.id, "path": str(registry.raw_dir(spec))},
            )

        try:
            graph = parse_graph(graph_payload)
        except IRParseError as exc:
            raise RunError(422, "invalid_ir", exc.message_key, exc.error_args, exc.detail) from exc
        issues, _ = validate_graph(graph)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            first = errors[0].to_dict()
            raise RunError(
                422,
                "invalid_ir",
                first["message_key"],
                first.get("args") or {},
                detail=str(first["code"]),
            )

        try:
            hyper = base.resolve_hyperparams(graph.hyper_defaults, body.get("hyperparams"))
        except base.HyperError as exc:
            raise RunError(422, exc.message_key, exc.message_key, exc.error_args, exc.detail) from exc

        try:
            probe_specs = probe_hooks.resolve_probes(body.get("probes"), graph)
        except probe_hooks.ProbeError as exc:
            raise RunError(
                422, "invalid_probe", exc.message_key, exc.error_args, exc.detail
            ) from exc

        seed_value = body.get("seed")
        seed = (
            int(seed_value)
            if isinstance(seed_value, (int, float)) and not isinstance(seed_value, bool)
            else config.DEFAULT_SEED
        )
        raw_name = body.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else _default_name(graph)

        run_id = db.new_run_id()
        files.ensure_run_dir(run_id)
        probe_payload = [spec.to_dict() for spec in probe_specs]
        db.insert_run(
            {
                "id": run_id,
                "name": name,
                "kind": graph.kind or "dl",
                "model_id": str(model_id) if model_id else None,
                "graph_json": json.dumps(graph.to_dict(), ensure_ascii=False),
                "dataset_id": spec.id,
                "hyperparams_json": json.dumps(hyper, ensure_ascii=False),
                "probes_json": json.dumps(probe_payload, ensure_ascii=False),
                "status": base.STATUS_CREATED,
                "device": None,
                "seed": seed,
                "created_at": db.now_iso(),
                "started_at": db.now_iso(),
                "finished_at": None,
                "error": None,
                "best_metric": None,
                "total_steps": 0,
            }
        )

        context = multiprocessing.get_context("spawn")
        event_queue = context.Queue()
        control_queue = context.Queue()
        child_config = {
            "run_id": run_id,
            "graph": graph.to_dict(),
            "dataset_id": spec.id,
            "hyperparams": hyper,
            "seed": seed,
            "checkpoint_path": str(files.checkpoint_path(run_id)),
            "snapshots_dir": str(files.snapshot_dir(run_id)),
            "probes": probe_payload,
        }
        try:
            process = context.Process(
                target=torch_runner.entry,
                args=(child_config, event_queue, control_queue),
                name=f"run-{run_id}",
                daemon=True,
            )
            process.start()
        except Exception as exc:  # noqa: BLE001 - 启动失败要落库并回 500
            db.update_run(
                run_id,
                status=base.STATUS_FAILED,
                finished_at=db.now_iso(),
                error=f"spawn_failed: {type(exc).__name__}: {exc}",
            )
            raise RunError(
                500, "errors.run.spawnFailed", "errors.run.spawnFailed", detail=str(exc)
            ) from exc

        run = RunProcess(run_id, process, event_queue, control_queue)
        self._current = run
        run.tasks.append(asyncio.create_task(self._periodic_loop(run)))
        threading.Thread(target=self._reader, args=(run,), name=f"reader-{run_id}", daemon=True).start()
        log.info("run %s 已启动（dataset=%s, device 待定）", run_id, spec.id)
        record = db.get_run(run_id)
        await hub.broadcast(
            {
                "type": base.WIRE_STATUS,
                "run_id": run_id,
                "status": base.STATUS_CREATED,
                "step": 0,
                "epoch": 0,
                "device": None,
                "elapsed_s": 0.0,
            }
        )
        return record if record is not None else {"id": run_id, "status": base.STATUS_CREATED}

    # ------------------------------------------------------------- 控制

    async def control(self, run_id: str, action: str, value: Any = None) -> dict[str, Any]:
        if action not in base.CONTROL_ACTIONS:
            raise RunError(400, "errors.run.badAction", "errors.run.badAction", {"action": action})
        run = self._current
        if run is None or run.run_id != run_id:
            if db.get_run(run_id) is None:
                raise RunError(404, "errors.run.notFound", "errors.run.notFound", {"id": run_id})
            raise RunError(409, "errors.run.notActive", "errors.run.notActive", {"id": run_id})

        if action == base.CTRL_KILL:
            run.stop_requested = True
            log.info("run %s 强制终止（terminate）", run_id)
            run.process.terminate()
            return {"run_id": run_id, "action": action, "status": run.status}

        if action == base.CTRL_STOP:
            run.stop_requested = True
            log.info("run %s 收到停止请求（最多 %.1fs 后强制终止）", run_id, config.RUN_STOP_GRACE_S)
            run.control_queue.put({"action": base.CTRL_STOP})
            run.tasks.append(asyncio.create_task(self._enforce_stop(run)))
            return {"run_id": run_id, "action": action, "status": run.status}

        if action == base.CTRL_PAUSE and run.status != base.STATUS_RUNNING:
            raise RunError(409, "errors.run.conflict", "errors.run.conflict", {"status": run.status})
        if action == base.CTRL_RESUME and run.status != base.STATUS_PAUSED:
            raise RunError(409, "errors.run.conflict", "errors.run.conflict", {"status": run.status})

        if action == base.CTRL_SET_LR:
            lr = _positive_float(value)
            if lr is None or lr > 1.0:
                raise RunError(422, "errors.run.invalidHyper", "errors.run.invalidHyper", {"param": "lr"})
            run.control_queue.put({"action": action, "value": lr})
        elif action == base.CTRL_SET_BATCH_SIZE:
            size = _positive_int(value)
            if size is None or size > 4096:
                raise RunError(
                    422, "errors.run.invalidHyper", "errors.run.invalidHyper", {"param": "batch_size"}
                )
            run.control_queue.put({"action": action, "value": size})
        else:
            run.control_queue.put({"action": action})

        log.info("run %s 控制命令：%s (%s)", run_id, action, value)
        return {"run_id": run_id, "action": action, "status": run.status}

    async def _enforce_stop(self, run: RunProcess) -> None:
        await asyncio.sleep(config.RUN_STOP_GRACE_S)
        if run.finalized or not run.alive():
            return
        log.warning("run %s 超过 %.1fs 未退出，terminate", run.run_id, config.RUN_STOP_GRACE_S)
        run.process.terminate()
        await asyncio.sleep(config.RUN_KILL_GRACE_S)
        if not run.finalized and run.alive():
            log.warning("run %s 仍未退出，kill", run.run_id)
            run.process.kill()

    async def shutdown(self) -> None:
        run = self._current
        if run is None:
            return
        log.info("服务关闭：停止活动 run %s", run.run_id)
        run.stop_requested = True
        run.control_queue.put({"action": base.CTRL_STOP})
        deadline = time.monotonic() + config.RUN_STOP_GRACE_S
        while time.monotonic() < deadline and run.alive():
            await asyncio.sleep(0.1)
        if run.alive():
            run.process.terminate()
            await asyncio.sleep(config.RUN_KILL_GRACE_S)
        if run.alive():
            run.process.kill()
        for task in run.tasks:
            task.cancel()
        self._flush(run, force=True)
        db.update_run(
            run.run_id,
            status=base.STATUS_STOPPED,
            finished_at=db.now_iso(),
            total_steps=run.step,
            best_metric=run.best_metric,
            device=run.device,
        )
        self._current = None

    # ------------------------------------------------ 子进程事件 → 落库/广播

    def _reader(self, run: RunProcess) -> None:
        loop = self._loop
        if loop is None:
            return
        dead_since: float | None = None
        while True:
            try:
                event = run.event_queue.get(timeout=0.1)
                dead_since = None
            except queue.Empty:
                if run.alive():
                    continue
                # 进程已退出但队列可能还在 flush（mp.Queue 的 feeder 线程），留 0.5s 收尾
                if dead_since is None:
                    dead_since = time.monotonic()
                elif time.monotonic() - dead_since > 0.5:
                    break
                continue
            except Exception:  # noqa: BLE001 - 队列异常（进程已亡）直接收尾
                break
            if not self._dispatch(loop, self._on_event, run, event):
                break
            if event.get("type") == base.EVENT_BYE:
                break
        run.process.join(timeout=5)
        self._dispatch(loop, self._on_finished, run)

    def _dispatch(self, loop: asyncio.AbstractEventLoop, callback: Any, *args: Any) -> bool:
        """从 reader 线程投递回调；事件循环已关闭（服务退出/测试结束）时静默放弃。"""
        if loop.is_closed():
            return False
        try:
            loop.call_soon_threadsafe(callback, *args)
            return True
        except RuntimeError:  # noqa: BLE001 - 循环正在关闭
            return False

    def _on_event(self, run: RunProcess, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == base.EVENT_BYE:
            run.exit_code = int(event.get("exit_code") or 0)
            return
        if run.finalized:
            return
        run.last_event_at = time.monotonic()
        run.watchdog_warned = False

        if event_type == base.EVENT_STATUS:
            run.status = str(event.get("status") or run.status)
            run.step = int(event.get("step") or run.step)
            run.epoch = int(event.get("epoch") or run.epoch)
            if event.get("device"):
                run.device = str(event["device"])
            if event.get("planned_steps"):
                run.planned_steps = int(event["planned_steps"])
            if event.get("best_metric") is not None:
                run.best_metric = float(event["best_metric"])
            if run.status in (base.STATUS_FINISHED, base.STATUS_STOPPED, base.STATUS_FAILED):
                run.finished_at = run.finished_at or db.now_iso()
                run.finalized = True
                self._flush(run, force=True)
                db.update_run(
                    run.run_id,
                    status=run.status,
                    device=run.device,
                    finished_at=run.finished_at,
                )
                # 终态即释放活动槽位：后续 REST 控制/DELETE 立刻按“已结束”处理
                if self._current is run:
                    self._current = None
            else:
                db.update_run(run.run_id, status=run.status, device=run.device)
            payload = dict(event)
            payload["type"] = base.WIRE_STATUS
            payload["run_id"] = run.run_id
            payload["device"] = run.device
            self._broadcast(payload)
            return

        if event_type == base.EVENT_METRICS:
            for point in event.get("points") or []:
                step = int(point["step"])
                epoch = point.get("epoch")
                epoch_value = int(epoch) if epoch is not None else None
                for name, value in (point.get("values") or {}).items():
                    if name not in METRIC_NAMES:
                        continue
                    run.metrics_buffer.append((step, epoch_value, name, float(value)))
                run.step = max(run.step, step)
                if epoch_value:
                    run.epoch = max(run.epoch, epoch_value)
            if len(run.metrics_buffer) >= config.METRICS_FLUSH_ROWS:
                self._flush(run)
            payload = dict(event)
            payload["run_id"] = run.run_id
            self._broadcast(payload)
            return

        if event_type == base.EVENT_ERROR:
            run.error = json.dumps(
                {key: event.get(key) for key in ("code", "message_key", "args", "node_id", "detail")},
                ensure_ascii=False,
            )[:1000]
            db.update_run(run.run_id, error=run.error)

        if event_type == base.EVENT_PROBE:
            snapshot_id = str(event.get("snapshot_id"))
            db.insert_snapshot(
                {
                    "id": snapshot_id,
                    "run_id": run.run_id,
                    "step": int(event.get("step") or 0),
                    "epoch": event.get("epoch"),
                    "node_id": event.get("node_id"),
                    "kind": event.get("kind"),
                    "shape": event.get("shape") or [],
                    "min": event.get("min"),
                    "max": event.get("max"),
                    "file_path": event.get("file_path") or str(files.snapshot_path(run.run_id, snapshot_id)),
                    "created_at": db.now_iso(),
                }
            )
            # 线上只推元数据（docs/02 §6.3 / §8.2）：payload 由前端按需带 ETag 拉取
            self._broadcast(
                {
                    "type": base.WIRE_PROBE,
                    "run_id": run.run_id,
                    "snapshot_id": snapshot_id,
                    "step": int(event.get("step") or 0),
                    "epoch": event.get("epoch"),
                    "node_id": event.get("node_id"),
                    "kind": event.get("kind"),
                    "shape": event.get("shape") or [],
                }
            )
            return

        if event_type in (base.EVENT_LOG, base.EVENT_ERROR):
            payload = dict(event)
            payload["run_id"] = run.run_id
            self._broadcast(payload)

    def _on_finished(self, run: RunProcess) -> None:
        for task in run.tasks:
            task.cancel()
        self._flush(run, force=True)
        if run.finalized:
            log.info(
                "run %s 子进程已退出（exit=%s, steps=%s, elapsed=%.1fs）",
                run.run_id,
                run.exit_code if run.exit_code is not None else run.process.exitcode,
                run.step,
                run.elapsed_s,
            )
            return

        exit_code = run.exit_code if run.exit_code is not None else run.process.exitcode
        if run.stop_requested:
            final = base.STATUS_STOPPED
        elif exit_code == 0:
            final = base.STATUS_FINISHED
        else:
            final = base.STATUS_FAILED
        if run.status == base.STATUS_FINISHED:
            final = base.STATUS_FINISHED
        run.status = final
        run.finalized = True
        db.update_run(
            run.run_id,
            status=final,
            finished_at=run.finished_at or db.now_iso(),
            total_steps=run.step,
            best_metric=run.best_metric,
            device=run.device,
        )
        log.info(
            "run %s 结束：%s（exit=%s, steps=%s, elapsed=%.1fs）",
            run.run_id,
            final,
            run.exit_code,
            run.step,
            run.elapsed_s,
        )
        self._broadcast(
            {
                "type": base.WIRE_STATUS,
                "run_id": run.run_id,
                "status": final,
                "step": run.step,
                "epoch": run.epoch,
                "device": run.device,
                "elapsed_s": round(run.elapsed_s, 3),
                "best_metric": run.best_metric,
                "exit_code": run.exit_code,
            }
        )
        if self._current is run:
            self._current = None

    async def _periodic_loop(self, run: RunProcess) -> None:
        try:
            while True:
                await asyncio.sleep(config.METRICS_FLUSH_INTERVAL_S)
                self._flush(run)
                self._watchdog(run)
        except asyncio.CancelledError:
            return

    def _flush(self, run: RunProcess, force: bool = False) -> None:
        if run.metrics_buffer:
            db.insert_metrics(run.run_id, run.metrics_buffer)
            run.metrics_buffer = []
        db.update_run(
            run.run_id,
            total_steps=run.step,
            best_metric=run.best_metric,
            device=run.device,
        )

    def _watchdog(self, run: RunProcess) -> None:
        if run.status != base.STATUS_RUNNING or run.watchdog_warned:
            return
        average_epoch = run.elapsed_s / max(run.epoch, 1)
        threshold = max(config.WATCHDOG_MIN_STALE_S, 3 * average_epoch)
        if time.monotonic() - run.last_event_at < threshold:
            return
        run.watchdog_warned = True
        log.warning("run %s 超过 %.1fs 无事件，推送 stale 提示", run.run_id, threshold)
        self._broadcast(
            {
                "type": base.EVENT_ERROR,
                "run_id": run.run_id,
                "code": "stale",
                "message_key": "errors.run.stale",
                "args": {"seconds": round(threshold, 1)},
            }
        )

    def _broadcast(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.ensure_future(hub.broadcast(payload))


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(round(float(value)))
    return number if number > 0 else None


manager = RunnerManager()
