"""ML 训练子进程（契约见 docs/02 §13.2 / §13.4 / §13.5）：numpy 自实现、step 级指标 + boundary 单流快照。"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np

from app.algos import ml as algo_ml
from app.algos import spec as algo_spec
from app.datasets import registry, synth2d
from app.probes import encode as probe_encode
from app.probes.writer import SnapshotWriter
from app.runners import base

log = logging.getLogger(__name__)

GRID_SIDE = 96          # 决策网格边长（docs/02 §13.5）
BOUNDARY_MARGIN = 0.05  # 训练集包围盒外扩比例


def _resolve_dataset(raw_id: Any) -> str:
    """ML 只吃 `synth2d` 的四个二维合成集（docs/02 §13.3）。"""
    spec = registry.get_spec(raw_id) if isinstance(raw_id, str) else None
    if spec is None:
        raise base.RunFailed("dataset_not_found", "errors.dataset.notFound", {"id": raw_id})
    if spec.loader != "synth2d":
        raise base.RunFailed(
            "dataset_kind_mismatch",
            "errors.dataset.kindMismatch",
            {"id": spec.id, "kind": algo_spec.KIND_ML, "expected": "synth2d"},
        )
    return spec.id


class BoundaryProbe:
    """ML 单流探针（`model:boundary`）：把当前决策场编码成快照（docs/02 §13.4 / §13.5）。"""

    def __init__(
        self,
        probes: list[dict[str, Any]],
        model: algo_ml.MLModel,
        dataset_id: str,
        *,
        run_id: str,
        snapshots_dir: str,
        emit: Any,
    ) -> None:
        self.model = model
        node_id, kind = algo_spec.SINGLE_STREAM[algo_spec.KIND_ML]
        self.node_id, self.kind = node_id, kind
        self.every_n = int(probes[0]["every_n_steps"]) if probes else 0
        x0, x1, y0, y1 = synth2d.bounds(dataset_id, BOUNDARY_MARGIN)
        self.x_range = (x0, x1)
        self.y_range = (y0, y1)
        grid_x, grid_y = np.meshgrid(
            np.linspace(x0, x1, GRID_SIDE), np.linspace(y0, y1, GRID_SIDE), indexing="xy"
        )
        # 行序即 y 升序（(0,0) 在左下），与前端画布的坐标约定一致
        self.points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        self.writer = SnapshotWriter(run_id=run_id, snapshots_dir=snapshots_dir, emit=emit)

    def active(self) -> bool:
        return self.every_n > 0

    def maybe_capture(self, step: int, epoch: int) -> None:
        if self.every_n <= 0 or step % self.every_n != 0:
            return
        values, mode = self.model.boundary(self.points)
        grid = np.asarray(values, dtype=np.float32).reshape(GRID_SIDE, GRID_SIDE)
        payload = probe_encode.encode_boundary(
            grid,
            mode=mode,
            x_range=self.x_range,
            y_range=self.y_range,
            algo=self.model.algo,
            meta=self.model.boundary_meta(),
        )
        self.writer.write(
            node_id=self.node_id, kind=self.kind, step=step, epoch=epoch, payload=payload
        )


def run_training(config: dict[str, Any], event_queue: Any, control_queue: Any) -> int:
    """子进程入口主体；返回退出码（0 正常/停止，1 失败）。"""
    start = time.monotonic()
    reporter = base.Reporter(event_queue)
    step = 0
    epoch = 0
    best_metric: float | None = None

    def elapsed() -> float:
        return time.monotonic() - start

    event_queue.put(
        {
            "type": base.EVENT_LOG,
            "level": "info",
            "key": "log.run.starting",
            "args": {"device": "cpu"},
        }
    )

    try:
        try:
            algo = algo_spec.parse_spec(config.get("graph"))
        except algo_spec.AlgoSpecError as exc:
            raise base.RunFailed(exc.code, exc.message_key, exc.error_args, detail=exc.detail) from exc
        if algo.kind != algo_spec.KIND_ML:
            raise base.RunFailed(
                "invalid_algo", "errors.algo.badSpec", {"param": "kind"}, detail=str(algo.kind)
            )

        dataset_id = _resolve_dataset(config.get("dataset_id"))
        train_x, train_y, val_x, val_y = synth2d.split_arrays(dataset_id)
        seed = int(config.get("seed") or 0)
        model = algo_ml.build(algo.algo, train_x, train_y, algo.params, seed)
        delay_s = float(algo.params.get("render_delay_ms") or 0) / 1000.0
        probe = BoundaryProbe(
            config.get("probes") or [],
            model,
            dataset_id,
            run_id=config["run_id"],
            snapshots_dir=config.get("snapshots_dir"),
            emit=event_queue.put,
        )

        event_queue.put(
            base.status_event(
                base.STATUS_RUNNING,
                step=0,
                epoch=0,
                elapsed_s=elapsed(),
                device="cpu",
                planned_steps=model.total_steps,
                steps_per_epoch=model.steps_per_epoch,
            )
        )
        event_queue.put(
            {
                "type": base.EVENT_LOG,
                "level": "info",
                "key": "log.run.dataset",
                "args": {"train": len(train_x), "val": len(val_x)},
            }
        )
        if probe.active():
            probe.writer.log_attached(1)

        state = base.ControlState(lr=float(algo.params.get("lr") or 0.0))
        while not model.finished:
            epoch += 1
            if state.batch_size is not None:
                size, state.batch_size = state.batch_size, None
                model.set_batch_size(size)
                event_queue.put(
                    {
                        "type": base.EVENT_LOG,
                        "level": "info",
                        "key": "log.run.batchSize",
                        "args": {"batch_size": size},
                    }
                )
                event_queue.put(
                    base.status_event(
                        base.STATUS_RUNNING,
                        step=step,
                        epoch=epoch,
                        elapsed_s=elapsed(),
                        planned_steps=model.total_steps,
                        steps_per_epoch=model.steps_per_epoch,
                    )
                )

            for _ in range(model.steps_per_epoch):
                if model.finished:
                    break
                base.drain_control(control_queue, state)
                if state.stop:
                    raise base.RunStopped
                base.handle_pause(
                    control_queue,
                    state,
                    reporter=reporter,
                    emit=event_queue.put,
                    step=step,
                    epoch=epoch,
                    elapsed=elapsed,
                )
                if state.lr_dirty:
                    model.set_lr(state.lr)
                    state.lr_dirty = False

                values = model.step()
                if not values:
                    break
                step += 1
                reporter.add(step, epoch, values)
                probe.maybe_capture(step, epoch)
                reporter.maybe_flush()
                if delay_s:
                    time.sleep(delay_s)

            # epoch 末：树 / 森林只走一轮，val_* 因此正好挂在最后一步（docs/02 §13.2）
            val_values = {
                "val_loss": model.loss_on(val_x, val_y),
                "val_acc": model.acc_on(val_x, val_y),
            }
            reporter.add(step, epoch, val_values)
            reporter.flush()
            val_acc = val_values["val_acc"]
            if best_metric is None or val_acc > best_metric:
                best_metric = val_acc
            event_queue.put(
                base.status_event(
                    base.STATUS_RUNNING,
                    step=step,
                    epoch=epoch,
                    elapsed_s=elapsed(),
                    best_metric=best_metric,
                    planned_steps=model.total_steps,
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

    except base.RunStopped:
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
    except Exception as exc:  # noqa: BLE001 - 兜底：任何异常都要变成可读错误事件
        reporter.flush()
        for event in base.failure_payload(exc, step=step, epoch=epoch, elapsed_s=elapsed()):
            event_queue.put(event)
        return 1


def entry(config: dict[str, Any], event_queue: Any, control_queue: Any) -> None:
    """multiprocessing 入口（spawn 要求模块级函数）；纯 numpy，冷启动不加载 torch。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    exit_code = run_training(config, event_queue, control_queue)
    event_queue.put({"type": base.EVENT_BYE, "exit_code": exit_code})
    sys.exit(exit_code)
