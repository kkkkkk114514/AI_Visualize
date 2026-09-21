"""RL 训练子进程（契约见 docs/02 §13.2 / §13.4 / §13.5）：表格 Q-learning + grid 单流快照。"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from app.algos import rl as algo_rl
from app.algos import spec as algo_spec
from app.datasets import registry
from app.probes import encode as probe_encode
from app.probes.writer import SnapshotWriter
from app.runners import base

log = logging.getLogger(__name__)


def _resolve_dataset(raw_id: Any) -> str:
    """RL 只吃 `gridworld`（环境本身来自 spec.env，数据集只作就绪检查，docs/02 §13.3）。"""
    spec = registry.get_spec(raw_id) if isinstance(raw_id, str) else None
    if spec is None:
        raise base.RunFailed("dataset_not_found", "errors.dataset.notFound", {"id": raw_id})
    if spec.loader != "gridworld":
        raise base.RunFailed(
            "dataset_kind_mismatch",
            "errors.dataset.kindMismatch",
            {"id": spec.id, "kind": algo_spec.KIND_RL, "expected": "gridworld"},
        )
    return spec.id


class GridProbe:
    """RL 单流探针（`agent:grid`）：把 `V(s)` / 策略 / 轨迹编码成快照（docs/02 §13.4 / §13.5）。"""

    def __init__(
        self,
        probes: list[dict[str, Any]],
        model: algo_rl.QLearning,
        *,
        run_id: str,
        snapshots_dir: str,
        emit: Any,
    ) -> None:
        self.model = model
        node_id, kind = algo_spec.SINGLE_STREAM[algo_spec.KIND_RL]
        self.node_id, self.kind = node_id, kind
        self.every_n = int(probes[0]["every_n_steps"]) if probes else 0
        self.writer = SnapshotWriter(run_id=run_id, snapshots_dir=snapshots_dir, emit=emit)

    def active(self) -> bool:
        return self.every_n > 0

    def maybe_capture(self, step: int, epoch: int) -> None:
        if self.every_n <= 0 or step % self.every_n != 0:
            return
        env = self.model.env
        values = self.model.values().reshape(env.height, env.width).copy()
        for x, y in env.obstacles:
            values[y, x] = 0.0
        payload = probe_encode.encode_grid(
            values,
            policy=self.model.policy(),
            meta=self.model.grid_payload_meta(),
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
            spec = algo_spec.parse_spec(config.get("graph"))
        except algo_spec.AlgoSpecError as exc:
            raise base.RunFailed(exc.code, exc.message_key, exc.error_args, detail=exc.detail) from exc
        if spec.kind != algo_spec.KIND_RL:
            raise base.RunFailed(
                "invalid_algo", "errors.algo.badSpec", {"param": "kind"}, detail=str(spec.kind)
            )

        _resolve_dataset(config.get("dataset_id"))
        seed = int(config.get("seed") or 0)
        env = algo_rl.GridWorld(spec.env, spec.params["max_steps"])
        model = algo_rl.QLearning(env, spec.params, seed)
        delay_s = float(spec.params.get("render_delay_ms") or 0) / 1000.0
        probe = GridProbe(
            config.get("probes") or [],
            model,
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
                planned_steps=model.episodes * env.max_steps,
            )
        )
        event_queue.put(
            {
                "type": base.EVENT_LOG,
                "level": "info",
                "key": "log.run.env",
                "args": {
                    "width": env.width,
                    "height": env.height,
                    "obstacles": len(env.obstacles),
                    "rewards": len(env.bonus),
                },
            }
        )
        if probe.active():
            probe.writer.log_attached(1)

        state = base.ControlState(lr=model.alpha)
        while not model.finished:
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
            if state.batch_size is not None:  # RL 是单步更新：批大小无意义（docs/02 §13.4）
                state.batch_size = None
                event_queue.put(
                    {"type": base.EVENT_LOG, "level": "info", "key": "log.run.batchIgnored"}
                )

            values = model.step()
            step += 1
            if "episode_reward" in values:
                epoch = model.episodes_done
                rolling = model.rolling_reward()
                if best_metric is None or rolling > best_metric:
                    best_metric = rolling
            reporter.add(step, epoch, values)
            probe.maybe_capture(step, epoch)
            reporter.maybe_flush()
            if delay_s:
                time.sleep(delay_s)

            if "episode_reward" in values:
                reporter.flush()
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
    base.bootstrap()
    exit_code = run_training(config, event_queue, control_queue)
    event_queue.put({"type": base.EVENT_BYE, "exit_code": exit_code})
    sys.exit(exit_code)
