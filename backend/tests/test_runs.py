"""M2 验收测试：run 生命周期（启动 → 暂停 → 恢复 → 改 lr → 停止 → 落库回放）。"""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.datasets import registry
from app.metrics import downsample_series, lttb
from app.store import db, files

# 小规模超参：CPU 上单次 run 约 3~4s（子进程冷启动另计）
SMALL_HYPER: dict[str, Any] = {
    "optimizer": "adam",
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 4,
    "loss": "cross_entropy",
    "grad_clip": 5.0,
    "train_size": 3200,
    "val_size": 256,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    if not registry.is_cached(registry.MNIST):
        pytest.skip("MNIST 未缓存，跳过训练类测试")
    tmp = tmp_path_factory.mktemp("data")
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "DATA_DIR", tmp)
    patch.setattr(config, "RUNS_DIR", tmp / "runs")
    patch.setattr(config, "DB_PATH", tmp / "app.db")
    with TestClient(main.app) as test_client:
        yield test_client
    patch.undo()
    db.close()


def wait_for(
    client: TestClient,
    run_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    timeout: float = 120.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        latest = response.json()["run"]
        if predicate(latest):
            return latest
        time.sleep(0.2)
    raise AssertionError(f"等待超时，最后状态：{latest}")


def start_run(client: TestClient, **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "model_id": "mlp-mnist",
        "dataset_id": "mnist",
        "seed": 42,
        "hyperparams": dict(SMALL_HYPER),
    }
    payload.update(overrides)
    # 上一 run 收尾（reader 线程清理活动槽位）有百毫秒级延迟，这里重试几次
    deadline = time.monotonic() + 5.0
    while True:
        response = client.post("/api/runs", json=payload)
        if response.status_code == 201:
            return response.json()["run"]["id"]
        if response.status_code == 409 and time.monotonic() < deadline:
            time.sleep(0.2)
            continue
        raise AssertionError(response.text)


def stop_run(client: TestClient, run_id: str) -> None:
    response = client.post(f"/api/runs/{run_id}/control", json={"action": "stop"})
    if response.status_code == 409:
        return
    assert response.status_code == 200, response.text
    wait_for(client, run_id, lambda run: run["status"] in ("stopped", "finished", "failed"))


# ------------------------------------------------------------------ 基础查询


def test_dataset_list_reports_cache(client: TestClient) -> None:
    datasets = client.get("/api/datasets").json()["datasets"]
    mnist = next(item for item in datasets if item["id"] == "mnist")
    assert mnist["cached"] is True
    assert mnist["size_bytes"] > 0
    assert mnist["manual_dir"].endswith("manual")
    assert len(mnist["files"]) == 4
    assert all(item["md5_ok"] for item in mnist["files"])


def test_run_requires_model(client: TestClient) -> None:
    response = client.post("/api/runs", json={"dataset_id": "mnist"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "errors.run.needModel"


def test_run_unknown_dataset(client: TestClient, tmp_path) -> None:
    response = client.post("/api/runs", json={"model_id": "mlp-mnist", "dataset_id": "cifar10"})
    assert response.status_code == 404
    assert response.json()["error"]["message_key"] == "errors.dataset.notFound"


def test_run_invalid_hyperparams(client: TestClient) -> None:
    response = client.post(
        "/api/runs",
        json={
            "model_id": "mlp-mnist",
            "dataset_id": "mnist",
            "hyperparams": {"epochs": 999},
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["args"] == {"param": "epochs"}


# ------------------------------------------------------------------ 生命周期


def test_run_lifecycle(client: TestClient) -> None:
    run_id = start_run(client)
    detail = wait_for(client, run_id, lambda run: run["step"] >= 10 and run["status"] == "running")
    assert detail["device"] == "cpu"
    assert detail["planned_steps"] > 0
    assert detail["elapsed_s"] is not None

    # 并发约束：同时只允许一个活动 run
    busy = client.post(
        "/api/runs",
        json={"model_id": "mlp-mnist", "dataset_id": "mnist", "hyperparams": dict(SMALL_HYPER)},
    )
    assert busy.status_code == 409
    assert busy.json()["error"]["code"] == "errors.run.busy"
    assert busy.json()["error"]["args"]["run_id"] == run_id

    # 暂停：状态变为 paused，且 step 不再前进
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "pause"}).status_code == 200
    paused = wait_for(client, run_id, lambda run: run["status"] == "paused")
    frozen_step = paused["step"]
    time.sleep(1.2)
    again = client.get(f"/api/runs/{run_id}").json()["run"]
    assert again["status"] == "paused"
    assert again["step"] == frozen_step, "暂停后 step 不应继续增长"

    # 恢复：从暂停点继续，不丢 step
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "resume"}).status_code == 200
    resumed = wait_for(client, run_id, lambda run: run["status"] == "running" and run["step"] > frozen_step)
    assert resumed["step"] > frozen_step

    # 训练中调 lr：下一 step 生效并记入指标
    assert (
        client.post(f"/api/runs/{run_id}/control", json={"action": "set_lr", "value": 0.0001}).status_code
        == 200
    )
    lr_series: list[float] = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        payload = client.get(f"/api/runs/{run_id}/metrics", params={"names": "lr"}).json()
        lr_series = payload["series"].get("lr", {}).get("values", [])
        if any(abs(value - 0.0001) < 1e-12 for value in lr_series):
            break
        time.sleep(0.3)
    assert any(abs(value - 0.0001) < 1e-12 for value in lr_series), lr_series[:5]
    assert lr_series[0] == pytest.approx(0.001, rel=1e-6), "起始 lr 应记录为 0.001"

    # 非法控制值
    bad = client.post(f"/api/runs/{run_id}/control", json={"action": "set_lr", "value": -1})
    assert bad.status_code == 422
    assert bad.json()["error"]["args"] == {"param": "lr"}
    conflict = client.post(f"/api/runs/{run_id}/control", json={"action": "pause", "value": None})
    assert conflict.status_code == 200  # 仍在 running，pause 合法

    # 停止：3s 内子进程退出并落库为 stopped
    assert client.post(f"/api/runs/{run_id}/control", json={"action": "stop"}).status_code == 200
    final = wait_for(client, run_id, lambda run: run["status"] in ("stopped", "finished", "failed"))
    assert final["status"] == "stopped", final
    assert final["total_steps"] > frozen_step
    assert final["finished_at"]

    # 已结束的 run 再发命令 → 409 notActive
    inactive = client.post(f"/api/runs/{run_id}/control", json={"action": "pause"})
    assert inactive.status_code == 409
    assert inactive.json()["error"]["code"] == "errors.run.notActive"


def test_metrics_persisted_and_replayable(client: TestClient) -> None:
    runs = client.get("/api/runs", params={"limit": 5}).json()["runs"]
    assert runs, "应有历史 run"
    run = next(item for item in runs if item["total_steps"] and item["status"] == "stopped")
    response = client.get(
        f"/api/runs/{run['id']}/metrics", params={"names": "loss,acc,lr,val_loss,val_acc,grad_norm,throughput"}
    )
    assert response.status_code == 200
    payload = response.json()
    series = payload["series"]
    assert len(series["loss"]["steps"]) > 10
    assert series["loss"]["steps"] == sorted(series["loss"]["steps"])
    assert any(value is not None for value in series["acc"]["values"])
    assert series["val_loss"]["steps"], "epoch 末应有 val 指标"
    assert series["grad_norm"]["values"][0] > 0
    assert series["throughput"]["values"][0] > 0
    assert "vram_mb" not in series, "CPU 不应写显存指标"

    # 落库行数与降采样无关：DB 里是完整序列
    assert db.count_metrics(run["id"]) > 100
    detail = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert detail["best_metric"] is not None
    assert detail["hyperparams"]["epochs"] == SMALL_HYPER["epochs"]
    assert detail["graph"]["nodes"], "详情应带图 IR 供回放"

    # 降采样：max_points 生效
    small = client.get(f"/api/runs/{run['id']}/metrics", params={"names": "loss", "max_points": 8}).json()
    assert len(small["series"]["loss"]["steps"]) <= 8
    assert small["series"]["loss"]["total"] == len(series["loss"]["steps"])


def test_finished_run_and_delete(client: TestClient) -> None:
    run_id = start_run(
        client, hyperparams={**SMALL_HYPER, "epochs": 1, "train_size": 320, "val_size": 64}
    )
    finished = wait_for(client, run_id, lambda run: run["status"] in ("finished", "failed"))
    assert finished["status"] == "finished", finished
    assert finished["best_metric"] is not None
    assert files.checkpoint_path(run_id).is_file(), "最优权重应落盘"

    deleted = client.delete(f"/api/runs/{run_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert not files.run_dir(run_id).exists()


# ------------------------------------------------------------------ 单元层


def test_lttb_keeps_endpoints_and_order() -> None:
    steps = list(range(5000))
    values = [float(index) ** 0.5 for index in steps]
    out_steps, out_values = downsample_series(steps, values, 2000)
    assert len(out_steps) == 2000
    assert out_steps[0] == 0 and out_steps[-1] == 4999
    assert all(later > earlier for earlier, later in zip(out_steps, out_steps[1:]))
    assert len(out_values) == len(out_steps)
    assert downsample_series(steps, values, 5000)[0] == steps


def test_lttb_small_inputs() -> None:
    assert lttb([1, 2], [1.0, 2.0], 10) == ([1, 2], [1.0, 2.0])
    steps, values = lttb([1, 2, 3, 4], [4.0, 1.0, 3.0, 2.0], 3)
    assert steps[0] == 1 and steps[-1] == 4


def test_orphan_runs_marked_interrupted(client: TestClient) -> None:
    run_id = db.new_run_id()
    files.ensure_run_dir(run_id)
    db.insert_run(
        {
            "id": run_id,
            "name": "孤儿 run",
            "kind": "dl",
            "status": "running",
            "created_at": db.now_iso(),
        }
    )
    orphans = db.mark_orphans_interrupted()
    assert run_id in orphans
    assert db.get_run(run_id)["status"] == "interrupted"
    db.delete_run(run_id)


# ------------------------------------------------------------------ M4 历史与清理


def test_storage_and_clear_history(client: TestClient) -> None:
    # 自给自足造两条：一条已结束（待清理）、一条活动（清理时应被跳过）
    done_id = start_run(
        client, hyperparams={**SMALL_HYPER, "epochs": 1, "train_size": 320, "val_size": 64}
    )
    finished = wait_for(client, done_id, lambda run: run["status"] in ("finished", "failed"))
    assert finished["status"] == "finished", finished
    active_id = start_run(client)
    wait_for(client, active_id, lambda run: run["step"] >= 1)

    stats = client.get("/api/runs/storage")
    assert stats.status_code == 200, stats.text
    storage = stats.json()
    listed = [item["id"] for item in client.get("/api/runs", params={"limit": 200}).json()["runs"]]
    assert storage["run_count"] == len(listed) >= 2
    assert set(storage["by_run"]) == set(listed)
    assert storage["total_bytes"] == sum(storage["by_run"].values()) > 0
    assert storage["runs_dir"].endswith("runs")

    others = [run_id for run_id in listed if run_id != active_id]
    cleared = client.delete("/api/runs")
    assert cleared.status_code == 200, cleared.text
    payload = cleared.json()
    assert payload["deleted"] == len(others)
    assert payload["kept_active"] == active_id

    remaining = client.get("/api/runs", params={"limit": 200}).json()
    assert [item["id"] for item in remaining["runs"]] == [active_id]
    assert all(not files.run_dir(run_id).exists() for run_id in others)
    assert db.get_run(others[0]) is None
    after = client.get("/api/runs/storage").json()
    assert after["run_count"] == 1
    assert set(after["by_run"]) == {active_id}

    # 活动 run 本身仍可控制
    assert client.post(f"/api/runs/{active_id}/control", json={"action": "pause"}).status_code == 200
    stop_run(client, active_id)
