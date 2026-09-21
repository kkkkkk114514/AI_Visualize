"""M5-4 测试：manager 按 kind 分派与 `/api/algos`（契约见 docs/02 §13.2 / §13.4）。"""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.algos import schema as algo_schema, spec as algo_spec
from app.api.models import read_preset
from app.api.ws import hub
from app.datasets import registry
from app.metrics import METRIC_NAMES
from app.runners import base
from app.store import db

ML_PRESETS = (
    "ml-linear-blobs",
    "ml-logreg-moons",
    "ml-tree-spiral",
    "ml-forest-moons",
    "ml-svm-circles",
)

ML_SPEC: dict[str, Any] = {
    "ir_version": 1,
    "id": "inline-logreg",
    "kind": "ml",
    "task": "binary_classification",
    "name": {"zh": "逻辑回归 · 双月", "en": "Logistic Regression · Moons"},
    "algo": "logistic_regression",
    "params": {"lr": 0.5, "epochs": 4, "batch_size": 32, "l2": 0.0, "render_delay_ms": 0},
    "probe_defaults": {"every_n_steps": 5},
}

RL_SPEC: dict[str, Any] = {
    "ir_version": 1,
    "id": "inline-qlearning",
    "kind": "rl",
    "task": "control",
    "algo": "q_learning",
    "env": {
        "width": 10,
        "height": 6,
        "start": [0, 0],
        "goal": [9, 5],
        "obstacles": [[3, 1], [3, 2], [6, 3]],
        "rewards": [[7, 4, 1.0]],
    },
    "params": {
        "episodes": 8,
        "max_steps": 20,
        "alpha": 0.2,
        "gamma": 0.95,
        "epsilon_start": 0.5,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.98,
        "render_delay_ms": 0,
    },
    "probe_defaults": {"every_n_steps": 40},
}

RL_METRICS = ("reward", "q_delta", "episode_reward", "epsilon", "episode_steps", "success")


EVENTS: list[dict[str, Any]] = []


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("data")
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "DATA_DIR", tmp)
    patch.setattr(config, "RUNS_DIR", tmp / "runs")
    patch.setattr(config, "DB_PATH", tmp / "app.db")

    async def collect(payload: dict[str, Any]) -> None:
        EVENTS.append(payload)

    patch.setattr(hub, "broadcast", collect)  # 拦截 WS 广播，断言事件链路（无需真连 WS）
    with TestClient(main.app) as test_client:
        yield test_client
    patch.undo()
    EVENTS.clear()
    db.close()


def wait_for(
    client: TestClient,
    run_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    timeout: float = 90.0,
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


def wait_event(
    run_id: str, event_type: str, predicate: Callable[[dict[str, Any]], bool], timeout: float = 10.0
) -> dict[str, Any]:
    """从广播里找一条事件（ML / RL 单次 run 只活几百毫秒，live 字段只能靠事件断言）。"""
    deadline = time.monotonic() + timeout
    while True:
        for event in EVENTS:
            if event.get("type") == event_type and event.get("run_id") == run_id:
                if predicate(event):
                    return event
        if time.monotonic() >= deadline:
            raise AssertionError(f"未收到匹配事件：{event_type}")
        time.sleep(0.05)


def start_run(client: TestClient, **payload: Any) -> str:
    """提交 run；上一 run 的槽位释放在 reader 线程里有百毫秒级延迟，409 时重试。"""
    deadline = time.monotonic() + 10.0
    while True:
        response = client.post("/api/runs", json=payload)
        if response.status_code == 201:
            return response.json()["run"]["id"]
        if response.status_code == 409 and time.monotonic() < deadline:
            time.sleep(0.2)
            continue
        raise AssertionError(response.text)


# ------------------------------------------------------------------ schema 接口


def test_algos_catalog(client: TestClient) -> None:
    payload = client.get("/api/algos").json()
    algos = payload["algos"]
    assert set(algos) == {
        "linear_regression",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "svm",
        "q_learning",
    }
    kinds = {name: schema["kind"] for name, schema in algos.items()}
    assert kinds["q_learning"] == "rl"
    assert set(kinds.values()) == {"ml", "rl"}

    logreg = algos["logistic_regression"]
    by_name = {field["name"]: field for field in logreg["fields"]}
    assert by_name["lr"]["default"] == 0.5
    assert (by_name["lr"]["min"], by_name["lr"]["max"]) == (0.01, 1.0)
    assert by_name["lr"]["type"] == "float" and by_name["lr"]["step"] == 0.01
    assert all(field["label_key"].startswith("algo.") for field in logreg["fields"])

    kernel = next(field for field in algos["svm"]["fields"] if field["name"] == "kernel")
    assert kernel["choices"] == ["linear", "rbf"] and kernel["default"] == "rbf"
    q_fields = {field["name"]: field["default"] for field in algos["q_learning"]["fields"]}
    assert q_fields["episodes"] == 250 and q_fields["epsilon_start"] == 0.5


# ------------------------------------------------------------------ 两条端到端


def test_ml_run_dispatch_and_boundary(client: TestClient) -> None:
    run_id = start_run(client, graph=ML_SPEC, dataset_id="moons", seed=3)
    run = wait_for(client, run_id, lambda run: run["status"] == "finished")
    assert run["kind"] == "ml"
    assert run["device"] == "cpu"
    assert run["hyperparams"] == {}
    assert run["probes"][0]["node_id"] == "model"
    assert run["probes"][0]["kind"] == "boundary"
    assert run["probes"][0]["every_n_steps"] == 5
    assert run["best_metric"] is not None and 0.0 <= run["best_metric"] <= 1.0
    graph = run["graph"]
    assert graph["kind"] == "ml" and graph["algo"] == "logistic_regression"
    assert graph["dataset_id"] == "moons"
    assert graph["params"]["epochs"] == 4  # 缺省值已归一化落库

    # 事件链：ML 的初始 status 带 device=cpu 与 planned_steps 上界（epochs × steps_per_epoch）
    opening = wait_event(run_id, base.WIRE_STATUS, lambda event: event["status"] == "running")
    assert opening["device"] == "cpu"
    assert opening["planned_steps"] == 40  # 4 × ceil(300 / 32)
    closing = wait_event(run_id, base.WIRE_STATUS, lambda event: event["status"] == "finished")
    assert closing["step"] == run["total_steps"] == 40

    # 曲线：ML 指标名与 DL 同源，val_* 在 METRIC_NAMES 之内
    series = client.get(f"/api/runs/{run_id}/metrics?names=loss,val_acc").json()["series"]
    assert set(series) <= {"loss", "val_acc"}
    assert series["loss"]["total"] == 40
    assert series["val_acc"]["total"] == 4

    snapshots = client.get(f"/api/runs/{run_id}/snapshots?kind=boundary").json()["snapshots"]
    assert snapshots and all(item["node_id"] == "model" for item in snapshots)
    first = snapshots[0]
    assert first["shape"] == [96, 96]
    payload = client.get(f"/api/runs/{run_id}/snapshots/{first['id']}").json()
    assert payload["layout"] == "HW" and payload["dtype"] == "uint8"
    assert payload["meta"]["mode"] == "score"  # 逻辑回归走决策函数连续场
    assert len(payload["meta"]["x_range"]) == 2 and len(payload["meta"]["y_range"]) == 2
    assert payload["meta"]["algo"] == "logistic_regression"


def test_rl_run_dispatch_and_grid(client: TestClient) -> None:
    run_id = start_run(client, graph=RL_SPEC, dataset_id="gridworld", seed=5)
    run = wait_for(client, run_id, lambda run: run["status"] == "finished")
    assert run["kind"] == "rl"
    assert run["device"] == "cpu"
    assert run["total_steps"] == 8 * 20
    graph = run["graph"]
    assert graph["algo"] == "q_learning" and graph["dataset_id"] == "gridworld"
    assert graph["env"]["width"] == 10 and graph["env"]["height"] == 6
    assert graph["env"]["obstacles"] == [[3, 1], [3, 2], [6, 3]]

    opening = wait_event(run_id, base.WIRE_STATUS, lambda event: event["status"] == "running")
    assert opening["planned_steps"] == 8 * 20 and opening["device"] == "cpu"

    names = client.get(f"/api/runs/{run_id}/metrics").json()["series"].keys()
    assert names and set(names) <= set(METRIC_NAMES)
    assert set(names) <= set(RL_METRICS)

    snapshots = client.get(f"/api/runs/{run_id}/snapshots?kind=grid").json()["snapshots"]
    assert snapshots and snapshots[0]["node_id"] == "agent"
    payload = client.get(f"/api/runs/{run_id}/snapshots/{snapshots[0]['id']}").json()
    assert payload["shape"] == [6, 10] and payload["layout"] == "HW"
    assert len(payload["meta"]["policy"]) == 60
    assert "trajectory" in payload["meta"]


# ------------------------------------------------------------------ 校验与错误


def test_algo_spec_rejected(client: TestClient) -> None:
    unknown = dict(ML_SPEC, algo="knn")
    response = client.post("/api/runs", json={"graph": unknown, "dataset_id": "moons"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_algo" and error["message_key"] == "errors.algo.unknownAlgo"

    bad_param = dict(ML_SPEC, params={"epochs": 999})
    response = client.post("/api/runs", json={"graph": bad_param, "dataset_id": "moons"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_algo" and error["args"]["param"] == "epochs"

    bad_env = dict(RL_SPEC, env={**RL_SPEC["env"], "goal": [0, 0]})
    response = client.post("/api/runs", json={"graph": bad_env, "dataset_id": "gridworld"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_algo"


def test_algo_dataset_kind_mismatch(client: TestClient) -> None:
    response = client.post("/api/runs", json={"graph": ML_SPEC, "dataset_id": "gridworld"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["message_key"] == "errors.dataset.kindMismatch"
    assert error["args"] == {"id": "gridworld", "kind": "ml", "expected": "csv2d / synth2d"}

    response = client.post("/api/runs", json={"graph": RL_SPEC, "dataset_id": "moons"})
    assert response.json()["error"]["args"] == {"id": "moons", "kind": "rl", "expected": "gridworld"}

    # DL 不吃 gridworld（docs/02 §13.3）
    response = client.post(
        "/api/runs", json={"model_id": "mlp-mnist", "dataset_id": "gridworld"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["args"]["kind"] == "dl"


def test_algo_dataset_missing(client: TestClient) -> None:
    response = client.post("/api/runs", json={"graph": ML_SPEC})
    assert response.status_code == 404
    assert response.json()["error"]["message_key"] == "errors.dataset.notFound"
    response = client.post("/api/runs", json={"graph": ML_SPEC, "dataset_id": "mnist"})
    assert response.status_code == 422  # loader 不匹配先于缓存检查


def test_algo_probe_validation(client: TestClient) -> None:
    response = client.post(
        "/api/runs",
        json={
            "graph": ML_SPEC,
            "dataset_id": "moons",
            "probes": [{"node_id": "agent", "kind": "grid"}],
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_probe" and error["message_key"] == "errors.probe.badStream"
    assert error["args"] == {"expected": "model:boundary"}


def test_algo_run_appears_in_history(client: TestClient) -> None:
    runs = client.get("/api/runs?limit=50").json()["runs"]
    kinds = {run["kind"] for run in runs}
    assert {"ml", "rl"} <= kinds
    for run in runs:
        if run["kind"] in ("ml", "rl"):
            assert run["total_steps"] > 0 and run["status"] == "finished"


# ------------------------------------------------------------------ M5 预置


def test_algo_presets_validate() -> None:
    loaders = {"ml": "synth2d", "rl": "gridworld"}
    for model_id in ML_PRESETS + ("rl-gridworld",):
        model = algo_spec.parse_spec(read_preset(model_id))
        assert model.id == model_id
        assert model.name.get("zh") and model.desc.get("en")
        dataset = registry.get_spec(model.dataset_id or "")
        assert dataset is not None and dataset.loader == loaders[model.kind]

        schema = algo_schema.get(model.algo)
        assert schema is not None and schema.kind == model.kind
        assert set(model.params) == {field.name for field in schema.fields}

        probes = algo_spec.resolve_probes(None, model)
        assert probes[0]["every_n_steps"] == model.probe_every_n
        assert probes[0]["node_id"] == model.node_id and probes[0]["kind"] == model.probe_kind

    grid = algo_spec.parse_spec(read_preset("rl-gridworld"))
    assert grid.env is not None and (grid.env.width, grid.env.height) == (10, 6)
    assert grid.env.start == (0, 0) and grid.env.goal == (9, 5)
    assert grid.params["epsilon_start"] == 0.5 and grid.params["epsilon_decay"] == 0.98


def test_ml_preset_runs_via_model_id(client: TestClient) -> None:
    # 不给 dataset_id：应回落到 spec 里的 `blobs`（契约见 docs/02 §7.2）
    run_id = start_run(client, model_id="ml-linear-blobs", seed=2)
    run = wait_for(client, run_id, lambda item: item["status"] == "finished", timeout=120)
    assert run["kind"] == "ml" and run["model_id"] == "ml-linear-blobs"
    assert run["dataset_id"] == "blobs"
    assert run["graph"]["algo"] == "linear_regression"
    assert run["graph"]["dataset_id"] == "blobs"
    assert "nodes" not in run["graph"] and "env" not in run["graph"]  # ML spec 既无图也无环境

    snapshots = client.get(f"/api/runs/{run_id}/snapshots?kind=boundary").json()["snapshots"]
    assert len(snapshots) >= 5  # every_n_steps = 10 ⇒ 40 步的每一步都该有快照
    assert all(item["node_id"] == "model" for item in snapshots)
