"""M3 集成测试：探针 run 端到端（真实子进程训练 → 快照落盘/落库 → ETag 拉取）。"""

from __future__ import annotations

import base64
import sqlite3
import json
import time
from typing import Any, Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.datasets import registry
from app.graph import shapes
from app.graph.ir import bind_dataset, load_preset, validate_graph
from app.runners import base
from app.runners import manager as manager_module
from app.store import db, files

# 字符级小 GPT 的最小可跑配置：20 step（2 epoch × 10），CPU 上约 2~3s
HYPER: dict[str, Any] = {
    "optimizer": "adamw",
    "lr": 0.003,
    "batch_size": 16,
    "epochs": 2,
    "loss": "cross_entropy",
    "grad_clip": 1.0,
    "train_size": 160,
    "val_size": 32,
}

PROBES: list[dict[str, Any]] = [
    {"node_id": "attn1", "kind": "attention", "every_n_steps": 1},
    {"node_id": "fc1", "kind": "histogram", "every_n_steps": 1},
]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    if not registry.is_cached(registry.ALICE):
        pytest.skip("alice 字符级语料未就位，跳过探针集成测试")
    tmp = tmp_path_factory.mktemp("probe-data")
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


def start_probe_run(client: TestClient, probes: Any = PROBES) -> dict[str, Any]:
    payload = {
        "model_id": "gpt-char",
        "dataset_id": "alice",
        "seed": 7,
        "hyperparams": dict(HYPER),
        "probes": probes,
    }
    deadline = time.monotonic() + 5.0
    while True:
        response = client.post("/api/runs", json=payload)
        if response.status_code == 201:
            return response.json()["run"]
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


def dequantize(payload: dict[str, Any]) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(payload["data_b64"]), dtype=np.uint8)
    quant = raw.reshape(payload["shape"]).astype(np.float32)
    return payload["min"] + quant / 255.0 * (payload["max"] - payload["min"])


def test_m3_preset_binds_to_char_corpus() -> None:
    """预置 gpt-char：绑定语料后词表对齐（词表 76），探针默认指向两层注意力。"""
    graph = load_preset("gpt-char")
    assert graph is not None
    assert graph.task == "text_lm" and graph.probe_defaults == ["attn1", "attn2"]
    issues, params = validate_graph(graph)
    assert not [issue for issue in issues if issue.severity == "error"]
    bind_dataset(graph, params, registry.meta(registry.ALICE))
    assert params["emb"]["num_embeddings"] == 76
    assert params["head"]["out_features"] == 76
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]
    assert result["nodes"]["attn1"]["out_shape"] == [2, 32, 64]
    assert 100_000 < result["total_params"] < 130_000


def test_probe_run_persists_snapshots_and_serves_etag(client: TestClient) -> None:
    started = start_probe_run(client)
    run_id = started["id"]
    assert [item["kind"] for item in started["probes"]] == ["attention", "histogram"]

    def enough(run: dict[str, Any]) -> bool:
        snapshots = client.get(f"/api/runs/{run_id}/snapshots").json()["snapshots"]
        kinds = {item["kind"] for item in snapshots}
        return run["status"] == "running" and kinds == {"attention", "histogram"} and len(snapshots) >= 4

    wait_for(client, run_id, enough)
    snapshots = client.get(f"/api/runs/{run_id}/snapshots").json()["snapshots"]
    attention = next(item for item in snapshots if item["kind"] == "attention")
    histogram = next(item for item in snapshots if item["kind"] == "histogram")
    assert attention["shape"] == [4, 32, 32], attention
    assert histogram["shape"] == [32], histogram
    assert attention["node_id"] == "attn1" and histogram["node_id"] == "fc1"
    assert attention["step"] >= 1 and attention["run_id"] == run_id

    # 按流过滤（前端探针面板的取数路径）
    only_attn = client.get(
        f"/api/runs/{run_id}/snapshots", params={"node_id": "attn1", "kind": "attention"}
    ).json()["snapshots"]
    assert only_attn and all(item["kind"] == "attention" for item in only_attn)

    # payload：真实值域 + 因果掩码上三角为 0（docs/02 §6.2）
    url = f"/api/runs/{run_id}/snapshots/{attention['id']}"
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.headers["etag"] == f'"{attention["id"]}"'
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    record = response.json()
    assert record["step"] == attention["step"] and record["dtype"] == "uint8"
    assert record["meta"] == {"heads": 4, "tokens": 32, "causal": True}
    assert record["created_at"]
    matrices = dequantize(record)
    upper = np.triu(np.ones((32, 32), dtype=bool), k=1)
    assert np.all(matrices[:, upper] == 0.0), "因果掩码上三角必须为 0"

    # 内容不可变：命中 If-None-Match 返回 304
    cached = client.get(url, headers={"If-None-Match": f'"{attention["id"]}"'})
    assert cached.status_code == 304
    assert cached.content == b""
    assert cached.headers["etag"] == f'"{attention["id"]}"'

    # 落盘 + DB 索引
    path = files.snapshot_path(run_id, attention["id"])
    assert path.is_file(), path
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == record, "接口返回的就是落盘内容（同一份 JSON）"
    row = db.get_snapshot(attention["id"])
    assert row is not None and row["run_id"] == run_id
    assert row["file_path"] == str(path) and row["kind"] == "attention"
    assert db.list_snapshots(run_id, kind="histogram"), "历史 run 的探针流可回放"

    stop_run(client, run_id)
    stopped = client.get(f"/api/runs/{run_id}").json()["run"]
    assert stopped["status"] in ("stopped", "finished")
    # run 结束后快照仍可按 id 拉取（回放路径）
    assert client.get(url).status_code == 200
    assert client.get(f"/api/runs/{run_id}/snapshots").json()["snapshots"]


def test_probe_run_rejects_invalid_config(client: TestClient) -> None:
    before = db.count_runs()
    for probes, key in (
        ([{"node_id": "nope"}], "errors.probe.unknownNode"),
        ([{"node_id": "attn1", "kind": "rainbow"}], "errors.probe.badKind"),
        ([{"node_id": "attn1", "every_n_steps": 0}], "errors.probe.badValue"),
        ("attn1", "errors.probe.badConfig"),
    ):
        response = client.post(
            "/api/runs",
            json={
                "model_id": "gpt-char",
                "dataset_id": "alice",
                "hyperparams": dict(HYPER),
                "probes": probes,
            },
        )
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_probe", error
        assert error["message_key"] == key, error
    assert db.count_runs() == before, "探针非法不应留下半成品 run"


def test_probe_defaults_come_from_graph(client: TestClient) -> None:
    """不传 probes 时用图的 probe_defaults（attn1/attn2 → attention）。"""
    started = start_probe_run(client, probes=None)
    assert [(item["node_id"], item["kind"]) for item in started["probes"]] == [
        ("attn1", "attention"),
        ("attn2", "attention"),
    ]
    assert started["probes"][0]["every_n_steps"] == config.PROBE_DEFAULT_EVERY_N
    stop_run(client, started["id"])


def test_runs_schema_migration_adds_probes_json(tmp_path: Any) -> None:
    """老库（M2 建的 runs 表无 probes_json）启动时自动补列（docs/02 §7.2）。"""
    old_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(old_path))
    conn.executescript(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, "
        "model_id TEXT, graph_json TEXT, dataset_id TEXT, hyperparams_json TEXT, "
        "status TEXT NOT NULL, device TEXT, seed INTEGER, created_at TEXT, started_at TEXT, "
        "finished_at TEXT, error TEXT, best_metric REAL, total_steps INTEGER);"
    )
    conn.execute("INSERT INTO runs (id, name, kind, status) VALUES ('r_old', 'old', 'dl', 'finished')")
    conn.commit()
    conn.close()

    original = config.DB_PATH
    db.close()
    config.DB_PATH = old_path
    try:
        db.init_db()
        assert db.get_run("r_old")["probes"] == [], "老 run 的 probes 回退为空数组"
        columns = {row["name"] for row in db.connect().execute("PRAGMA table_info(runs)")}
        assert "probes_json" in columns
    finally:
        db.close()
        config.DB_PATH = original


class _FakeProcess:
    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        return None


def test_probe_event_inserts_row_and_broadcasts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """主进程收到子进程的 probe 事件：落 snapshots 表 + 只广播元数据（线上名 probe.snapshot）。"""
    manager = manager_module.manager
    run_id = db.new_run_id()
    snapshot_id = db.new_snapshot_id()
    files.ensure_run_dir(run_id)
    broadcast: list[dict[str, Any]] = []
    monkeypatch.setattr(manager, "_broadcast", broadcast.append)
    run = manager_module.RunProcess(run_id, _FakeProcess(), None, None)

    manager._on_event(
        run,
        {
            "type": base.EVENT_PROBE,
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "step": 12,
            "epoch": 2,
            "node_id": "attn2",
            "kind": "attention",
            "shape": [4, 32, 32],
            "min": 0.0,
            "max": 1.0,
            "file_path": str(files.snapshot_path(run_id, snapshot_id)),
        },
    )

    row = db.get_snapshot(snapshot_id)
    assert row is not None
    assert (row["run_id"], row["step"], row["epoch"], row["node_id"], row["kind"]) == (
        run_id,
        12,
        2,
        "attn2",
        "attention",
    )
    assert row["shape"] == [4, 32, 32]
    assert broadcast == [
        {
            "type": "probe.snapshot",
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "step": 12,
            "epoch": 2,
            "node_id": "attn2",
            "kind": "attention",
            "shape": [4, 32, 32],
        }
    ], broadcast
