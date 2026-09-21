"""M6 上传测试（契约见 docs/02 §7.5）：三种格式、清单持久化、训练接入与删除。"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import config, main
from app.datasets import loaders, points2d, registry, upload
from app.store import db


def _png(color: tuple[int, int, int], size: tuple[int, int] = (40, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_zip(class_count: int = 2, per_class: int = 9, root_stray: bool = True) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(class_count):
            for shot in range(per_class):
                color = (255 - index * 60, 40 + shot * 10, 200 - shot * 5)
                archive.writestr(f"class-{index}/img_{shot:03d}.png", _png(color))
        archive.writestr("notes.txt", "ignored")
        if root_stray:
            archive.writestr("root.png", _png((0, 0, 0)))
    return buffer.getvalue()


def make_csv(rows: int = 60, header: bool = True) -> bytes:
    rng = np.random.default_rng(3)
    lines = ["x,y,label"] if header else []
    for index in range(rows):
        x = rng.normal(-0.3 if index % 2 else 0.3, 0.2)
        y = rng.normal(0.0, 0.3)
        lines.append(f"{x:.4f},{y:.4f},{index % 2}")
    return "\n".join(lines).encode("utf-8")


def make_txt(repeat: int = 220) -> bytes:
    return ("AI-Visualize 字符级语料示例，用于上传测试。" * repeat).encode("utf-8")


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("upload-data")
    patch = pytest.MonkeyPatch()
    # 训练子进程是 spawn 出来的新解释器：数据目录必须同时经环境变量传递，否则它读默认 data/
    patch.setenv("AI_VISUALIZE_DATA_DIR", str(tmp))
    patch.setattr(config, "DATA_DIR", tmp)
    patch.setattr(config, "DATASETS_DIR", tmp / "datasets")
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


def start_run(client: TestClient, **payload: Any) -> str:
    deadline = time.monotonic() + 10.0
    while True:
        response = client.post("/api/runs", json=payload)
        if response.status_code == 201:
            return response.json()["run"]["id"]
        if response.status_code == 409 and time.monotonic() < deadline:
            time.sleep(0.2)
            continue
        raise AssertionError(response.text)


def post_file(client: TestClient, name: str, data: bytes) -> Any:
    return client.post(
        "/api/datasets/upload", files={"file": (name, data, "application/octet-stream")}
    )


def uploaded_id(client: TestClient, name: str, data: bytes) -> str:
    response = post_file(client, name, data)
    assert response.status_code == 200, response.text
    return response.json()["dataset"]["id"]


def dataset_payload(client: TestClient, dataset_id: str) -> dict[str, Any]:
    datasets = client.get("/api/datasets").json()["datasets"]
    return next(item for item in datasets if item["id"] == dataset_id)


# ---------------------------------------------------------------- 三种格式


def test_upload_zip_image_dir(client: TestClient) -> None:
    payload = make_zip(class_count=2, per_class=9)
    response = post_file(client, "My Cats.zip", payload)
    assert response.status_code == 200, response.text
    state = response.json()["dataset"]
    dataset_id = state["id"]
    assert dataset_id.startswith("up-my-cats-")
    assert state["loader"] == "image_dir"
    assert state["cached"] is True
    assert state["uploaded"] is True
    assert state["input_shape"] == [3, 64, 64]
    assert state["num_classes"] == 2
    assert "2 类" in state["note"]["zh"]

    # 落盘 = 上传时写定的 npz（md5 与清单一致），根下散图与 readme 被忽略
    spec = registry.get_spec(dataset_id)
    assert spec is not None and registry.is_cached(spec)
    raw = registry.raw_dir(spec) / "data.npz"
    entry = next(item for item in upload.read_manifest() if item["id"] == dataset_id)
    assert entry["file"]["md5"] == hashlib.md5(raw.read_bytes()).hexdigest()
    assert 0.0 < spec.mean < 1.0 and 0.0 < spec.std < 1.0

    # 每类 9 张 → 8 训练 / 1 验证（seed 42 切分在上传时固化）
    train_loader, val_loader, info = loaders.build_loaders(
        spec, batch_size=4, train_size=0, val_size=0
    )
    assert (info["train_samples"], info["val_samples"]) == (16, 2)
    batch_x, batch_y = next(iter(train_loader))
    assert tuple(batch_x.shape) == (4, 3, 64, 64) and batch_x.dtype.is_floating_point is False
    assert batch_y.dtype == torch.int64
    assert set(batch_y.tolist()) <= {0, 1}


def test_upload_csv_points(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "toy points.csv", make_csv(rows=60))
    state = dataset_payload(client, dataset_id)
    assert state["loader"] == "csv2d" and state["cached"] is True
    assert state["input_shape"] == [2] and state["num_classes"] == 2

    train = client.get(f"/api/datasets/{dataset_id}/points?split=train").json()
    val = client.get(f"/api/datasets/{dataset_id}/points?split=val").json()
    assert train["count"] == 48 and val["count"] == 12
    assert len(train["points"]) == 48 and len(train["labels"]) == 48
    again = client.get(f"/api/datasets/{dataset_id}/points?split=train").json()
    assert again["points"] == train["points"]  # 同一文件切分可复现

    spec = registry.get_spec(dataset_id)
    assert spec is not None
    x0, x1, y0, y1 = points2d.bounds(spec)
    assert x0 < x1 and y0 < y1

    # 内置合成集仍可画点；非点集数据集 404
    assert client.get("/api/datasets/moons/points").status_code == 200
    assert client.get("/api/datasets/mnist/points").status_code == 404


def test_upload_txt_text_char(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "my corpus.txt", make_txt())
    state = dataset_payload(client, dataset_id)
    assert state["loader"] == "text_char" and state["cached"] is True
    assert state["task"] == "text_lm" and state["vocab_size"] == state["num_classes"]
    assert state["input_shape"] == [32]

    spec = registry.get_spec(dataset_id)
    assert spec is not None and spec.corpus == "corpus.txt" and spec.vocab_cap == 1024
    _, _, info = loaders.build_loaders(spec, batch_size=8, train_size=0, val_size=0)
    assert info["seq_len"] == 32 and info["vocab_size"] > 1


# ---------------------------------------------------------------- 清单与重启


def test_manifest_reload_survives_restart(client: TestClient) -> None:
    before = [item for item in client.get("/api/datasets").json()["datasets"] if item["id"].startswith("up-")]
    assert len(before) >= 3
    ids = sorted(item["id"] for item in before)

    # 模拟重启：清空内存注册表后从 uploads.json 重新注入（main.lifespan 的同一条路径）
    for dataset_id in ids:
        registry.SPECS.pop(dataset_id)
    registry.UPLOADED.clear()
    upload.register_all()

    after = [item for item in client.get("/api/datasets").json()["datasets"] if item["id"].startswith("up-")]
    assert sorted(item["id"] for item in after) == ids
    assert all(item["cached"] for item in after)


def test_manifest_file_deleted_shows_not_cached(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "vanishing.csv", make_csv(rows=40))
    spec = registry.get_spec(dataset_id)
    assert spec is not None
    raw = registry.raw_dir(spec) / "points.csv"
    raw.unlink()
    state = dataset_payload(client, dataset_id)
    assert state["cached"] is False
    assert client.get(f"/api/datasets/{dataset_id}/points").status_code == 409


# ---------------------------------------------------------------- 错误路径


def test_upload_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 缺 file 字段
    response = client.post("/api/datasets/upload")
    assert response.status_code == 400
    assert response.json()["error"]["message_key"] == "errors.upload.noFile"

    # 扩展名不受支持
    response = post_file(client, "model.json", b"{}")
    assert response.status_code == 415
    error = response.json()["error"]
    assert error["message_key"] == "errors.upload.badType" and error["args"] == {"ext": ".json"}

    # 空文件
    response = post_file(client, "empty.txt", b"")
    assert response.status_code == 422

    # zip：不是 zip / 只有一个类别 / 某类不足 8 张
    assert post_file(client, "bad.zip", b"not a zip").status_code == 422
    assert post_file(client, "one-class.zip", make_zip(class_count=1)).status_code == 422
    thin = post_file(client, "thin.zip", make_zip(class_count=2, per_class=3))
    assert thin.status_code == 422
    assert "need >= 8" in thin.json()["error"]["detail"]

    # csv：标签越界 / 行数不足 / 列数不足 / 非 UTF-8
    bad_label = make_csv(rows=40).replace(b",1\n", b",2\n", 1)
    assert post_file(client, "label.csv", bad_label).status_code == 422
    assert post_file(client, "short.csv", make_csv(rows=6)).status_code == 422
    assert post_file(client, "cols.csv", b"x,y\n1,2\n").status_code == 422
    assert post_file(client, "binary.csv", b"\xff\xfe\x00\x01" * 64).status_code == 422

    # txt：字符数不足 / 非 UTF-8
    short = post_file(client, "short.txt", "只有一点点。".encode("utf-8"))
    assert short.status_code == 422
    assert "need >= 2000" in short.json()["error"]["detail"]
    assert post_file(client, "binary.txt", b"\xff\xfe\x00\x01" * 600).status_code == 422

    # 超出大小上限（把上限压到 0 触发，避免造大文件）
    monkeypatch.setattr(config, "UPLOAD_TXT_MAX_MB", 0)
    too_large = post_file(client, "big.txt", make_txt())
    assert too_large.status_code == 413
    error = too_large.json()["error"]
    assert error["message_key"] == "errors.upload.tooLarge"
    assert error["args"] == {"format": "txt", "limit_mb": 0}

    # 失败的请求不留残留：清单里没有新增条目、数据目录里没有多出来的 up-* 目录
    ids = [item["id"] for item in client.get("/api/datasets").json()["datasets"] if item["id"].startswith("up-")]
    directories = sorted(path.name for path in (config.DATASETS_DIR).glob("up-*"))
    assert sorted(ids) == directories


# ---------------------------------------------------------------- 删除


def test_delete_builtin_and_unknown(client: TestClient) -> None:
    builtin = client.delete("/api/datasets/moons")
    assert builtin.status_code == 403
    assert builtin.json()["error"]["message_key"] == "errors.dataset.builtin"

    unknown = client.delete("/api/datasets/up-missing-0000")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["message_key"] == "errors.dataset.notFound"


def test_delete_uploaded_removes_everything(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "disposable.csv", make_csv(rows=40))
    spec = registry.get_spec(dataset_id)
    assert spec is not None
    directory = config.DATASETS_DIR / dataset_id
    assert directory.is_dir()

    response = client.delete(f"/api/datasets/{dataset_id}")
    assert response.status_code == 200 and response.json() == {"deleted": dataset_id}
    assert registry.get_spec(dataset_id) is None
    assert not directory.exists()
    assert all(item["id"] != dataset_id for item in upload.read_manifest())
    assert client.get(f"/api/datasets/{dataset_id}/points").status_code == 404


# ---------------------------------------------------------------- 训练接入


def _cnn_graph() -> dict[str, Any]:
    return {
        "ir_version": 1,
        "id": "upload-cnn",
        "kind": "dl",
        "task": "image_classification",
        "name": {"zh": "上传集小 CNN", "en": "Tiny CNN on upload"},
        "nodes": [
            {"id": "input", "type": "Input", "params": {"shape": [3, 64, 64]}},
            {"id": "conv", "type": "Conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "act", "type": "Activation", "params": {"name": "relu"}},
            {"id": "pool", "type": "MaxPool2d", "params": {"kernel_size": 2}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 2}},
            {"id": "output", "type": "Output", "params": {"classes": 2}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "conv"},
            {"id": "e2", "source": "conv", "target": "act"},
            {"id": "e3", "source": "act", "target": "pool"},
            {"id": "e4", "source": "pool", "target": "flat"},
            {"id": "e5", "source": "flat", "target": "fc"},
            {"id": "e6", "source": "fc", "target": "output"},
        ],
    }


def test_upload_dataset_trains_dl(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "train me.zip", make_zip(class_count=3, per_class=8))
    run_id = start_run(
        client,
        graph=_cnn_graph(),
        dataset_id=dataset_id,
        seed=42,
        hyperparams={
            "optimizer": "adam",
            "lr": 0.01,
            "batch_size": 8,
            "epochs": 2,
            "loss": "cross_entropy",
            "grad_clip": 5.0,
            "train_size": 0,
            "val_size": 0,
        },
    )
    run = wait_for(client, run_id, lambda item: item["status"] == "finished")
    assert run["dataset_id"] == dataset_id
    metrics = client.get(f"/api/runs/{run_id}/metrics?names=val_acc").json()
    assert metrics["series"]["val_acc"], "上传集训练应写出 val_acc"


def test_upload_dataset_trains_ml(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "ml points.csv", make_csv(rows=80))
    spec = {
        "ir_version": 1,
        "id": "upload-logreg",
        "kind": "ml",
        "task": "binary_classification",
        "name": {"zh": "逻辑回归 · 上传点集", "en": "Logistic Regression · Upload"},
        "algo": "logistic_regression",
        "params": {"lr": 0.5, "epochs": 3, "batch_size": 16, "l2": 0.0, "render_delay_ms": 0},
        "probe_defaults": {"every_n_steps": 5},
    }
    run_id = start_run(client, graph=spec, dataset_id=dataset_id)
    run = wait_for(client, run_id, lambda item: item["status"] == "finished")
    assert run["kind"] == "ml" and run["best_metric"] is not None
    snapshots = client.get(f"/api/runs/{run_id}/snapshots").json()["snapshots"]
    assert snapshots and all(item["kind"] == "boundary" for item in snapshots)


def test_ml_rejects_image_dir_upload(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "not for ml.zip", make_zip(class_count=2, per_class=8))
    spec = {
        "ir_version": 1,
        "id": "upload-logreg-2",
        "kind": "ml",
        "task": "binary_classification",
        "name": {"zh": "逻辑回归", "en": "Logistic Regression"},
        "algo": "logistic_regression",
        "params": {"lr": 0.5, "epochs": 2, "batch_size": 16, "l2": 0.0, "render_delay_ms": 0},
        "probe_defaults": {"every_n_steps": 5},
    }
    response = client.post("/api/runs", json={"graph": spec, "dataset_id": dataset_id})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["message_key"] == "errors.dataset.kindMismatch"
    assert error["args"]["expected"] == "csv2d / synth2d"


def test_upload_id_never_collides_with_builtin(client: TestClient) -> None:
    dataset_id = uploaded_id(client, "moons.csv", make_csv(rows=40))
    assert dataset_id != "moons" and dataset_id.startswith("up-moons-")
    assert registry.get_spec("moons") is not None
    assert registry.get_spec("moons").loader == "synth2d"
