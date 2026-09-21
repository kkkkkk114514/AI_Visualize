"""M3 语料与数据集绑定测试：字符级语料分词、滑窗切分、分类头绑定规则。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from app.datasets import loaders, registry, synth2d
from app.graph import shapes
from app.graph.ir import bind_dataset, parse_graph, validate_graph

TEXT_SPECS = {"alice": 76, "xiyouji": 1024}
SYNTH_IDS = ("moons", "circles", "blobs", "spiral")


def _tiny_lm_graph() -> dict:
    return {
        "id": "tiny-lm",
        "kind": "dl",
        "task": "text_lm",
        "nodes": [
            {"id": "input", "type": "Input", "params": {"shape": [32]}},
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 8, "embedding_dim": 32}},
            {"id": "pos", "type": "PositionalEncoding", "params": {"max_len": 64, "mode": "learned"}},
            {"id": "attn", "type": "MultiHeadAttention", "params": {"num_heads": 4, "causal": True}},
            {"id": "add", "type": "ResidualAdd", "params": {}},
            {"id": "norm", "type": "LayerNorm", "params": {"normalized_shape": 0}},
            {"id": "hidden", "type": "Linear", "params": {"out_features": 48}},
            {"id": "gelu", "type": "Activation", "params": {"name": "gelu"}},
            {"id": "head", "type": "Linear", "params": {"out_features": 5}},
            {"id": "output", "type": "Output", "params": {"classes": 5}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "emb"},
            {"id": "e2", "source": "emb", "target": "pos"},
            {"id": "e3", "source": "pos", "target": "attn"},
            {"id": "e4", "source": "pos", "target": "add", "target_port": "in1"},
            {"id": "e5", "source": "attn", "target": "add", "target_port": "in2"},
            {"id": "e6", "source": "add", "target": "norm"},
            {"id": "e7", "source": "norm", "target": "hidden"},
            {"id": "e8", "source": "hidden", "target": "gelu"},
            {"id": "e9", "source": "gelu", "target": "head"},
            {"id": "e10", "source": "head", "target": "output"},
        ],
    }


# ------------------------------------------------------------------ 语料与词表


def test_text_corpora_shipped_and_cached():
    for dataset_id, vocab in TEXT_SPECS.items():
        spec = registry.get_spec(dataset_id)
        assert spec is not None
        assert spec.loader == "text_char"
        assert spec.task == "text_lm"
        state = registry.cache_state(spec)
        assert state["cached"] is True
        assert state["vocab_size"] == vocab
        assert state["num_classes"] == vocab
        assert state["files"][0]["source"] == "corpora"
        assert state["size_bytes"] > 0
        assert registry.missing_files(spec) == []
        meta = registry.meta(spec)
        assert meta["input_shape"] == [32]
        assert meta["num_classes"] == vocab
        assert meta["vocab_size"] == vocab


def test_text_windows_are_next_token():
    spec = registry.get_spec("alice")
    assert spec is not None
    train_loader, val_loader, info = loaders.build_loaders(
        spec, batch_size=32, train_size=20000, val_size=5000, seed=42
    )
    x, y = next(iter(train_loader))
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert x.shape == (32, 32) and y.shape == (32, 32)
    assert int(x.max()) < info["vocab_size"]
    assert torch.equal(y[:, :-1], x[:, 1:]), "同一窗口内 y 应为 x 右移一位"
    assert info["seq_len"] == 32
    assert info["steps_per_epoch"] == math.ceil(info["train_samples"] / 32)
    assert 0.05 < info["val_samples"] / info["train_samples"] < 0.15

    limited, _, limited_info = loaders.build_loaders(
        spec, batch_size=16, train_size=320, val_size=64, seed=42
    )
    assert limited_info["train_samples"] == 320
    assert limited_info["val_samples"] == 64
    assert next(iter(limited))[0].shape == (16, 32)


def test_text_normalize_keeps_token_ids():
    meta = registry.meta(registry.ALICE)
    batch = torch.zeros(4, 32, dtype=torch.int64)
    out = loaders.normalize(batch, meta)
    assert out.dtype == torch.int64


# ------------------------------------------------------------------ 数据集绑定


def test_head_linear_binds_vocab_size():
    graph = parse_graph(_tiny_lm_graph())
    result = shapes.analyze(graph, dataset_id="xiyouji")
    assert result["ok"], result["errors"]
    assert result["nodes"]["input"]["out_shape"] == [2, 32]
    assert result["nodes"]["head"]["out_shape"] == [2, 32, 1024]
    assert result["nodes"]["hidden"]["out_shape"] == [2, 32, 48]
    assert result["nodes"]["output"]["out_shape"] == [2, 32, 1024]

    issues, params_by_node = validate_graph(parse_graph(_tiny_lm_graph()))
    assert not [i for i in issues if i.severity == "error"]
    bind_dataset(parse_graph(_tiny_lm_graph()), params_by_node, registry.meta(registry.XIYOUJI))
    assert params_by_node["head"]["out_features"] == 1024
    assert params_by_node["hidden"]["out_features"] == 48
    assert params_by_node["emb"]["num_embeddings"] == 1024


def test_regression_head_is_not_rebound():
    graph = parse_graph(
        {
            "id": "reg",
            "kind": "dl",
            "task": "regression",
            "nodes": [
                {"id": "input", "type": "Input", "params": {"shape": [1, 28, 28]}},
                {"id": "flat", "type": "Flatten", "params": {}},
                {"id": "head", "type": "Linear", "params": {"out_features": 1}},
                {"id": "output", "type": "Output", "params": {"classes": 10, "out_dim": 1}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "flat"},
                {"id": "e2", "source": "flat", "target": "head"},
                {"id": "e3", "source": "head", "target": "output"},
            ],
        }
    )
    result = shapes.analyze(graph, dataset_id="mnist")
    assert result["ok"], result["errors"]
    assert result["nodes"]["head"]["out_shape"] == [2, 1], "回归头不应被类别数覆盖"
    assert result["nodes"]["output"]["out_shape"] == [2, 1]


def test_text_graph_without_matching_dataset_keeps_declared_params():
    graph = parse_graph(_tiny_lm_graph())
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]
    assert result["nodes"]["head"]["out_shape"] == [2, 32, 5]
    assert result["nodes"]["emb"]["out_shape"] == [2, 32, 32]


@pytest.mark.parametrize("dataset_id", ["alice", "xiyouji"])
def test_text_dry_run_is_fast(dataset_id: str):
    graph = parse_graph(_tiny_lm_graph())
    shapes.clear_cache()
    result = shapes.analyze(graph, dataset_id=dataset_id)
    assert result["ok"], result["errors"]
    assert result["elapsed_ms"] < 1500, result["elapsed_ms"]


# ------------------------------------------------------------------ 二维合成数据集（M5）


@pytest.mark.parametrize("dataset_id", SYNTH_IDS)
def test_synth_points_are_deterministic_and_balanced(dataset_id: str):
    spec = registry.get_spec(dataset_id)
    assert spec is not None
    assert spec.loader == "synth2d"
    assert spec.files == ()
    assert registry.is_cached(spec) is True
    state = registry.cache_state(spec)
    assert state["cached"] is True
    assert state["files"] == []
    assert state["input_shape"] == [2]
    assert state["num_classes"] == 2
    assert registry.meta(spec)["loader"] == "synth2d"

    train_x, train_y, val_x, val_y = synth2d.split_arrays(dataset_id)
    assert train_x.shape == (300, 2) and val_x.shape == (200, 2)
    assert train_y.shape == (300,) and val_y.shape == (200,)
    assert train_x.dtype.name == "float32" and train_y.dtype.name == "int64"
    assert list(np.bincount(train_y)) == [150, 150]
    assert list(np.bincount(val_y)) == [100, 100]
    assert float(np.abs(train_x).max()) <= 1.0 and float(np.abs(val_x).max()) <= 1.0

    again_x, again_y = synth2d.make_points(dataset_id, "train")
    assert np.array_equal(again_x, train_x) and np.array_equal(again_y, train_y)


@pytest.mark.parametrize("dataset_id", SYNTH_IDS)
def test_synth_points_are_learnable(dataset_id: str):
    """数据要「有看头」：1-NN 明显好于随机猜，说明不是纯噪声。"""
    train_x, train_y = synth2d.make_points(dataset_id, "train")
    val_x, val_y = synth2d.make_points(dataset_id, "val")
    distance = ((val_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=-1)
    nearest = train_y[distance.argmin(axis=1)]
    assert float((nearest == val_y).mean()) >= 0.85


def test_synth_bounds_cover_points():
    low_x, high_x, low_y, high_y = synth2d.bounds("spiral")
    points, _ = synth2d.data("spiral")
    assert low_x < points[:, 0].min() and high_x > points[:, 0].max()
    assert low_y < points[:, 1].min() and high_y > points[:, 1].max()


def test_synth_loaders_feed_dl():
    spec = registry.get_spec("moons")
    train_loader, val_loader, info = loaders.build_loaders(spec, batch_size=32, seed=42)
    assert info["train_samples"] == 300 and info["val_samples"] == 200
    assert info["steps_per_epoch"] == math.ceil(300 / 32)
    x, y = next(iter(train_loader))
    assert x.shape == (32, 2) and x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert next(iter(val_loader))[0].shape[1] == 2
    normalized = loaders.normalize(x, registry.meta(spec))
    assert normalized.dtype == torch.float32
    assert float(normalized.abs().max()) > 0.5, "合成集不该被 /255 归一化"


def test_gridworld_spec_has_no_files():
    spec = registry.get_spec("gridworld")
    assert spec is not None
    assert spec.loader == "gridworld"
    assert spec.task == "control"
    assert spec.files == ()
    assert registry.is_cached(spec) is True
    assert registry.cache_state(spec)["cached"] is True


@pytest.fixture()
def api_client():
    """不走 lifespan 的客户端：只读端点，不碰真实 data/ 目录与 DB。"""
    from fastapi.testclient import TestClient

    from app import main

    return TestClient(main.app)


def test_points_endpoint(api_client):
    response = api_client.get("/api/datasets/moons/points")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["split"] == "train"
    assert payload["count"] == 300
    assert len(payload["points"]) == 300 and len(payload["labels"]) == 300
    assert set(payload["labels"]) == {0, 1}
    assert all(round(coord, 4) == coord for point in payload["points"] for coord in point)

    val = api_client.get("/api/datasets/spiral/points?split=val").json()
    assert val["count"] == 200 and val["split"] == "val"

    assert api_client.get("/api/datasets/mnist/points").status_code == 404
    assert api_client.get("/api/datasets/nope/points").status_code == 404
    bad = api_client.get("/api/datasets/moons/points?split=test")
    assert bad.status_code == 422
    assert bad.json()["error"]["message_key"] == "errors.dataset.badSplit"


def test_datasets_list_exposes_loader(api_client):
    payload = api_client.get("/api/datasets").json()["datasets"]
    loaders_by_id = {item["id"]: item["loader"] for item in payload}
    for dataset_id in (*SYNTH_IDS, "gridworld", "alice", "xiyouji", "mnist"):
        assert dataset_id in loaders_by_id, dataset_id
    assert loaders_by_id["moons"] == "synth2d"
    assert loaders_by_id["gridworld"] == "gridworld"
