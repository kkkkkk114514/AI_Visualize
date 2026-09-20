"""M3 语料与数据集绑定测试：字符级语料分词、滑窗切分、分类头绑定规则。"""

from __future__ import annotations

import math

import pytest
import torch

from app.datasets import loaders, registry
from app.graph import shapes
from app.graph.ir import bind_dataset, parse_graph, validate_graph

TEXT_SPECS = {"alice": 76, "xiyouji": 1024}


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
