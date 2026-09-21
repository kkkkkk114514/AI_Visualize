"""M1 验收测试：dry-run 形状/参数量与真实 PyTorch 一致，错误定位到节点。"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app.graph import shapes
from app.graph.ir import load_preset, parse_graph, validate_graph
from app.probes import hooks as probe_hooks

PRESET_DIR = None


def node_shapes(result: dict) -> dict[str, list[int]]:
    return {node_id: info.get("out_shape") for node_id, info in result["nodes"].items()}


# ---------------------------------------------------------------- 预置模型对照


def test_mlp_preset_matches_reference():
    graph = load_preset("mlp-mnist")
    assert graph is not None
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]

    reference = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    with torch.no_grad():
        expected = []
        h = torch.zeros(2, 1, 28, 28)
        for layer in reference:
            h = layer(h)
            expected.append(list(h.shape))

    shapes_by_node = node_shapes(result)
    assert shapes_by_node["input"] == [2, 1, 28, 28]
    assert shapes_by_node["flat"] == expected[0]
    assert shapes_by_node["fc1"] == expected[1]
    assert shapes_by_node["relu1"] == expected[2]
    assert shapes_by_node["fc2"] == expected[3]
    assert shapes_by_node["relu2"] == expected[4]
    assert shapes_by_node["head"] == expected[5]
    assert shapes_by_node["output"] == [2, 10]

    assert result["total_params"] == sum(p.numel() for p in reference.parameters())
    assert result["total_params"] == 109_386
    assert result["nodes"]["fc1"]["params"] == 100_480


def test_cnn_preset_matches_reference():
    graph = load_preset("cnn-mnist")
    assert graph is not None
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]

    reference = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(3136, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    with torch.no_grad():
        expected = []
        h = torch.zeros(2, 1, 28, 28)
        for layer in reference:
            h = layer(h)
            expected.append(list(h.shape))

    shapes_by_node = node_shapes(result)
    assert shapes_by_node["conv1"] == expected[0] == [2, 16, 28, 28]
    assert shapes_by_node["relu1"] == expected[1]
    assert shapes_by_node["pool1"] == expected[2] == [2, 16, 14, 14]
    assert shapes_by_node["flat"] == expected[3] == [2, 3136]
    assert shapes_by_node["fc1"] == expected[4]
    assert shapes_by_node["relu2"] == expected[5]
    assert shapes_by_node["head"] == expected[6] == [2, 10]

    assert result["total_params"] == sum(p.numel() for p in reference.parameters())
    assert result["total_params"] == 402_986


def test_preset_dry_run_is_fast():
    graph = load_preset("cnn-mnist")
    assert graph is not None
    shapes.clear_cache()
    result = shapes.analyze(graph)
    assert result["elapsed_ms"] < 500, result["elapsed_ms"]


# ---------------------------------------------------------------- 错误定位


def test_wrong_head_features_flags_output_node():
    graph = load_preset("cnn-mnist")
    assert graph is not None
    head = graph.node("head")
    assert head is not None
    head.params["out_features"] = 999

    result = shapes.analyze(graph)
    assert result["ok"] is False
    assert len(result["errors"]) == 1
    error = result["errors"][0]
    assert error["node_id"] == "output"
    assert error["code"] == "shape_mismatch"
    assert error["message_key"] == "errors.node.outputDim"
    assert error["args"] == {"expected": 10, "got": 999, "kind": "classification"}
    # 出错节点之前的节点仍给出形状，之后的节点无结果
    assert result["nodes"]["head"]["out_shape"] == [2, 999]
    assert result["nodes"]["output"]["out_shape"] is None
    assert result["nodes"]["output"]["params"] is None


def test_conv_kernel_too_large_is_localized():
    graph = parse_graph(
        {
            "id": "g",
            "kind": "dl",
            "task": "image_classification",
            "nodes": [
                {"id": "input", "type": "Input", "params": {"shape": [1, 8, 8]}},
                {"id": "conv", "type": "Conv2d", "params": {"out_channels": 4, "kernel_size": 32}},
                {"id": "flat", "type": "Flatten", "params": {}},
                {"id": "head", "type": "Linear", "params": {"out_features": 10}},
                {"id": "output", "type": "Output", "params": {"classes": 10}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv"},
                {"id": "e2", "source": "conv", "target": "flat"},
                {"id": "e3", "source": "flat", "target": "head"},
                {"id": "e4", "source": "head", "target": "output"},
            ],
        }
    )
    result = shapes.analyze(graph)
    assert result["ok"] is False
    assert result["errors"][0]["node_id"] == "conv"


def test_embedding_requires_integer_input():
    graph = parse_graph(
        {
            "id": "g",
            "kind": "dl",
            "task": "image_classification",
            "nodes": [
                {"id": "input", "type": "Input", "params": {"shape": [8]}},
                {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 32, "embedding_dim": 8}},
                {"id": "flat", "type": "Flatten", "params": {}},
                {"id": "head", "type": "Linear", "params": {"out_features": 10}},
                {"id": "output", "type": "Output", "params": {"classes": 10}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "emb"},
                {"id": "e2", "source": "emb", "target": "flat"},
                {"id": "e3", "source": "flat", "target": "head"},
                {"id": "e4", "source": "head", "target": "output"},
            ],
        }
    )
    result = shapes.analyze(graph)
    assert result["ok"] is False
    error = result["errors"][0]
    assert error["node_id"] == "emb"
    assert error["message_key"] == "errors.node.expectInt"


# ---------------------------------------------------------------- 结构校验


def _base_nodes():
    return [
        {"id": "input", "type": "Input", "params": {"shape": [1, 8, 8]}},
        {"id": "flat", "type": "Flatten", "params": {}},
        {"id": "head", "type": "Linear", "params": {"out_features": 10}},
        {"id": "output", "type": "Output", "params": {"classes": 10}},
    ]


def _base_edges():
    return [
        {"id": "e1", "source": "input", "target": "flat"},
        {"id": "e2", "source": "flat", "target": "head"},
        {"id": "e3", "source": "head", "target": "output"},
    ]


def codes(payload: dict) -> set[str]:
    graph = parse_graph(payload)
    issues, _ = validate_graph(graph)
    return {issue.code for issue in issues}


def test_structural_validation_cases():
    assert codes({"nodes": _base_nodes(), "edges": _base_edges()}) == set()

    no_output = {"nodes": _base_nodes()[:3], "edges": _base_edges()[:2]}
    assert "missing_output" in codes(no_output)

    two_inputs = copy.deepcopy(_base_nodes())
    two_inputs.append({"id": "input2", "type": "Input", "params": {"shape": [1, 8, 8]}})
    assert "multiple_inputs" in codes({"nodes": two_inputs, "edges": _base_edges()})

    unknown = copy.deepcopy(_base_nodes())
    unknown[1]["type"] = "Conv3d"
    assert "unknown_node_type" in codes({"nodes": unknown, "edges": _base_edges()})

    dangling_edges = _base_edges() + [{"id": "e9", "source": "ghost", "target": "output"}]
    assert "dangling_edge" in codes({"nodes": _base_nodes(), "edges": dangling_edges})

    cycle_nodes = [
        {"id": "input", "type": "Input", "params": {"shape": [1, 8, 8]}},
        {"id": "flat", "type": "Flatten", "params": {}},
        {"id": "add", "type": "ResidualAdd", "params": {}},
        {"id": "head", "type": "Linear", "params": {"out_features": 10}},
        {"id": "output", "type": "Output", "params": {"classes": 10}},
    ]
    cyclic_edges = [
        {"id": "e1", "source": "input", "target": "flat"},
        {"id": "e2", "source": "flat", "target": "add", "target_port": "in1"},
        {"id": "e3", "source": "add", "target": "head"},
        {"id": "e4", "source": "head", "target": "add", "target_port": "in2"},
        {"id": "e5", "source": "head", "target": "output"},
    ]
    assert "cycle" in codes({"nodes": cycle_nodes, "edges": cyclic_edges})

    unreachable = copy.deepcopy(_base_nodes())
    unreachable.append({"id": "orphan", "type": "Dropout", "params": {"p": 0.5}})
    assert "unreachable_node" in codes({"nodes": unreachable, "edges": _base_edges()})


def test_multi_input_ports_and_limits():
    nodes = [
        {"id": "input", "type": "Input", "params": {"shape": [1, 8, 8]}},
        {"id": "flat", "type": "Flatten", "params": {}},
        {"id": "head", "type": "Linear", "params": {"out_features": 16}},
        {"id": "add", "type": "ResidualAdd", "params": {}},
        {"id": "output", "type": "Output", "params": {"classes": 16}},
    ]
    edges = [
        {"id": "e1", "source": "input", "target": "flat"},
        {"id": "e2", "source": "flat", "target": "head", "target_port": "in"},
        {"id": "e3", "source": "head", "target": "add", "target_port": "in1"},
        {"id": "e4", "source": "head", "target": "add", "target_port": "in2"},
        {"id": "e5", "source": "add", "target": "output"},
    ]
    graph = parse_graph({"id": "g", "kind": "dl", "task": "image_classification", "nodes": nodes, "edges": edges})
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]
    assert result["nodes"]["add"]["in_shapes"] == [[2, 16], [2, 16]]
    assert result["nodes"]["add"]["out_shape"] == [2, 16]

    one_input = parse_graph(
        {
            "id": "g",
            "kind": "dl",
            "task": "image_classification",
            "nodes": nodes,
            "edges": [e for e in edges if e["id"] != "e4"],
        }
    )
    issues, _ = validate_graph(one_input)
    assert "fan_in_missing" in {i.code for i in issues}

    overflow_edges = edges + [
        {"id": "e6", "source": "head", "target": "add", "target_port": "in3"},
        {"id": "e7", "source": "head", "target": "add", "target_port": "in4"},
        {"id": "e8", "source": "head", "target": "add", "target_port": "in5"},
    ]
    overflow = parse_graph(
        {"id": "g", "kind": "dl", "task": "image_classification", "nodes": nodes, "edges": overflow_edges}
    )
    issues, _ = validate_graph(overflow)
    assert "fan_in_overflow" in {i.code for i in issues}


# ---------------------------------------------------------------- 序列 / 注意力


def test_text_graph_with_embedding_attention_lstm():
    graph = parse_graph(
        {
            "id": "tiny-lm",
            "kind": "dl",
            "task": "text_classification",
            "nodes": [
                {"id": "input", "type": "Input", "params": {"shape": [16]}},
                {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 50, "embedding_dim": 32}},
                {"id": "pos", "type": "PositionalEncoding", "params": {"max_len": 64, "mode": "learned"}},
                {"id": "attn", "type": "MultiHeadAttention", "params": {"num_heads": 4, "causal": True}},
                {"id": "lstm", "type": "LSTM", "params": {"hidden_size": 24, "return_sequences": False}},
                {"id": "head", "type": "Linear", "params": {"out_features": 10}},
                {"id": "output", "type": "Output", "params": {"classes": 10}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "emb"},
                {"id": "e2", "source": "emb", "target": "pos"},
                {"id": "e3", "source": "pos", "target": "attn"},
                {"id": "e4", "source": "attn", "target": "lstm"},
                {"id": "e5", "source": "lstm", "target": "head"},
                {"id": "e6", "source": "head", "target": "output"},
            ],
        }
    )
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]
    assert result["nodes"]["emb"]["out_shape"] == [2, 16, 32]
    assert result["nodes"]["attn"]["out_shape"] == [2, 16, 32]
    assert result["nodes"]["lstm"]["out_shape"] == [2, 24]

    reference = nn.LSTM(input_size=32, hidden_size=24, batch_first=True)
    assert result["nodes"]["lstm"]["params"] == sum(p.numel() for p in reference.parameters())


def test_lstm_char_preset_shapes_and_hidden_probe():
    graph = load_preset("lstm-char")
    assert graph is not None
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]
    assert result["nodes"]["emb"]["out_shape"] == [2, 32, 64]
    assert result["nodes"]["lstm1"]["out_shape"] == [2, 32, 64]  # return_sequences
    assert result["nodes"]["head"]["out_shape"] == [2, 32, 76]
    assert parse_graph(graph.to_dict()) is not None  # 预置文件可往返

    probes = probe_hooks.resolve_probes(None, graph)
    assert [spec.node_id for spec in probes] == ["lstm1"]
    assert probes[0].kind == "hidden"  # LSTM 的派生探针类型（docs/02 §6.1）


def test_sinusoidal_encoding_shapes():
    graph = parse_graph(
        {
            "id": "g",
            "kind": "dl",
            "task": "text_classification",
            "nodes": [
                {"id": "input", "type": "Input", "params": {"shape": [16]}},
                {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 50, "embedding_dim": 16}},
                {"id": "pos", "type": "PositionalEncoding", "params": {"max_len": 32, "mode": "sinusoidal"}},
                {"id": "flat", "type": "Flatten", "params": {}},
                {"id": "head", "type": "Linear", "params": {"out_features": 10}},
                {"id": "output", "type": "Output", "params": {"classes": 10}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "emb"},
                {"id": "e2", "source": "emb", "target": "pos"},
                {"id": "e3", "source": "pos", "target": "flat"},
                {"id": "e4", "source": "flat", "target": "head"},
                {"id": "e5", "source": "head", "target": "output"},
            ],
        }
    )
    result = shapes.analyze(graph)
    assert result["ok"], result["errors"]
    assert result["nodes"]["pos"]["out_shape"] == [2, 16, 16]
    assert result["nodes"]["pos"]["params"] == 0


# ---------------------------------------------------------------- 缓存与 API


def test_cache_ignores_ui_coordinates():
    graph = load_preset("mlp-mnist")
    assert graph is not None
    shapes.clear_cache()
    first = shapes.analyze(graph)
    for node in graph.nodes:
        node.ui = {"x": node.ui.get("x", 0) + 500, "y": 42}
    second = shapes.analyze(graph)
    assert len(shapes._cache) == 1
    assert first["total_params"] == second["total_params"]


@pytest.fixture()
def client():
    from app.main import app

    return TestClient(app)


def test_api_infer_and_validate(client):
    graph = load_preset("cnn-mnist")
    assert graph is not None
    payload = graph.to_dict()

    response = client.post("/api/graph/infer", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["nodes"]["conv1"]["out_shape"] == [2, 16, 28, 28]
    assert body["total_params"] == 402_986
    assert body["elapsed_ms"] >= 0

    payload["nodes"][-2]["params"]["out_features"] = 7
    response = client.post("/api/graph/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["errors"][0]["node_id"] == "output"


def test_api_infer_rejects_bad_ir(client):
    response = client.post("/api/graph/infer", json={"nodes": [], "edges": []})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_ir"
    assert body["error"]["message_key"] == "errors.graph.empty"


def test_api_models_list_and_detail(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    groups = response.json()["groups"]
    ids = {m["id"] for m in groups["dl"]}
    assert {"mlp-mnist", "cnn-mnist", "lstm-char"} <= ids
    assert {m["id"] for m in groups["ml"]} >= {
        "ml-linear-blobs",
        "ml-logreg-moons",
        "ml-tree-spiral",
        "ml-forest-moons",
        "ml-svm-circles",
    }
    assert {m["id"] for m in groups["rl"]} == {"rl-gridworld"}
    assert all(m["kind"] == "ml" for m in groups["ml"])

    response = client.get("/api/models/cnn-mnist")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "preset"
    assert body["graph"]["id"] == "cnn-mnist"

    response = client.get("/api/models/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_api_infer_full_serializable(client):
    graph = load_preset("mlp-mnist")
    assert graph is not None
    response = client.post("/api/graph/infer", json=graph.to_dict())
    json.dumps(response.json())
