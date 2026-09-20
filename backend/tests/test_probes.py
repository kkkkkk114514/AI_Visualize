"""M3 探针测试：编码（降采样/量化/因果掩码）、配置解析、采样步挂摘 hook 与流上限。"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
import torch

from app import config
from app.graph.ir import parse_graph, validate_graph
from app.graph.module import build_graph_module
from app.probes import encode, hooks


def decode(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data_b64"])
    dtype = {"uint8": np.uint8, "uint32": "<u4"}[payload["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(payload["shape"])


def dequantize(payload: dict) -> np.ndarray:
    quant = decode(payload).astype(np.float32)
    span = payload["max"] - payload["min"]
    return payload["min"] + quant / 255.0 * span


def tiny_lm_graph() -> dict:
    return {
        "id": "tiny-lm",
        "kind": "dl",
        "task": "text_lm",
        "nodes": [
            {"id": "input", "type": "Input", "params": {"shape": [16]}},
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 32, "embedding_dim": 32}},
            {"id": "pos", "type": "PositionalEncoding", "params": {"max_len": 32, "mode": "learned"}},
            {"id": "attn1", "type": "MultiHeadAttention", "params": {"num_heads": 4, "causal": True}},
            {"id": "add", "type": "ResidualAdd", "params": {}},
            {"id": "norm", "type": "LayerNorm", "params": {"normalized_shape": 0}},
            {"id": "fc1", "type": "Linear", "params": {"out_features": 32}},
            {"id": "gelu", "type": "Activation", "params": {"name": "gelu"}},
            {"id": "head", "type": "Linear", "params": {"out_features": 32}},
            {"id": "output", "type": "Output", "params": {"classes": 32}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "emb"},
            {"id": "e2", "source": "emb", "target": "pos"},
            {"id": "e3", "source": "pos", "target": "attn1"},
            {"id": "e4", "source": "pos", "target": "add", "target_port": "in1"},
            {"id": "e5", "source": "attn1", "target": "add", "target_port": "in2"},
            {"id": "e6", "source": "add", "target": "norm"},
            {"id": "e7", "source": "norm", "target": "fc1"},
            {"id": "e8", "source": "fc1", "target": "gelu"},
            {"id": "e9", "source": "gelu", "target": "head"},
            {"id": "e10", "source": "head", "target": "output"},
        ],
    }


class Harness:
    """搭好 GraphModule + ProbeEngine，模拟若干训练步。"""

    def __init__(self, tmp_path, probes: list[dict] | None):
        graph = parse_graph(tiny_lm_graph())
        issues, params_by_node = validate_graph(graph)
        assert not [i for i in issues if i.severity == "error"]
        self.graph = graph
        self.model = build_graph_module(graph, params_by_node)
        self.model.eval()
        with torch.no_grad():
            self.model(torch.zeros(2, 16, dtype=torch.int64))
        self.model.train()
        self.events: list[dict] = []
        specs = hooks.resolve_probes(probes, graph)
        self.engine = hooks.ProbeEngine(
            specs,
            self.model,
            {node.id: node.type for node in graph.nodes},
            run_id="r_test",
            snapshots_dir=tmp_path,
            emit=self.events.append,
        )
        self.tmp_path = tmp_path

    def step(self, step: int, *, forward: bool = True) -> torch.Tensor | None:
        began = self.engine.begin_step(step, 1)
        output = None
        if forward:
            with torch.no_grad():
                output = self.model(torch.zeros(2, 16, dtype=torch.int64))
        self.engine.end_step()
        return output if began else None

    def snapshots(self) -> list[dict]:
        return [event for event in self.events if event["type"] == "probe"]

    def logs(self) -> list[dict]:
        return [event for event in self.events if event["type"] == "log"]

    def files(self) -> list[str]:
        return sorted(item.name for item in self.tmp_path.glob("*.json"))


# ---------------------------------------------------------------- 编码


def test_encode_feature_grid_downsamples_and_quantizes():
    tensor = torch.randn(2, 64, 48, 48)
    payload = encode.encode("feature_grid", tensor, max_items=32, sample_index=1)
    assert payload["shape"] == [32, 32, 32]
    assert payload["dtype"] == "uint8" and payload["layout"] == "CHW"

    sample = tensor[1, :32]
    pooled = torch.nn.functional.adaptive_avg_pool2d(sample, (32, 32)).numpy()
    assert payload["min"] == pytest.approx(float(pooled.min()), abs=1e-6)
    assert payload["max"] == pytest.approx(float(pooled.max()), abs=1e-6)

    restored = dequantize(payload)
    span = payload["max"] - payload["min"]
    assert np.abs(restored - pooled).max() <= span / 255.0 + 1e-5

    fewer = encode.encode("feature_grid", tensor, max_items=4)
    assert fewer["shape"] == [4, 32, 32]


def test_encode_attention_keeps_causal_zeros():
    weights = torch.rand(1, 6, 16, 16)
    mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    weights = weights.masked_fill(mask, 0.0)
    payload = encode.encode("attention", weights, max_items=4, causal=True)
    assert payload["shape"] == [4, 16, 16]
    assert payload["meta"] == {"heads": 4, "tokens": 16, "causal": True}
    assert payload["min"] == 0.0 and payload["max"] == 1.0
    restored = dequantize(payload)
    upper = np.triu(np.ones((16, 16), dtype=bool), k=1)
    assert np.all(restored[:, upper] == 0.0), "因果掩码上三角必须为 0"

    long_seq = torch.rand(1, 2, 200, 200)
    sampled = encode.encode("attention", long_seq, max_items=8)
    assert sampled["shape"][-1] <= 64


def test_encode_hidden_and_weights():
    hidden = encode.encode("hidden", torch.randn(2, 20, 40), sample_index=1)
    assert hidden["shape"] == [20, 40] and hidden["layout"] == "TH"

    weight = encode.encode("weights", torch.randn(64, 128))
    assert weight["shape"] == [32, 1, 32]

    with pytest.raises(encode.ProbeShapeError):
        encode.encode("feature_grid", torch.randn(16))


def test_encode_histogram_bins_sum_to_numel():
    values = torch.randn(3, 40)
    payload = encode.encode("histogram", values)
    assert payload["dtype"] == "uint32" and payload["shape"] == [32]
    assert payload["meta"] == {"bins": 32}
    assert int(decode(payload).sum()) == values.numel()

    constant = encode.encode("histogram", torch.full((10,), 2.5))
    assert constant["min"] == pytest.approx(2.0) and constant["max"] == pytest.approx(3.0)
    counts = decode(constant)
    assert int(counts.sum()) == 10 and int(np.argmax(counts)) == 16


# ---------------------------------------------------------------- 配置解析


def test_resolve_probes_from_defaults_and_overrides():
    graph = parse_graph(tiny_lm_graph())
    graph.probe_defaults = ["attn1", "fc1"]
    specs = hooks.resolve_probes(None, graph)
    assert [(s.node_id, s.kind) for s in specs] == [("attn1", "attention"), ("fc1", "histogram")]
    assert specs[0].every_n_steps == config.PROBE_DEFAULT_EVERY_N
    assert specs[0].sample_index == 0 and specs[0].max_items == config.PROBE_MAX_ITEMS

    explicit = hooks.resolve_probes(
        [
            {"node_id": "attn1", "kind": "weights", "every_n_steps": 5, "sample_index": 3, "max_items": 2},
            {"node_id": "attn1", "kind": "weights"},
            {"node_id": "head", "kind": "histogram", "every_n_steps": 30},
        ],
        graph,
    )
    assert len(explicit) == 2, "同一 (node, kind) 去重"
    assert explicit[0].kind == "weights" and explicit[0].every_n_steps == 5
    assert explicit[1].node_id == "head"

    for bad in (
        [{"node_id": "nope"}],
        [{"node_id": "fc1", "kind": "rainbow"}],
        [{"node_id": "fc1", "every_n_steps": 0}],
        [{"node_id": "fc1", "sample_index": -1}],
        "attn1",
    ):
        with pytest.raises(hooks.ProbeError):
            hooks.resolve_probes(bad, graph)


# ---------------------------------------------------------------- 采样引擎


def test_engine_samples_only_on_marked_steps(tmp_path):
    harness = Harness(tmp_path, [{"node_id": "fc1", "kind": "histogram", "every_n_steps": 3}])
    assert harness.engine.active()
    module = harness.model.node_module("fc1")

    harness.step(1)
    harness.step(2)
    assert len(module._forward_hooks) == 0, "非采样步不得残留 hook"
    assert harness.snapshots() == []
    assert harness.files() == []

    harness.step(3)
    snapshots = harness.snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["step"] == 3 and snapshots[0]["node_id"] == "fc1"
    assert snapshots[0]["kind"] == "histogram" and snapshots[0]["shape"] == [32]
    assert snapshots[0]["run_id"] == "r_test"
    assert len(module._forward_hooks) == 0, "采样步结束后 hook 必须摘除"

    files = harness.files()
    assert len(files) == 1 and files[0].startswith("s_")
    record = json.loads((tmp_path / files[0]).read_text(encoding="utf-8"))
    assert record["id"] == snapshots[0]["snapshot_id"]
    assert record["step"] == 3 and record["node_id"] == "fc1" and record["kind"] == "histogram"
    assert record["data_b64"] and record["created_at"]

    harness.step(6)
    assert len(harness.snapshots()) == 2


def test_engine_attention_and_weights(tmp_path):
    harness = Harness(
        tmp_path,
        [
            {"node_id": "attn1", "kind": "attention", "every_n_steps": 2},
            {"node_id": "head", "kind": "weights", "every_n_steps": 2},
        ],
    )
    attn_module = harness.model.node_module("attn1")
    harness.step(1)
    assert attn_module.capture_attention is False

    harness.step(2)
    by_kind = {event["kind"]: event for event in harness.snapshots()}
    assert set(by_kind) == {"attention", "weights"}
    record = json.loads((tmp_path / f"{by_kind['attention']['snapshot_id']}.json").read_text("utf-8"))
    assert record["shape"] == [4, 16, 16]
    assert record["meta"] == {"heads": 4, "tokens": 16, "causal": True}
    weights_record = json.loads((tmp_path / f"{by_kind['weights']['snapshot_id']}.json").read_text("utf-8"))
    assert weights_record["shape"] == [32, 1, 32]
    assert attn_module.capture_attention is False, "采样后必须复位 need_weights 开关"
    assert attn_module.last_attn_weights is None, "采样后必须释放注意力权重"


def test_engine_disables_mismatched_kind_and_caps(tmp_path, monkeypatch):
    harness = Harness(tmp_path, [{"node_id": "fc1", "kind": "feature_grid", "every_n_steps": 1}])
    harness.step(1)
    assert harness.snapshots() == []
    assert harness.files() == []
    assert harness.logs()[0]["key"] == "log.probe.disabled"
    assert harness.logs()[0]["args"]["node_id"] == "fc1"
    harness.step(2)
    assert len(harness.logs()) == 1, "已停用的流不再重复告警"

    monkeypatch.setattr(config, "PROBE_MAX_PER_STREAM", 2)
    capped = Harness(tmp_path, [{"node_id": "fc1", "kind": "histogram", "every_n_steps": 1}])
    capped.step(1)
    capped.step(2)
    capped.step(3)
    assert len(capped.snapshots()) == 2
    assert [event["key"] for event in capped.logs()] == ["log.probe.capped"]
