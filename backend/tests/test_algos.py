"""M5 ML 测试：参数 schema / algo spec 校验 / 四个 numpy 自实现 / ml_runner 事件与快照。"""

from __future__ import annotations

import json
import queue
from pathlib import Path

import numpy as np
import pytest

from app.algos import ml, schema as algo_schema, spec as algo_spec
from app.datasets import synth2d
from app.probes import encode
from app.runners import base, ml_runner

SYNTH_IDS = ("moons", "circles", "blobs", "spiral")
ML_ALGOS = ("linear_regression", "logistic_regression", "decision_tree", "random_forest", "svm")


class EventSink:
    """收集子进程事件；只需 `put`，与 mp.Queue 的最小接口一致。"""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def put(self, event: dict) -> None:
        self.events.append(event)

    def of(self, event_type: str) -> list[dict]:
        return [event for event in self.events if event["type"] == event_type]

    def metrics(self) -> list[dict]:
        points: list[dict] = []
        for event in self.of(base.EVENT_METRICS):
            points.extend(event["points"])
        return points

    def statuses(self) -> list[str]:
        return [event["status"] for event in self.of(base.EVENT_STATUS)]


def algo_spec_payload(algo: str, dataset_id: str, every_n: int | None = None, **params) -> dict:
    payload = {
        "ir_version": 1,
        "id": f"inline-{algo}",
        "kind": "ml",
        "task": "binary_classification",
        "name": {"zh": "测试", "en": "test"},
        "algo": algo,
        "params": {"render_delay_ms": 0, **params},
        "dataset_id": dataset_id,
    }
    if every_n is not None:
        payload["probe_defaults"] = {"every_n_steps": every_n}
    return payload


def probe_payload(every_n: int = 5) -> list[dict]:
    return [
        {
            "node_id": "model",
            "kind": "boundary",
            "every_n_steps": every_n,
            "sample_index": 0,
            "max_items": 32,
        }
    ]


def run_ml(
    algo: str,
    dataset_id: str,
    params: dict,
    *,
    tmp_path: Path,
    probes: list[dict] | None = None,
    every_n: int | None = None,
) -> tuple[EventSink, int]:
    sink = EventSink()
    config = {
        "run_id": "r_test",
        "graph": algo_spec_payload(algo, dataset_id, every_n, **params),
        "dataset_id": dataset_id,
        "seed": 0,
        "snapshots_dir": str(tmp_path),
        "probes": probe_payload(5) if probes is None else probes,
    }
    code = ml_runner.run_training(config, sink, queue.Queue())
    return sink, code


# ------------------------------------------------------------------ schema 与 spec


def test_schema_catalog_covers_ml_and_rl():
    catalog = algo_schema.catalog()["algos"]
    for algo in (*ML_ALGOS, "q_learning"):
        assert algo in catalog, algo
    for algo in ML_ALGOS:
        entry = catalog[algo]
        assert entry["kind"] == "ml"
        names = [field["name"] for field in entry["fields"]]
        assert "render_delay_ms" in names
        for field in entry["fields"]:
            assert field["label_key"].startswith("algo.")
            assert field["type"] in ("int", "float", "choice")
            if field["type"] == "choice":
                assert field["choices"]
            else:
                assert field["min"] <= field["default"] <= field["max"]

    svm_fields = {field["name"]: field for field in catalog["svm"]["fields"]}
    assert svm_fields["kernel"]["choices"] == ["linear", "rbf"]
    assert svm_fields["kernel"]["default"] == "rbf"
    assert svm_fields["gamma"]["default"] == 8.0
    assert svm_fields["lr"]["default"] == 1.0
    assert svm_fields["epochs"]["default"] == 4


def test_spec_kind_detection():
    assert algo_spec.spec_kind({"kind": "ml"}) == "ml"
    assert algo_spec.spec_kind({"kind": "rl"}) == "rl"
    assert algo_spec.spec_kind({"kind": "dl"}) is None
    assert algo_spec.spec_kind({"nodes": []}) is None
    assert algo_spec.spec_kind(None) is None


def test_parse_spec_fills_defaults_and_probe_interval():
    spec = algo_spec.parse_spec(algo_spec_payload("decision_tree", "spiral"))
    assert spec.kind == "ml"
    assert spec.task == "binary_classification"
    assert spec.params == {
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_nodes": 63,
        "render_delay_ms": 0,
    }
    assert spec.probe_every_n == 2  # 树默认采样间隔
    assert spec.node_id == "model" and spec.probe_kind == "boundary"
    assert spec.dataset_id == "spiral"

    overridden = algo_spec.parse_spec(algo_spec_payload("logistic_regression", "moons", 7))
    assert overridden.probe_every_n == 7

    raw = algo_spec.parse_spec(algo_spec_payload("svm", "circles"))
    assert (raw.params["kernel"], raw.params["gamma"]) == ("rbf", 8.0)
    assert raw.to_dict()["params"]["render_delay_ms"] == 0


@pytest.mark.parametrize(
    "algo,params,expected",
    [
        ("logistic_regression", {"lr": 99.0}, "errors.algo.badValue"),
        ("logistic_regression", {"epochs": 0}, "errors.algo.badValue"),
        ("logistic_regression", {"epochs": 999}, "errors.algo.badValue"),
        ("decision_tree", {"max_depth": True}, "errors.algo.badValue"),
        ("svm", {"kernel": "poly"}, "errors.algo.badValue"),
        ("svm", {"gamma": 0.0}, "errors.algo.badValue"),
    ],
)
def test_validate_params_rejects_bad_values(algo, params, expected):
    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.parse_spec({**algo_spec_payload(algo, "moons"), "params": params})
    assert excinfo.value.message_key == expected
    assert excinfo.value.code == "invalid_algo"


def test_unknown_algo_and_kind_mismatch():
    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.parse_spec({**algo_spec_payload("logistic_regression", "moons"), "algo": "knn"})
    assert excinfo.value.message_key == "errors.algo.unknownAlgo"

    # kind=ml 但 algo 是 RL 的 → 同样按「未知算法」拒绝
    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.parse_spec({**algo_spec_payload("logistic_regression", "moons"), "algo": "q_learning"})
    assert excinfo.value.message_key == "errors.algo.unknownAlgo"

    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.parse_spec({**algo_spec_payload("svm", "moons"), "kind": "dl"})
    assert excinfo.value.message_key == "errors.algo.badSpec"


def test_resolve_probes_single_stream():
    spec = algo_spec.parse_spec(algo_spec_payload("svm", "circles"))
    default = algo_spec.resolve_probes(None, spec)
    assert len(default) == 1
    assert default[0]["node_id"] == "model" and default[0]["kind"] == "boundary"
    assert default[0]["every_n_steps"] == 20  # SVM 默认间隔

    custom = algo_spec.resolve_probes([{"node_id": "model", "every_n_steps": 3}], spec)
    assert custom[0]["every_n_steps"] == 3

    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.resolve_probes([{"node_id": "conv1", "kind": "boundary"}], spec)
    assert excinfo.value.message_key == "errors.probe.badStream"
    assert excinfo.value.code == "invalid_probe"

    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.resolve_probes([{"node_id": "model", "kind": "grid"}], spec)
    assert excinfo.value.code == "invalid_probe"


def test_parse_env_normalizes_and_validates():
    env = algo_spec.parse_env(
        {
            "width": 12,
            "height": 8,
            "start": [0, 0],
            "goal": [11, 7],
            "obstacles": [[3, 1], [3, 1], [7, 4]],
            "rewards": [[5, 2, 1.0], [5, 2, 2.0]],
        }
    )
    assert env.to_dict()["obstacles"] == [[3, 1], [7, 4]]
    assert env.to_dict()["rewards"] == [[5, 2, 2.0]]  # 同格后写覆盖
    assert env.start == (0, 0) and env.goal == (11, 7)

    for bad in (
        {"start": [0, 0], "goal": [0, 0]},
        {"width": 2, "start": [0, 0], "goal": [1, 1]},
        {"start": [0, 0], "goal": [1, 1], "obstacles": [[30, 1]]},
        {"start": [0, 0], "goal": [1, 1], "obstacles": [[1, 1]]},
        {"start": [0, 0], "goal": [1, 1], "obstacles": [[2, 2]], "rewards": [[2, 2, 1.0]]},
        {"start": [0, 0], "goal": [1, 1], "rewards": [[2, 2, "x"]]},
    ):
        with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
            algo_spec.parse_env(bad)
        assert excinfo.value.message_key == "errors.algo.badEnv"


# ------------------------------------------------------------------ 四个算法


def _fit(algo: str, dataset_id: str, params: dict, seed: int = 0) -> ml.MLModel:
    train_x, train_y, _, _ = synth2d.split_arrays(dataset_id)
    full = algo_schema.get(algo).defaults()
    full.update(params)
    full.setdefault("render_delay_ms", 0)
    model = ml.build(algo, train_x, train_y, full, seed)
    guard = 0
    while not model.finished:
        model.step()
        guard += 1
        assert guard < 100_000, "模型没有收敛到 finished"
    return model


@pytest.mark.parametrize(
    "algo,dataset_id,params,min_val_acc",
    [
        ("linear_regression", "blobs", {"epochs": 40}, 0.90),
        ("logistic_regression", "moons", {"epochs": 40}, 0.80),
        ("logistic_regression", "blobs", {"epochs": 40}, 0.93),
        ("decision_tree", "moons", {}, 0.90),
        ("decision_tree", "spiral", {}, 0.70),
        ("random_forest", "spiral", {"n_estimators": 24}, 0.82),
        ("svm", "circles", {"epochs": 4}, 0.93),
        ("svm", "spiral", {"epochs": 4}, 0.88),
        ("svm", "blobs", {"kernel": "linear", "epochs": 4}, 0.90),
    ],
)
def test_models_actually_learn(algo, dataset_id, params, min_val_acc):
    model = _fit(algo, dataset_id, params)
    val_x, val_y = synth2d.make_points(dataset_id, "val")
    assert model.acc_on(val_x, val_y) >= min_val_acc


@pytest.mark.parametrize("algo", ML_ALGOS)
def test_models_are_seed_deterministic(algo):
    first = _fit(algo, "moons", {})
    second = _fit(algo, "moons", {})
    points = np.linspace(-0.9, 0.9, 40).reshape(-1, 2).astype(np.float64)
    assert np.allclose(first.score(points), second.score(points))


@pytest.mark.parametrize("algo", ML_ALGOS)
def test_scores_are_finite(algo):
    model = _fit(algo, "spiral", {})
    grid = np.linspace(-1.0, 1.0, 21)
    mesh = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    values, _ = model.boundary(mesh)
    assert np.isfinite(values).all()


def test_linear_and_logistic_decision_thresholds():
    linear = _fit("linear_regression", "blobs", {"epochs": 20})
    logistic = _fit("logistic_regression", "blobs", {"epochs": 20})
    assert linear.threshold == 0.5 and logistic.threshold == 0.0
    # 逻辑回归的 score 是 logit：blobs 上两侧符号应分明
    train_x, train_y, _, _ = synth2d.split_arrays("blobs")
    assert float(logistic.score(train_x[train_y == 1]).mean()) > 0.0
    assert float(logistic.score(train_x[train_y == 0]).mean()) < 0.0


def test_tree_grows_monotonically_and_refines_boundary():
    train_x, train_y, _, _ = synth2d.split_arrays("spiral")
    params = algo_schema.get("decision_tree").defaults()
    model = ml.DecisionTree(train_x, train_y, params, 0)
    previous_nodes = 1
    previous_leaves = 1
    assert model.boundary_mode == "label"
    for _ in range(8):
        values = model.step()
        assert values["n_nodes"] >= previous_nodes
        assert values["leaves"] >= previous_leaves
        assert values["depth"] >= 0
        assert values["loss"] <= 1.0 + 1e-9
        previous_nodes, previous_leaves = values["n_nodes"], values["leaves"]

    grid = np.linspace(-0.95, 0.95, 41)
    mesh = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    values, mode = model.boundary(mesh)
    assert mode == "label"
    assert set(np.unique(values)) <= {0.0, 1.0}
    meta = model.boundary_meta()
    assert meta["nodes"] == int(previous_nodes) and meta["leaves"] == int(previous_leaves)


def test_forest_averages_tree_probabilities():
    train_x, train_y, _, _ = synth2d.split_arrays("circles")
    params = algo_schema.get("random_forest").defaults()
    params["n_estimators"] = 6
    model = ml.RandomForest(train_x, train_y, params, 0)
    assert model.boundary_mode == "score"
    before = model.score(train_x)
    assert np.allclose(before, 0.5)  # 空森林 → 0.5
    for _ in range(6):
        model.step()
    assert model.finished
    probs = model.score(train_x)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert model.boundary_meta()["n_trees"] == 6
    assert probs.std() > 0.05


def test_svm_margin_and_support_vectors():
    linear = _fit("svm", "blobs", {"kernel": "linear", "epochs": 4})
    margin = linear.margin()
    assert margin is not None and margin > 0.0
    alpha = linear.alpha
    assert 0 < int(np.count_nonzero(np.abs(alpha) > 1e-8)) < len(alpha)
    meta = linear.boundary_meta()
    assert meta["margin"] == round(margin, 4)
    assert meta["n_sv"] == int(np.count_nonzero(np.abs(alpha) > 1e-8))
    assert len(meta["support_vectors"]) == min(meta["n_sv"], 64)
    assert all(len(point) == 2 for point in meta["support_vectors"])

    # 核模型下 ‖w‖ 无直观含义：不给 margin
    rbf = _fit("svm", "moons", {})
    assert rbf.margin() is None
    assert "margin" not in rbf.boundary_meta()


def test_set_lr_and_batch_size_control_commands():
    model = _fit("logistic_regression", "moons", {"epochs": 2})
    model.set_lr(0.25)
    assert model.lr == 0.25
    model.set_batch_size(100)
    assert model.batch_size == 100
    assert model.steps_per_epoch == 3  # ceil(300/100)

    tree = _fit("decision_tree", "moons", {})
    tree.set_lr(0.9)
    tree.set_batch_size(8)
    assert not hasattr(tree, "lr") and not hasattr(tree, "batch_size")


# ------------------------------------------------------------------ ml_runner


def test_ml_runner_emits_metrics_and_boundary_snapshots(tmp_path):
    sink, code = run_ml(
        "logistic_regression", "moons", {"epochs": 4, "batch_size": 64}, tmp_path=tmp_path
    )
    assert code == 0
    statuses = sink.statuses()
    assert statuses[0] == "running" and statuses[-1] == "finished"
    assert set(statuses) == {"running", "finished"}
    assert not sink.of(base.EVENT_ERROR)

    points = sink.metrics()
    step_points = [point for point in points if "loss" in point["values"]]
    val_points = [point for point in points if "val_acc" in point["values"]]
    assert [point["step"] for point in step_points] == list(range(1, len(step_points) + 1))
    # val_* 挂在该 epoch 最后一个 step 上（与 DL 同惯例）
    assert [point["step"] for point in val_points] == [5, 10, 15, 20]
    assert [point["step"] for point in points] == sorted(point["step"] for point in points)
    assert len(step_points) == 20  # ceil(300/64) * 4 epoch
    assert {point["epoch"] for point in val_points} == {1, 2, 3, 4}
    for point in step_points:
        assert set(point["values"]) == {"loss", "acc", "lr"}
    for point in val_points:
        assert set(point["values"]) == {"val_loss", "val_acc"}

    finished = sink.of(base.EVENT_STATUS)[-1]
    assert finished["best_metric"] is not None
    started = sink.of(base.EVENT_STATUS)[0]
    assert started["device"] == "cpu" and started["planned_steps"] == 20
    assert sink.of(base.EVENT_LOG)[0]["key"] == "log.run.starting"
    assert "log.probe.attached" in [event["key"] for event in sink.of(base.EVENT_LOG)]

    probe_events = sink.of(base.EVENT_PROBE)
    assert [event["step"] for event in probe_events] == [5, 10, 15, 20]
    for event in probe_events:
        assert (event["node_id"], event["kind"]) == ("model", "boundary")
        assert event["shape"] == [96, 96]
        record = json.loads(Path(event["file_path"]).read_text(encoding="utf-8"))
        assert record["id"] == event["snapshot_id"] and record["run_id"] == "r_test"
        assert record["dtype"] == "uint8" and record["layout"] == "HW"
        assert len(record["data_b64"]) > 0
        meta = record["meta"]
        assert meta["mode"] == "score" and meta["algo"] == "logistic_regression"
        assert meta["x_range"][0] < meta["x_range"][1]
        assert meta["y_range"][0] < meta["y_range"][1]
        assert record["min"] < record["max"]

    # 训练进度：epoch 单调不减，step 与 epoch 同步
    running = [event for event in sink.of(base.EVENT_STATUS) if event["status"] == "running"]
    assert [event["epoch"] for event in running] == sorted(event["epoch"] for event in running)


def test_ml_runner_tree_sends_val_only_at_last_step(tmp_path):
    sink, code = run_ml("decision_tree", "spiral", {}, tmp_path=tmp_path)
    assert code == 0
    points = sink.metrics()
    val_points = [point for point in points if "val_acc" in point["values"]]
    assert len(val_points) == 1
    assert val_points[0]["step"] == max(point["step"] for point in points)
    assert val_points[0]["epoch"] == 1
    structure = {
        name for point in points for name in point["values"] if name not in ("val_loss", "val_acc")
    }
    assert structure == {"loss", "acc", "depth", "leaves", "n_nodes"}
    last_probe = sink.of(base.EVENT_PROBE)[-1]
    meta = json.loads(Path(last_probe["file_path"]).read_text(encoding="utf-8"))["meta"]
    assert meta["mode"] == "label"
    assert meta["nodes"] > 1 and meta["leaves"] >= 2

    finished = sink.of(base.EVENT_STATUS)[-1]
    assert finished["epoch"] == 1
    assert finished["step"] <= 31  # max_nodes 63 的上限


def test_ml_runner_forest_and_svm_metrics(tmp_path):
    sink, code = run_ml(
        "random_forest", "moons", {"n_estimators": 4, "max_depth": 4}, tmp_path=tmp_path
    )
    assert code == 0
    names = {name for point in sink.metrics() for name in point["values"]}
    assert names == {"loss", "acc", "n_trees", "val_loss", "val_acc"}
    started = sink.of(base.EVENT_STATUS)[0]
    assert started["planned_steps"] == 4 and started["steps_per_epoch"] == 4
    assert sink.of(base.EVENT_STATUS)[-1]["step"] == 4

    sink, code = run_ml("svm", "blobs", {"kernel": "linear", "epochs": 2}, tmp_path=tmp_path)
    assert code == 0
    names = {name for point in sink.metrics() for name in point["values"]}
    assert names == {"loss", "acc", "lr", "margin", "n_sv", "val_loss", "val_acc"}
    val_steps = [
        point["step"] for point in sink.metrics() if "val_acc" in point["values"]
    ]
    assert val_steps == [300, 600]


def test_ml_runner_without_probes_writes_nothing(tmp_path):
    sink, code = run_ml("linear_regression", "blobs", {"epochs": 2}, tmp_path=tmp_path, probes=[])
    assert code == 0
    assert not sink.of(base.EVENT_PROBE)
    assert not list(tmp_path.iterdir())
    assert "log.probe.attached" not in [event["key"] for event in sink.of(base.EVENT_LOG)]


def test_ml_runner_rejects_wrong_dataset_and_spec(tmp_path):
    sink, code = run_ml("logistic_regression", "mnist", {"epochs": 1}, tmp_path=tmp_path)
    assert code == 1
    error = sink.of(base.EVENT_ERROR)[0]
    assert error["code"] == "dataset_kind_mismatch"
    assert error["message_key"] == "errors.dataset.kindMismatch"
    assert error["args"]["kind"] == "ml"
    assert sink.statuses() == ["failed"]

    sink = EventSink()
    bad = {
        "run_id": "r_test",
        "graph": {**algo_spec_payload("logistic_regression", "moons"), "params": {"lr": 10.0}},
        "dataset_id": "moons",
        "seed": 0,
        "snapshots_dir": str(tmp_path),
        "probes": [],
    }
    assert ml_runner.run_training(bad, sink, queue.Queue()) == 1
    assert sink.of(base.EVENT_ERROR)[0]["message_key"] == "errors.algo.badValue"


def test_boundary_payload_round_trip(tmp_path):
    sink, _ = run_ml("decision_tree", "moons", {}, tmp_path=tmp_path)
    event = sink.of(base.EVENT_PROBE)[0]
    record = json.loads(Path(event["file_path"]).read_text(encoding="utf-8"))
    payload = {
        key: record[key] for key in ("shape", "dtype", "layout", "min", "max", "data_b64", "meta")
    }
    decoded = np.frombuffer(
        __import__("base64").b64decode(payload["data_b64"]), dtype=np.uint8
    )
    assert decoded.size == 96 * 96
    # label 模式：min / max 固定 0 / 1，量化值只落在两端
    assert (payload["min"], payload["max"]) == (0.0, 1.0)
    assert set(np.unique(decoded)) <= {0, 255}
    assert encode.KINDS[-2:] == ("boundary", "grid")
