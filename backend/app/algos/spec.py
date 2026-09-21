"""algo spec（ML / RL 的模型定义，契约见 docs/02 §13.1–§13.3）：解析、校验与规范化。

校验失败 → `AlgoSpecError`：API 层按 `code` 转 422（`invalid_algo` / `invalid_probe`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app import config
from app.algos import schema as algo_schema

KIND_ML = "ml"
KIND_RL = "rl"
ALGO_KINDS = (KIND_ML, KIND_RL)

IR_VERSION = 1
GRID_MIN, GRID_MAX = 4, 24
REWARD_MIN, REWARD_MAX = -10.0, 10.0

# 单流探针（docs/02 §13.4）：node_id:kind
SINGLE_STREAM: dict[str, tuple[str, str]] = {
    KIND_ML: ("model", "boundary"),
    KIND_RL: ("agent", "grid"),
}
TASK_OF_KIND = {KIND_ML: "binary_classification", KIND_RL: "control"}
DEFAULT_EVERY_N: dict[str, int] = {
    "linear_regression": 10,
    "logistic_regression": 10,
    "decision_tree": 2,
    "random_forest": 1,
    "svm": 20,
    "q_learning": 80,
}


class AlgoSpecError(Exception):
    """algo spec 非法：带 i18n key、字段名与错误码，API 层转 422。"""

    def __init__(
        self,
        message_key: str,
        error_args: dict[str, Any] | None = None,
        detail: str = "",
        code: str = "invalid_algo",
    ):
        super().__init__(detail or message_key)
        self.message_key = message_key
        self.error_args = error_args or {}
        self.detail = detail
        self.code = code


def spec_kind(raw: Any) -> str | None:
    """请求体里的 `graph` 是 DL 图 IR 还是 ML / RL algo spec（kind 缺省按 dl）。"""
    if isinstance(raw, dict) and raw.get("kind") in ALGO_KINDS:
        return str(raw["kind"])
    return None


# ---------------------------------------------------------------- 参数


def _number(value: Any, field: algo_schema.Field) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgoSpecError("errors.algo.badValue", {"param": field.name})
    number = float(value)
    if not math.isfinite(number) or number < float(field.minimum) or number > float(field.maximum):
        raise AlgoSpecError(
            "errors.algo.badValue",
            {"param": field.name, "min": field.minimum, "max": field.maximum},
        )
    return number if field.type == algo_schema.FLOAT else int(round(number))


def validate_params(schema: algo_schema.AlgoSchema, raw: Any) -> dict[str, Any]:
    """按 schema 归一参数：缺项取默认值，类型 / 范围 / 枚举不符即抛错。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AlgoSpecError("errors.algo.badSpec", {"param": "params"}, detail=type(raw).__name__)
    resolved: dict[str, Any] = {}
    for field in schema.fields:
        value = raw.get(field.name, field.default)
        if field.type == algo_schema.CHOICE:
            if not isinstance(value, str) or value not in field.choices:
                raise AlgoSpecError(
                    "errors.algo.badValue",
                    {"param": field.name, "choices": ", ".join(field.choices)},
                )
            resolved[field.name] = value
            continue
        resolved[field.name] = _number(value, field)
    return resolved


# ---------------------------------------------------------------- 环境


@dataclass(frozen=True)
class EnvSpec:
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: tuple[tuple[int, int], ...]
    rewards: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [list(cell) for cell in self.obstacles],
            "rewards": [[x, y, value] for x, y, value in self.rewards],
        }


def _cell(value: Any, param: str, env: dict[str, Any]) -> tuple[int, int]:
    width = int(env["width"])
    height = int(env["height"])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AlgoSpecError("errors.algo.badEnv", {"param": param}, detail=repr(value))
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise AlgoSpecError("errors.algo.badEnv", {"param": param}, detail=repr(value))
    ix, iy = int(x), int(y)
    if ix < 0 or ix >= width or iy < 0 or iy >= height:
        raise AlgoSpecError(
            "errors.algo.badEnv", {"param": param}, detail=f"cell {ix},{iy} out of {width}x{height}"
        )
    return ix, iy


def parse_env(raw: Any) -> EnvSpec:
    if not isinstance(raw, dict):
        raise AlgoSpecError("errors.algo.badEnv", {"param": "env"}, detail=type(raw).__name__)

    size: dict[str, int] = {}
    for name, default in (("width", 12), ("height", 8)):
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AlgoSpecError("errors.algo.badEnv", {"param": name}, detail=repr(value))
        number = int(round(float(value)))
        if number < GRID_MIN or number > GRID_MAX:
            raise AlgoSpecError(
                "errors.algo.badEnv", {"param": name}, detail=f"{number} not in [{GRID_MIN},{GRID_MAX}]"
            )
        size[name] = number

    start = _cell(raw.get("start"), "start", size)
    goal = _cell(raw.get("goal"), "goal", size)
    if start == goal:
        raise AlgoSpecError("errors.algo.badEnv", {"param": "goal"}, detail="goal == start")

    raw_obstacles = raw.get("obstacles") or []
    if not isinstance(raw_obstacles, list):
        raise AlgoSpecError("errors.algo.badEnv", {"param": "obstacles"})
    obstacles: list[tuple[int, int]] = []
    for entry in raw_obstacles:
        cell = _cell(entry, "obstacles", size)
        if cell == start or cell == goal:
            raise AlgoSpecError(
                "errors.algo.badEnv", {"param": "obstacles"}, detail=f"obstacle on {cell}"
            )
        if cell not in obstacles:
            obstacles.append(cell)

    raw_rewards = raw.get("rewards") or []
    if not isinstance(raw_rewards, list):
        raise AlgoSpecError("errors.algo.badEnv", {"param": "rewards"})
    rewards: list[tuple[int, int, float]] = []
    for entry in raw_rewards:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise AlgoSpecError("errors.algo.badEnv", {"param": "rewards"}, detail=repr(entry))
        cell = _cell(entry[:2], "rewards", size)
        value = entry[2]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AlgoSpecError("errors.algo.badEnv", {"param": "rewards"}, detail=repr(entry))
        if cell in obstacles:
            raise AlgoSpecError(
                "errors.algo.badEnv", {"param": "rewards"}, detail=f"reward on obstacle {cell}"
            )
        rewards = [item for item in rewards if item[:2] != cell]
        rewards.append((cell[0], cell[1], float(value)))

    return EnvSpec(
        width=size["width"],
        height=size["height"],
        start=start,
        goal=goal,
        obstacles=tuple(obstacles),
        rewards=tuple(rewards),
    )


# ---------------------------------------------------------------- spec


@dataclass(frozen=True)
class AlgoSpec:
    ir_version: int
    id: str
    kind: str
    task: str
    name: dict[str, str]
    desc: dict[str, str]
    algo: str
    params: dict[str, Any]
    env: EnvSpec | None
    probe_every_n: int
    dataset_id: str | None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ir_version": self.ir_version,
            "id": self.id,
            "kind": self.kind,
            "task": self.task,
            "name": dict(self.name),
            "desc": dict(self.desc),
            "algo": self.algo,
            "params": dict(self.params),
            "probe_defaults": {"every_n_steps": self.probe_every_n},
        }
        if self.env is not None:
            data["env"] = self.env.to_dict()
        if self.dataset_id:
            data["dataset_id"] = self.dataset_id
        return data

    @property
    def node_id(self) -> str:
        return SINGLE_STREAM[self.kind][0]

    @property
    def probe_kind(self) -> str:
        return SINGLE_STREAM[self.kind][1]


def _text_pair(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AlgoSpecError("errors.algo.badSpec", {"param": "name"}, detail=type(value).__name__)
    return {str(key): str(item) for key, item in value.items() if isinstance(item, str)}


def parse_spec(raw: Any) -> AlgoSpec:
    """请求里的 algo spec → 规范化的 `AlgoSpec`（参数已套默认值并校验）。"""
    if not isinstance(raw, dict):
        raise AlgoSpecError("errors.algo.badSpec", detail=type(raw).__name__)

    kind = raw.get("kind")
    if kind not in ALGO_KINDS:
        raise AlgoSpecError("errors.algo.badSpec", {"param": "kind"}, detail=repr(kind))

    algo = raw.get("algo")
    schema = algo_schema.get(algo) if isinstance(algo, str) else None
    if schema is None or schema.kind != kind:
        raise AlgoSpecError("errors.algo.unknownAlgo", {"algo": algo, "kind": kind})

    params = validate_params(schema, raw.get("params"))
    env = parse_env(raw.get("env")) if kind == KIND_RL else None

    raw_every = (raw.get("probe_defaults") or {}).get("every_n_steps") if isinstance(raw.get("probe_defaults"), dict) else None
    if raw_every is None:
        every_n = DEFAULT_EVERY_N.get(str(algo), config.PROBE_DEFAULT_EVERY_N)
    else:
        field = algo_schema.Field("every_n_steps", algo_schema.INT, 1, "algo.common.everyN", 1, 100_000)
        every_n = int(_number(raw_every, field))

    name = _text_pair(raw.get("name"))
    desc = _text_pair(raw.get("desc"))
    spec_id = raw.get("id")
    dataset_id = raw.get("dataset_id")
    task = raw.get("task")
    return AlgoSpec(
        ir_version=int(raw.get("ir_version") or IR_VERSION),
        id=str(spec_id) if isinstance(spec_id, str) and spec_id else f"inline-{algo}",
        kind=str(kind),
        task=str(task) if isinstance(task, str) and task else TASK_OF_KIND[str(kind)],
        name=name,
        desc=desc,
        algo=str(algo),
        params=params,
        env=env,
        probe_every_n=every_n,
        dataset_id=str(dataset_id) if isinstance(dataset_id, str) and dataset_id else None,
    )


def resolve_probes(raw: Any, spec: AlgoSpec) -> list[dict[str, Any]]:
    """固定单流（docs/02 §13.4）：缺省按 spec 的探针间隔；不匹配条目 → 422 `invalid_probe`。"""
    node_id, kind = spec.node_id, spec.probe_kind
    expected = f"{node_id}:{kind}"
    if raw is None:
        entries: list[Any] = [{"node_id": node_id}]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise AlgoSpecError(
            "errors.probe.badConfig", {"param": "probes"}, detail=type(raw).__name__, code="invalid_probe"
        )

    field = algo_schema.Field("every_n_steps", algo_schema.INT, 1, "algo.common.everyN", 1, 100_000)
    every: int | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            raise AlgoSpecError(
                "errors.probe.badConfig", {"param": "probes"}, detail=type(entry).__name__, code="invalid_probe"
            )
        entry_kind = entry.get("kind") or kind
        if entry.get("node_id") != node_id or entry_kind != kind:
            raise AlgoSpecError(
                "errors.probe.badStream", {"expected": expected}, code="invalid_probe"
            )
        if every is None:
            every = int(_number(entry.get("every_n_steps", spec.probe_every_n), field))
    return [
        {
            "node_id": node_id,
            "kind": kind,
            "every_n_steps": spec.probe_every_n if every is None else every,
            "sample_index": 0,
            "max_items": config.PROBE_MAX_ITEMS,
        }
    ]
