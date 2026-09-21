"""ML / RL 算法参数 schema（契约见 docs/02 §13.2）：`GET /api/algos` 与服务端校验共用同一份表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INT = "int"
FLOAT = "float"
CHOICE = "choice"


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    default: Any
    label_key: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.type == CHOICE:
            data["choices"] = list(self.choices)
        else:
            data["min"] = self.minimum
            data["max"] = self.maximum
            if self.type == FLOAT:
                data["step"] = self.step
        data["default"] = self.default
        data["label_key"] = self.label_key
        return data


@dataclass(frozen=True)
class AlgoSchema:
    algo: str
    kind: str
    fields: tuple[Field, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "kind": self.kind,
            "fields": [field.to_dict() for field in self.fields],
        }

    def field(self, name: str) -> Field | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def defaults(self) -> dict[str, Any]:
        return {field.name: field.default for field in self.fields}


def _render_delay(default: int) -> Field:
    return Field("render_delay_ms", INT, default, "algo.common.renderDelay", 0, 500)


_GRADIENT_FIELDS = (
    Field("lr", FLOAT, 0.5, "algo.common.lr", 0.01, 1.0, 0.01),
    Field("epochs", INT, 40, "algo.common.epochs", 1, 200),
    Field("batch_size", INT, 32, "algo.common.batchSize", 1, 256),
    Field("l2", FLOAT, 0.0, "algo.common.l2", 0.0, 1.0, 0.001),
)

SCHEMAS: dict[str, AlgoSchema] = {
    schema.algo: schema
    for schema in (
        AlgoSchema("linear_regression", "ml", _GRADIENT_FIELDS + (_render_delay(20),)),
        AlgoSchema("logistic_regression", "ml", _GRADIENT_FIELDS + (_render_delay(20),)),
        AlgoSchema(
            "decision_tree",
            "ml",
            (
                Field("max_depth", INT, 6, "algo.tree.maxDepth", 1, 12),
                Field("min_samples_leaf", INT, 5, "algo.tree.minSamplesLeaf", 1, 100),
                Field("max_nodes", INT, 63, "algo.tree.maxNodes", 3, 255),
                _render_delay(60),
            ),
        ),
        AlgoSchema(
            "random_forest",
            "ml",
            (
                Field("n_estimators", INT, 24, "algo.forest.nEstimators", 1, 64),
                Field("max_depth", INT, 6, "algo.forest.maxDepth", 1, 12),
                Field("min_samples_leaf", INT, 3, "algo.forest.minSamplesLeaf", 1, 100),
                Field("feature_fraction", FLOAT, 0.7, "algo.forest.featureFraction", 0.1, 1.0, 0.05),
                _render_delay(120),
            ),
        ),
        AlgoSchema(
            "svm",
            "ml",
            (
                Field("kernel", CHOICE, "rbf", "algo.svm.kernel", choices=("linear", "rbf")),
                Field("C", FLOAT, 1.0, "algo.svm.c", 0.05, 20.0, 0.05),
                Field("gamma", FLOAT, 8.0, "algo.svm.gamma", 0.5, 32.0, 0.5),
                Field("lr", FLOAT, 1.0, "algo.svm.lr", 0.05, 5.0, 0.05),
                Field("epochs", INT, 4, "algo.common.epochs", 1, 50),
                _render_delay(5),
            ),
        ),
        AlgoSchema(
            "q_learning",
            "rl",
            (
                Field("episodes", INT, 250, "algo.qlearning.episodes", 1, 2000),
                Field("max_steps", INT, 60, "algo.qlearning.maxSteps", 1, 500),
                Field("alpha", FLOAT, 0.2, "algo.qlearning.alpha", 0.001, 1.0, 0.001),
                Field("gamma", FLOAT, 0.95, "algo.qlearning.gamma", 0.0, 0.999, 0.001),
                Field("epsilon_start", FLOAT, 0.5, "algo.qlearning.epsilonStart", 0.0, 1.0, 0.01),
                Field("epsilon_end", FLOAT, 0.05, "algo.qlearning.epsilonEnd", 0.0, 1.0, 0.01),
                Field("epsilon_decay", FLOAT, 0.98, "algo.qlearning.epsilonDecay", 0.5, 1.0, 0.001),
                _render_delay(5),
            ),
        ),
    )
}


def get(algo: str) -> AlgoSchema | None:
    return SCHEMAS.get(algo)


def kind_of(algo: str) -> str | None:
    schema = SCHEMAS.get(algo)
    return schema.kind if schema else None


def catalog() -> dict[str, Any]:
    return {"algos": {name: schema.to_dict() for name, schema in SCHEMAS.items()}}
