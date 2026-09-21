"""四个经典 ML 算法的 numpy 自实现（契约见 docs/02 §13.2）：step 级迭代、可查决策场。

不引入 scikit-learn（D10）：step 级迭代与树的逐层生长必须由我们控制。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-9
MAX_SUPPORT_VECTORS = 64


# ---------------------------------------------------------------- 通用


class MLModel:
    """step 驱动的二分类模型：`step()` 走一步、`score()` 给决策场、`predict()` 给类别。

    `score` 的语义是「越大越像类别 1」的实值决策函数，判定阈值由 `threshold` 给出
    （线性 0.5 / 逻辑与 SVM 取 0 / 概率类 0.5）。
    """

    algo = ""
    boundary_mode = "score"
    threshold = 0.5

    def __init__(
        self, train_x: np.ndarray, train_y: np.ndarray, params: dict[str, Any], seed: int
    ) -> None:
        self.train_x = np.ascontiguousarray(train_x, dtype=np.float64)
        self.train_y = np.ascontiguousarray(train_y, dtype=np.float64)
        self.params = dict(params)
        self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.steps_per_epoch = 1
        self.total_steps = 1

    # ---- 驱动

    def step(self) -> dict[str, float]:
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        """是否已无步可走（树 / 森林在长完后自然结束，不受 total_steps 影响）。"""
        return self.step_index >= self.total_steps

    def set_lr(self, value: float) -> None:
        """`set_lr` 控制命令：连续型算法覆盖，离散生长类静默忽略。"""

    def set_batch_size(self, value: int) -> None:
        """`set_batch_size` 控制命令（下一 epoch 生效）：梯度类算法覆盖。"""

    # ---- 决策

    def score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) > self.threshold).astype(np.int64)

    def acc_on(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y, dtype=np.float64)))

    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def boundary_meta(self) -> dict[str, Any]:
        return {}

    def boundary(self, points: np.ndarray) -> tuple[np.ndarray, str]:
        """决策网格取值：`label` 模式给类别下标（0/1），`score` 模式给实值决策函数。"""
        if self.boundary_mode == "label":
            return self.predict(points).astype(np.float64), "label"
        return self.score(points), "score"


class _BatchGD(MLModel):
    """逐 mini-batch 梯度下降的公共部分（线性 / 逻辑回归）。"""

    def __init__(self, train_x, train_y, params, seed) -> None:
        super().__init__(train_x, train_y, params, seed)
        self.lr = float(params["lr"])
        self.l2 = float(params["l2"])
        self.epochs = int(params["epochs"])
        self.batch_size = int(params["batch_size"])
        self.w = np.zeros(self.train_x.shape[1], dtype=np.float64)
        self.b = 0.0
        self._order = self.rng.permutation(len(self.train_x))
        self._cursor = 0
        self._recount()

    def _recount(self) -> None:
        self.steps_per_epoch = max(1, math.ceil(len(self.train_x) / self.batch_size))
        self.total_steps = self.epochs * self.steps_per_epoch

    def set_lr(self, value: float) -> None:
        self.lr = max(1e-6, float(value))

    def set_batch_size(self, value: int) -> None:
        self.batch_size = max(1, min(int(value), 256))
        self._order = self.rng.permutation(len(self.train_x))
        self._cursor = 0
        self._recount()

    def _batch(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cursor >= len(self.train_x):
            self._order = self.rng.permutation(len(self.train_x))
            self._cursor = 0
        index = self._order[self._cursor : self._cursor + self.batch_size]
        self._cursor += len(index)
        return self.train_x[index], self.train_y[index]


# ---------------------------------------------------------------- 线性 / 逻辑回归


class LinearRegression(_BatchGD):
    """线性打分 `s(x) = w·x + b`，½MSE 训练；决策线 `s(x) = 0.5`。"""

    algo = "linear_regression"

    def step(self) -> dict[str, float]:
        batch_x, batch_y = self._batch()
        scores = batch_x @ self.w + self.b
        error = scores - batch_y
        count = len(batch_x)
        self.w -= self.lr * (batch_x.T @ error / count + self.l2 * self.w)
        self.b -= self.lr * float(error.mean())
        self.step_index += 1
        return {
            "loss": float(0.5 * np.mean(error * error)),
            "acc": float(np.mean((scores > 0.5) == batch_y)),
            "lr": self.lr,
        }

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.w + self.b

    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        error = self.score(x) - np.asarray(y, dtype=np.float64)
        return float(0.5 * np.mean(error * error))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class LogisticRegression(_BatchGD):
    """sigmoid + 交叉熵；决策线是 logit `s(x) = 0`。"""

    algo = "logistic_regression"
    threshold = 0.0

    def step(self) -> dict[str, float]:
        batch_x, batch_y = self._batch()
        logits = batch_x @ self.w + self.b
        prob = _sigmoid(logits)
        error = prob - batch_y
        count = len(batch_x)
        self.w -= self.lr * (batch_x.T @ error / count + self.l2 * self.w)
        self.b -= self.lr * float(error.mean())
        self.step_index += 1
        return {
            "loss": float(-np.mean(batch_y * np.log(prob + EPS) + (1.0 - batch_y) * np.log(1.0 - prob + EPS))),
            "acc": float(np.mean((prob > 0.5) == batch_y)),
            "lr": self.lr,
        }

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.w + self.b

    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        prob = np.clip(_sigmoid(self.score(x)), EPS, 1.0 - EPS)
        target = np.asarray(y, dtype=np.float64)
        return float(-np.mean(target * np.log(prob) + (1.0 - target) * np.log(1.0 - prob)))


# ---------------------------------------------------------------- 树与森林


@dataclass(eq=False)
class _TreeNode:
    indices: np.ndarray
    depth: int
    prob: float
    feature: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    p = float(labels.mean())
    return 2.0 * p * (1.0 - p)


def _best_split(
    x: np.ndarray, y: np.ndarray, features: np.ndarray, min_leaf: int
) -> tuple[float, int, float] | None:
    """最佳 (加权不纯度下降, 特征, 阈值)：只考察相邻不同值的中点，两侧都不少于 min_leaf。"""
    total = len(y)
    parent = _gini(y)
    best: tuple[float, int, float] | None = None
    left_n = np.arange(1, total, dtype=np.float64)
    right_n = total - left_n
    keep = (left_n >= min_leaf) & (right_n >= min_leaf)
    if not keep.any():
        return None
    for feature in features:
        column = x[:, int(feature)]
        order = np.argsort(column, kind="stable")
        values = column[order]
        cumulative = np.cumsum(y[order])
        eligible = (values[1:] > values[:-1]) & keep
        if not eligible.any():
            continue
        p_left = cumulative[:-1] / left_n
        p_right = (cumulative[-1] - cumulative[:-1]) / right_n
        impurity = (
            left_n * 2.0 * p_left * (1.0 - p_left) + right_n * 2.0 * p_right * (1.0 - p_right)
        ) / total
        gains = np.where(eligible, parent - impurity, -np.inf)
        index = int(np.argmax(gains))
        gain = float(gains[index])
        if best is None or gain > best[0]:
            best = (gain, int(feature), float((values[index] + values[index + 1]) / 2.0))
    return best


class _Grower:
    """最佳优先（best-first）树生长状态机：`split_once()` 执行一次全局最优分裂。

    单树每 step 调一次 `split_once`，森林一次 `grow()` 长到不能再长。
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        *,
        max_depth: int,
        min_leaf: int,
        max_nodes: int,
        feature_fraction: float,
        rng: np.random.Generator,
    ) -> None:
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_nodes = max_nodes
        self.feature_fraction = feature_fraction
        self.rng = rng
        self.dim = int(x.shape[1])
        self.root = _TreeNode(indices=indices, depth=0, prob=float(y[indices].mean()))
        self.nodes: list[_TreeNode] = [self.root]
        self.leaves: list[_TreeNode] = [self.root]
        self.frontier: list[tuple[float, int, _TreeNode, int, float]] = []
        self._serial = 0
        self._offer(self.root)

    @property
    def done(self) -> bool:
        return not self.frontier

    def _features(self) -> np.ndarray:
        if self.feature_fraction >= 1.0:
            return np.arange(self.dim)
        count = max(1, int(round(self.dim * self.feature_fraction)))
        return self.rng.choice(self.dim, size=count, replace=False)

    def _offer(self, node: _TreeNode) -> None:
        if node.depth >= self.max_depth or node.indices.size < 2 * self.min_leaf:
            return
        if self.max_nodes and len(self.nodes) + 2 > self.max_nodes:
            return
        local = self.x[node.indices]
        best = _best_split(local, self.y[node.indices], self._features(), self.min_leaf)
        if best is None:
            return
        gain, feature, threshold = best
        self._serial += 1
        heapq.heappush(self.frontier, (-gain, self._serial, node, feature, threshold))

    def split_once(self) -> bool:
        if not self.frontier:
            return False
        _, _, node, feature, threshold = heapq.heappop(self.frontier)
        mask = self.x[node.indices, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = _TreeNode(
            node.indices[mask], node.depth + 1, float(self.y[node.indices[mask]].mean())
        )
        node.right = _TreeNode(
            node.indices[~mask], node.depth + 1, float(self.y[node.indices[~mask]].mean())
        )
        self.nodes += [node.left, node.right]
        self.leaves.remove(node)
        self.leaves += [node.left, node.right]
        self._offer(node.left)
        self._offer(node.right)
        return True

    def grow(self) -> _TreeNode:
        while self.split_once():
            pass
        return self.root


def _tree_probs(root: _TreeNode, points: np.ndarray) -> np.ndarray:
    """按树结构分区求每点的叶类别 1 频率（向量化：每层只在子集上比较）。"""
    out = np.empty(len(points), dtype=np.float64)
    stack: list[tuple[_TreeNode, np.ndarray]] = [(root, np.arange(len(points)))]
    while stack:
        node, index = stack.pop()
        if node.is_leaf or index.size == 0:
            out[index] = node.prob
            continue
        mask = points[index, node.feature] <= node.threshold
        stack.append((node.left, index[mask]))
        stack.append((node.right, index[~mask]))
    return out


def _leaf_ids(root: _TreeNode, points: np.ndarray) -> tuple[np.ndarray, list[_TreeNode]]:
    out = np.full(len(points), -1, dtype=np.int64)
    leaves: list[_TreeNode] = []
    stack: list[tuple[_TreeNode, np.ndarray]] = [(root, np.arange(len(points)))]
    while stack:
        node, index = stack.pop()
        if index.size == 0:
            continue
        if node.is_leaf:
            out[index] = len(leaves)
            leaves.append(node)
            continue
        mask = points[index, node.feature] <= node.threshold
        stack.append((node.left, index[mask]))
        stack.append((node.right, index[~mask]))
    return out, leaves


def _partition_gini(root: _TreeNode, points: np.ndarray, labels: np.ndarray) -> float:
    """按树给出的分区计算标签的加权 Gini（树与森林的 loss 都用它，对 train / val 同一口径）。"""
    if len(points) == 0:
        return 0.0
    ids, _ = _leaf_ids(root, points)
    total = len(labels)
    loss = 0.0
    for leaf in np.unique(ids):
        mask = ids == leaf
        loss += float(mask.sum()) / total * _gini(labels[mask])
    return loss


class DecisionTree(MLModel):
    """CART 二分类树：step = 一次全局增益最大的分裂（epoch 恒 1）。"""

    algo = "decision_tree"
    boundary_mode = "label"

    def __init__(self, train_x, train_y, params, seed) -> None:
        super().__init__(train_x, train_y, params, seed)
        self.max_depth = int(params["max_depth"])
        self.min_leaf = int(params["min_samples_leaf"])
        self.max_nodes = int(params["max_nodes"])
        self._grower = _Grower(
            self.train_x,
            self.train_y,
            np.arange(len(self.train_x)),
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
            max_nodes=self.max_nodes,
            feature_fraction=1.0,
            rng=self.rng,
        )
        self.steps_per_epoch = self.total_steps = max(1, (self.max_nodes - 1) // 2)

    @property
    def finished(self) -> bool:
        return self._grower.done

    def step(self) -> dict[str, float]:
        if not self._grower.split_once():
            return {}
        self.step_index += 1
        return self._metrics()

    def _metrics(self) -> dict[str, float]:
        leaves = self._grower.leaves
        probs = np.array([leaf.prob for leaf in leaves], dtype=np.float64)
        fractions = np.array([leaf.indices.size for leaf in leaves], dtype=np.float64) / len(self.train_x)
        return {
            "loss": float(np.sum(fractions * 2.0 * probs * (1.0 - probs))),
            "acc": self.acc_on(self.train_x, self.train_y),
            "depth": float(max(leaf.depth for leaf in leaves)),
            "leaves": float(len(leaves)),
            "n_nodes": float(len(self._grower.nodes)),
        }

    def score(self, x: np.ndarray) -> np.ndarray:
        return _tree_probs(self._grower.root, np.asarray(x, dtype=np.float64))

    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        return _partition_gini(self._grower.root, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))

    def boundary_meta(self) -> dict[str, Any]:
        leaves = self._grower.leaves
        return {
            "depth": int(max(leaf.depth for leaf in leaves)),
            "leaves": len(leaves),
            "nodes": len(self._grower.nodes),
        }


class RandomForest(MLModel):
    """Bagging：step = 生长一棵完整树（bootstrap + 每次分裂的特征子采样），软投票。"""

    algo = "random_forest"

    def __init__(self, train_x, train_y, params, seed) -> None:
        super().__init__(train_x, train_y, params, seed)
        self.n_estimators = int(params["n_estimators"])
        self.max_depth = int(params["max_depth"])
        self.min_leaf = int(params["min_samples_leaf"])
        self.feature_fraction = float(params["feature_fraction"])
        self.trees: list[_TreeNode] = []
        self.steps_per_epoch = self.total_steps = self.n_estimators

    @property
    def finished(self) -> bool:
        return len(self.trees) >= self.n_estimators

    def step(self) -> dict[str, float]:
        if self.finished:
            return {}
        size = len(self.train_x)
        bootstrap = self.rng.integers(0, size, size=size)
        grower = _Grower(
            self.train_x,
            self.train_y,
            bootstrap,
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
            max_nodes=0,
            feature_fraction=self.feature_fraction,
            rng=self.rng,
        )
        self.trees.append(grower.grow())
        self.step_index += 1
        return {
            "loss": self.loss_on(self.train_x, self.train_y),
            "acc": self.acc_on(self.train_x, self.train_y),
            "n_trees": float(len(self.trees)),
        }

    def score(self, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=np.float64)
        if not self.trees:
            return np.full(len(points), 0.5)
        total = np.zeros(len(points), dtype=np.float64)
        for tree in self.trees:
            total += _tree_probs(tree, points)
        return total / len(self.trees)

    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        prob = np.clip(self.score(x), EPS, 1.0 - EPS)
        target = np.asarray(y, dtype=np.float64)
        return float(-np.mean(target * np.log(prob) + (1.0 - target) * np.log(1.0 - prob)))

    def boundary_meta(self) -> dict[str, Any]:
        leaves = [leaf for tree in self.trees for leaf in _leaf_ids(tree, self.train_x)[1]]
        depth = sum(leaf.depth for leaf in leaves) / len(leaves) if leaves else 0.0
        return {"n_trees": len(self.trees), "depth": round(depth, 1)}


# ---------------------------------------------------------------- SVM


def _kernel_matrix(
    a: np.ndarray, b: np.ndarray, kernel: str, gamma: float
) -> np.ndarray:
    if kernel == "linear":
        return a @ b.T
    # 用 ‖a‖² + ‖b‖² − 2a·b 求平方距离，避免 (n,m,d) 的中间张量
    squared = (
        np.square(a).sum(axis=1)[:, None]
        + np.square(b).sum(axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.exp(-gamma * np.maximum(squared, 0.0))


class SupportVectorMachine(MLModel):
    """核化 Pegasos：step = 一个样本的次梯度更新（hinge + L2）。

    步长按 Pegasos 的 `1/t` 收缩（`η_t = lr·C·n/t`）——`lr` 是步长倍率而非绝对值，
    闭式调度保证 α 有界、老支持向量的影响按 1/t 衰减。
    """

    algo = "svm"
    threshold = 0.0

    def __init__(self, train_x, train_y, params, seed) -> None:
        super().__init__(train_x, train_y, params, seed)
        self.kernel = str(params["kernel"])
        self.C = float(params["C"])
        self.gamma = float(params["gamma"])
        self.lr = float(params["lr"])
        self.epochs = int(params["epochs"])
        self.signed = np.where(self.train_y > 0.5, 1.0, -1.0)
        self.alpha = np.zeros(len(self.train_x), dtype=np.float64)
        self.b = 0.0
        self.kernel_train = _kernel_matrix(self.train_x, self.train_x, self.kernel, self.gamma)
        self.steps_per_epoch = len(self.train_x)
        self.total_steps = self.epochs * self.steps_per_epoch
        self._order = self.rng.permutation(len(self.train_x))
        self._cursor = 0

    def set_lr(self, value: float) -> None:
        self.lr = max(1e-6, float(value))

    def step(self) -> dict[str, float]:
        if self._cursor >= len(self.train_x):
            self._order = self.rng.permutation(len(self.train_x))
            self._cursor = 0
        index = int(self._order[self._cursor])
        self._cursor += 1
        t = self.step_index + 1
        margin = self.signed[index] * float(self.kernel_train[index] @ self.alpha + self.b)
        self.alpha *= 1.0 - 1.0 / t
        if margin < 1.0:
            step = self.lr * self.C * len(self.train_x) / t
            self.alpha[index] += step * self.signed[index]
            self.b += step * self.signed[index]
        self.step_index += 1
        return self._metrics()

    def _values(self) -> np.ndarray:
        return self.kernel_train @ self.alpha + self.b

    def _metrics(self) -> dict[str, float]:
        values = self._values()
        metrics: dict[str, float] = {
            "loss": float(np.mean(np.maximum(0.0, 1.0 - self.signed * values))),
            "acc": float(np.mean((values > 0.0) == (self.train_y > 0.5))),
            "lr": self.lr,
            "n_sv": float(np.count_nonzero(np.abs(self.alpha) > 1e-8)),
        }
        margin = self.margin()
        if margin is not None:
            metrics["margin"] = margin
        return metrics

    def margin(self) -> float | None:
        """间隔 2/‖w‖：只有线性核的 ‖w‖ 有直观含义（核模型下是 RKHS 范数）。"""
        if self.kernel != "linear":
            return None
        norm_sq = float(self.alpha @ (self.kernel_train @ self.alpha))
        return 2.0 / math.sqrt(norm_sq) if norm_sq > 1e-12 else 0.0

    def score(self, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=np.float64)
        return _kernel_matrix(points, self.train_x, self.kernel, self.gamma) @ self.alpha + self.b

    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        signed = np.where(np.asarray(y, dtype=np.float64) > 0.5, 1.0, -1.0)
        return float(np.mean(np.maximum(0.0, 1.0 - signed * self.score(x))))

    def boundary_meta(self) -> dict[str, Any]:
        weights = np.abs(self.alpha)
        support = np.flatnonzero(weights > 1e-8)
        meta: dict[str, Any] = {"n_sv": int(support.size)}
        margin = self.margin()
        if margin is not None:
            meta["margin"] = round(margin, 4)
        if support.size:
            top = support[np.argsort(-weights[support])][:MAX_SUPPORT_VECTORS]
            meta["support_vectors"] = np.round(self.train_x[top], 4).tolist()
        return meta


MODELS: dict[str, type[MLModel]] = {
    LinearRegression.algo: LinearRegression,
    LogisticRegression.algo: LogisticRegression,
    DecisionTree.algo: DecisionTree,
    RandomForest.algo: RandomForest,
    SupportVectorMachine.algo: SupportVectorMachine,
}


def build(
    algo: str, train_x: np.ndarray, train_y: np.ndarray, params: dict[str, Any], seed: int
) -> MLModel:
    factory = MODELS.get(algo)
    if factory is None:
        raise ValueError(f"unknown ml algo: {algo}")
    return factory(train_x, train_y, params, seed)
