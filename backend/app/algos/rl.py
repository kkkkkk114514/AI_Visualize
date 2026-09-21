"""GridWorld 环境与表格 Q-learning（契约见 docs/02 §13.2 / §13.3 / §13.5）：纯 numpy，不加载 torch。

坐标约定：格子 `(x, y)` 的 `y` 向上增大；状态下标 = `layer * W * H + y * width + x`，
`layer` 记录「本 episode 是否已领过奖励格」（§13.3），因此 `V` 的二维视图行号即 y
（行 0 在下，与 `boundary` 同约定）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.algos.spec import EnvSpec

ACTION_COUNT = 4
# 动作下标（docs/02 §13.5）：0=上 / 1=下 / 2=左 / 3=右
ACTION_DELTAS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (-1, 0), (1, 0))
LAYERS = 2  # 状态层数：0 = 未领奖励格 / 1 = 已领（§13.3）
GOAL_REWARD = 1.0
ROLLING_WINDOW = 20


class GridWorld:
    """确定性环境下的一步交互：撞墙 / 障碍原地不动，踏入终点获得 +1 并结束。

    奖励每 episode 只结算一次（docs/02 §13.3）：先踏到的那个奖励格发放 `rewards` 里的
    值，其余奖励格与之后再次踏入都按普通格算——否则绕圈刷奖励格的折扣回报会高于一次性
    到达终点。领过没有不是隐藏变量，它是状态的一层（`layer`）：同一格在「未领」与
    「已领」两张页上各有自己的 Q，否则同一个 `(s,a)` 的目标值会在含 +1 与不含之间跳。
    """

    def __init__(self, env: EnvSpec, max_steps: int) -> None:
        self.width = env.width
        self.height = env.height
        self.start = env.start
        self.goal = env.goal
        self.obstacles = frozenset(env.obstacles)
        self.bonus = {(x, y): value for x, y, value in env.rewards}
        self.max_steps = int(max_steps)
        self.layer_size = self.width * self.height
        self.position = self.start
        self.steps = 0
        self.visited: list[tuple[int, int]] = [self.start]
        self.collected: set[tuple[int, int]] = set()

    def reset(self) -> int:
        self.position = self.start
        self.steps = 0
        self.visited = [self.start]
        self.collected.clear()
        return self.state_of(*self.start)

    @property
    def layer(self) -> int:
        """1 = 本 episode 已领过奖励格（§13.3），进状态下标的高位。"""
        return 1 if self.collected else 0

    def state_of(self, x: int, y: int) -> int:
        return self.layer * self.layer_size + y * self.width + x

    def cell_of(self, state: int) -> tuple[int, int]:
        index = int(state) % self.layer_size
        return index % self.width, index // self.width

    def is_obstacle(self, x: int, y: int) -> bool:
        return (x, y) in self.obstacles

    def step(self, action: int) -> tuple[int, float, bool]:
        self.steps += 1
        dx, dy = ACTION_DELTAS[int(action)]
        x, y = self.position
        target = (x + dx, y + dy)
        if not (0 <= target[0] < self.width and 0 <= target[1] < self.height):
            target = self.position
        elif self.is_obstacle(*target):
            target = self.position
        self.position = target
        self.visited.append(target)
        if target == self.goal:
            reward = GOAL_REWARD
        elif target in self.bonus and not self.collected:
            reward = self.bonus[target]
            self.collected.add(target)
        else:
            reward = 0.0
        done = target == self.goal or self.steps >= self.max_steps
        # 先结算再取状态下标：领奖励的这一步落到「已领」页，+1 只挂在这一次转移上
        return self.state_of(*target), reward, done


class QLearning:
    """表格 Q-learning：`Q(s,a) ← Q + α[r + γ·max Q(s′,a′) − Q]`，ε-greedy 行为策略。"""

    algo = "q_learning"

    def __init__(self, env: GridWorld, params: dict[str, Any], seed: int) -> None:
        self.env = env
        self.alpha = float(params["alpha"])
        self.gamma = float(params["gamma"])
        self.episodes = int(params["episodes"])
        self.epsilon_end = float(params["epsilon_end"])
        self.epsilon_decay = float(params["epsilon_decay"])
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((LAYERS * env.layer_size, ACTION_COUNT), dtype=np.float64)
        # 合法动作掩码：状态已知边界与障碍，撞墙动作不参与选择（见 `_act`）；两层同一张掩码
        base_legal = np.zeros((env.layer_size, ACTION_COUNT), dtype=bool)
        for state in range(env.layer_size):
            x, y = env.cell_of(state)
            for action, (dx, dy) in enumerate(ACTION_DELTAS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height and not env.is_obstacle(nx, ny):
                    base_legal[state, action] = True
        self.legal = np.tile(base_legal, (LAYERS, 1))
        self.epsilon = float(params["epsilon_start"])
        self.step_index = 0
        self.episodes_done = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.success = False
        self.state = env.reset()
        self.recent_rewards: list[float] = []
        self.last_trajectory: list[list[int]] = []

    @property
    def finished(self) -> bool:
        return self.episodes_done >= self.episodes

    def set_lr(self, value: float) -> None:
        """`set_lr` 控制命令：RL 下改的是学习率 α（下一 step 生效）。"""
        self.alpha = max(1e-6, min(float(value), 1.0))

    def _display_q(self) -> np.ndarray:
        """画布用的每格 Q 表 `[W*H, 4]`：优先**已领层**，它没值的格子回退未领层（§13.5）。

        两层各只覆盖路线的一半：未领层只被「去领奖励格的那段」训练到，是「奔向奖励格」的
        流场；已领层只被「领过之后的那段」训练到，是「从奖励格直奔终点」的流场。按格取已领层
        有值的那些：奖励格附近必须用已领层，否则未领层的箭头会（正确地）拐回去再领一次，
        静态场看起来就是在奖励格旁边绕圈；其余格子（含两层都没值、一次都没走到过的格）
        交给未领层——起点一侧只有它有数据。
        """
        size = self.env.layer_size
        collected = self.q[size:]
        return np.where((collected.max(axis=1) > 0.0)[:, None], collected, self.q[:size])

    def values(self) -> np.ndarray:
        """`V(s) = max_a Q(s,a)`，按格子下标展开（取 `_display_q`，§13.5）。"""
        return self._display_q().max(axis=1)

    def greedy_action(self, state: int) -> int:
        """指定状态下合法动作里的最大 Q；非法动作视为 -inf（同值取最小下标）。"""
        masked = np.where(self.legal[state], self.q[state], -np.inf)
        return int(np.argmax(masked))

    def policy(self) -> np.ndarray:
        """画布箭头用的贪心动作下标（`_display_q`）；障碍、终点与两层都没值的格子为 -1。"""
        size = self.env.layer_size
        display = self._display_q()
        masked = np.where(self.legal[:size], display, -np.inf)
        actions = masked.argmax(axis=1).astype(np.int64)
        for state in range(len(actions)):
            x, y = self.env.cell_of(state)
            if (
                self.env.is_obstacle(x, y)
                or (x, y) == self.env.goal
                or display[state].max() <= 0.0
            ):
                actions[state] = -1
        return actions

    def _act(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(self.legal[self.state])))
        masked = np.where(self.legal[self.state], self.q[self.state], -np.inf)
        # 同值动作随机挑：Q 全零的区域里恒取最小下标会退化成上下震荡，探索不到新格
        return int(self.rng.choice(np.flatnonzero(masked == masked.max())))

    def step(self) -> dict[str, float]:
        action = self._act()
        next_state, reward, done = self.env.step(action)
        target = reward + (0.0 if done else self.gamma * float(self.q[next_state].max()))
        delta = target - float(self.q[self.state, action])
        self.q[self.state, action] += self.alpha * delta
        self.state = next_state
        self.step_index += 1
        self.episode_steps += 1
        self.episode_reward += reward
        values = {"reward": float(reward), "q_delta": abs(float(delta))}
        if done:
            values.update(self._end_episode())
        return values

    def _end_episode(self) -> dict[str, float]:
        self.success = self.env.position == self.env.goal
        self.last_trajectory = [
            [int(x), int(y)] for x, y in self.env.visited[: self.env.max_steps]
        ]
        self.episodes_done += 1
        self.recent_rewards.append(self.episode_reward)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        completed = {
            "episode_reward": self.episode_reward,
            "epsilon": self.epsilon,
            "episode_steps": float(self.episode_steps),
            "success": 1.0 if self.success else 0.0,
        }
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.state = self.env.reset()
        return completed

    def rolling_reward(self) -> float:
        """最近 `ROLLING_WINDOW` 个 episode 的平均回报（best_metric 的口径，docs/02 §13.4）。"""
        window = self.recent_rewards[-ROLLING_WINDOW:]
        return float(np.mean(window)) if window else 0.0

    def grid_payload_meta(self) -> dict[str, Any]:
        """画布角标与轨迹：`episode` / `success` / `trajectory` 都描述最近一个**已完成** episode。"""
        return {
            "episode": self.episodes_done,
            "epsilon": round(self.epsilon, 4),
            "success": bool(self.success),
            "trajectory": self.last_trajectory,
        }
