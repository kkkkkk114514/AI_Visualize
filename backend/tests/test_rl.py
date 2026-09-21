"""M5 RL 测试：GridWorld 语义 / 表格 Q-learning 收敛与策略 / rl_runner 事件与 grid 快照。"""

from __future__ import annotations

import base64
import json
import queue
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from app import config
from app.algos import rl, spec as algo_spec
from app.probes import encode
from app.runners import base, rl_runner

ENV = {
    "width": 10,
    "height": 6,
    "start": [0, 0],
    "goal": [9, 5],
    "obstacles": [[3, 1], [3, 2], [6, 3]],
    "rewards": [[7, 4, 1.0]],
}
PARAMS = {
    "episodes": 250,
    "max_steps": 60,
    "alpha": 0.2,
    "gamma": 0.95,
    "epsilon_start": 0.5,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.98,
}
RL_METRICS = ("reward", "q_delta", "episode_reward", "epsilon", "episode_steps", "success")


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

    def statuses(self) -> list[dict]:
        return self.of(base.EVENT_STATUS)


def algo_spec_payload(env: dict | None = None, every_n: int | None = None, **params) -> dict:
    payload = {
        "ir_version": 1,
        "id": "inline-q_learning",
        "kind": "rl",
        "task": "control",
        "name": {"zh": "测试", "en": "test"},
        "algo": "q_learning",
        "env": ENV if env is None else env,
        "params": {"render_delay_ms": 0, **params},
        "dataset_id": "gridworld",
    }
    if every_n is not None:
        payload["probe_defaults"] = {"every_n_steps": every_n}
    return payload


def probe_payload(every_n: int = 500) -> list[dict]:
    return [
        {
            "node_id": "agent",
            "kind": "grid",
            "every_n_steps": every_n,
            "sample_index": 0,
            "max_items": 32,
        }
    ]


def run_rl(
    parameters: dict | None = None,
    *,
    tmp_path: Path,
    env: dict | None = None,
    probes: list[dict] | None = None,
    dataset_id: str = "gridworld",
    every_n: int | None = None,
    seed: int = 0,
    control: list[dict] | None = None,
) -> tuple[EventSink, int]:
    sink = EventSink()
    config = {
        "run_id": "r_test",
        "graph": algo_spec_payload(env, every_n, **({} if parameters is None else parameters)),
        "dataset_id": dataset_id,
        "seed": seed,
        "snapshots_dir": str(tmp_path),
        "probes": probe_payload() if probes is None else probes,
    }
    control_queue = queue.Queue()
    for command in control or []:
        control_queue.put(command)
    code = rl_runner.run_training(config, sink, control_queue)
    return sink, code


def fit(env: dict | None = None, parameters: dict | None = None, seed: int = 0) -> rl.QLearning:
    return train(env, parameters, seed)[0]


def train(
    env: dict | None = None, parameters: dict | None = None, seed: int = 0
) -> tuple[rl.QLearning, list[bool]]:
    """训练到 finished，顺带记录每个 episode 是否到达终点。"""
    full = dict(PARAMS)
    full.update(parameters or {})
    model = rl.QLearning(
        rl.GridWorld(algo_spec.parse_env(ENV if env is None else env), full["max_steps"]), full, seed
    )
    success: list[bool] = []
    seen = 0
    guard = 0
    while not model.finished:
        model.step()
        guard += 1
        assert guard < 100_000, "Q-learning 没有走到 finished"
        if model.episodes_done > seen:
            seen = model.episodes_done
            success.append(model.success)
    return model, success


def greedy_path(model: rl.QLearning, limit: int = 200) -> list[tuple[int, int]]:
    env = model.env
    env.reset()
    path = [env.position]
    for _ in range(limit):
        state = env.state_of(*env.position)
        _, _, done = env.step(model.greedy_action(state))
        path.append(env.position)
        if done:
            break
    return path


def display_path(model: rl.QLearning, limit: int = 200) -> list[tuple[int, int]]:
    """按画布箭头那张静态场（`meta.policy` 的来源）从起点走一遍，不看层。"""
    policy = model.policy()
    width, height = ENV["width"], ENV["height"]
    blocked = {tuple(cell) for cell in ENV["obstacles"]}
    goal = tuple(ENV["goal"])
    pos = tuple(ENV["start"])
    path = [pos]
    for _ in range(limit):
        action = int(policy[pos[1] * width + pos[0]])
        if action < 0:
            break
        dx, dy = rl.ACTION_DELTAS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or nxt in blocked:
            break
        pos = nxt
        path.append(pos)
        if pos == goal:
            break
    return path


def shortest_path(env: dict) -> int:
    start, goal = tuple(env["start"]), tuple(env["goal"])
    blocked = {tuple(cell) for cell in env["obstacles"]}
    seen, frontier = {start: 0}, deque([start])
    while frontier:
        cell = frontier.popleft()
        for dx, dy in rl.ACTION_DELTAS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if 0 <= nxt[0] < env["width"] and 0 <= nxt[1] < env["height"] and nxt not in blocked and nxt not in seen:
                seen[nxt] = seen[cell] + 1
                frontier.append(nxt)
    return seen[goal]


# ------------------------------------------------------------------ 环境语义


def test_collisions_consume_step_and_goal_terminates():
    env = rl.GridWorld(algo_spec.parse_env(ENV), 60)

    env.step(2)  # 从 (0,0) 向左：越界原地不动
    assert env.position == (0, 0) and env.steps == 1
    env.step(3)
    assert env.position == (1, 0)

    env.position = (2, 1)
    _, reward, done = env.step(3)  # (3,1) 是障碍
    assert env.position == (2, 1) and reward == 0.0 and not done

    env.position = (8, 5)
    _, reward, done = env.step(3)
    assert env.position == (9, 5) and reward == 1.0 and done

    limited = rl.GridWorld(algo_spec.parse_env(ENV), 3)
    rewards = [limited.step(0)[1] for _ in range(3)]
    assert rewards == [0.0, 0.0, 0.0]
    assert limited.step(0)[2] is True  # 第 4 步越界 → 已在第 3 步结束


def test_reward_cell_settles_once_per_episode():
    env = rl.GridWorld(algo_spec.parse_env(ENV), 60)
    env.position = (6, 4)
    assert env.step(3)[1] == 1.0  # 踏入 (7,4)
    assert env.step(2)[1] == 0.0  # 退回
    assert env.step(3)[1] == 0.0  # 再踏入：本 episode 已结算

    env.reset()
    env.position = (6, 4)
    assert env.step(3)[1] == 1.0

    goal_reward = dict(ENV, rewards=[[9, 5, 5.0]])  # 奖励格与终点重合：以终点结算
    goal_env = rl.GridWorld(algo_spec.parse_env(goal_reward), 60)
    goal_env.position = (9, 4)
    _, reward, done = goal_env.step(0)
    assert reward == 1.0 and done


def test_only_the_first_reward_pays_per_episode():
    two = dict(ENV, rewards=[[7, 4, 1.0], [1, 5, 3.0]])
    env = rl.GridWorld(algo_spec.parse_env(two), 60)
    env.position = (6, 4)
    assert env.step(3)[1] == 1.0  # 先踏到 (7,4)
    env.position = (1, 4)
    assert env.step(0)[1] == 0.0  # 本 episode 已结算过，(1,5) 不再发放
    env.reset()
    env.position = (1, 4)
    assert env.step(0)[1] == 3.0


def test_state_carries_the_reward_layer():
    env = rl.GridWorld(algo_spec.parse_env(ENV), 60)
    size = ENV["width"] * ENV["height"]
    fresh = env.state_of(0, 0)
    cell = 4 * ENV["width"] + 7  # (7,4) 在未领层的下标
    assert fresh == 0 and env.layer == 0

    env.position = (6, 4)
    state, reward, _ = env.step(3)
    assert reward == 1.0
    assert env.layer == 1
    assert state == cell + size  # 领奖那一步落到「已领」层
    assert env.cell_of(state) == (7, 4)  # 两层映回同一格
    assert env.cell_of(cell) == (7, 4)
    assert env.reset() == fresh and env.layer == 0


def test_q_table_covers_both_layers():
    model = rl.QLearning(rl.GridWorld(algo_spec.parse_env(ENV), 60), PARAMS, 0)
    size = ENV["width"] * ENV["height"]
    assert model.q.shape == (rl.LAYERS * size, rl.ACTION_COUNT)
    assert np.array_equal(model.legal[:size], model.legal[size:])  # 两层共用一张合法掩码
    assert model.values().shape == (size,)
    assert len(model.policy()) == size


def test_policy_arrows_stay_legal():
    model, _ = train(seed=0)
    policy = model.policy()
    assert len(policy) == ENV["width"] * ENV["height"]
    obstacles = {tuple(cell) for cell in ENV["obstacles"]}
    values = model.values()
    for state, action in enumerate(policy):
        cell = model.env.cell_of(state)
        if action == -1:
            # 不画箭头只有三种情形：障碍 / 终点 / 两层都没学到值
            assert cell in obstacles or cell == tuple(ENV["goal"]) or values[state] <= 0
            continue
        assert 0 <= action < 4
        dx, dy = rl.ACTION_DELTAS[int(action)]
        target = (cell[0] + dx, cell[1] + dy)
        assert 0 <= target[0] < ENV["width"] and 0 <= target[1] < ENV["height"]
        assert target not in obstacles


# ------------------------------------------------------------------ 收敛与策略


def test_q_learning_converges_to_goal():
    best = shortest_path(ENV)
    # 42 是 `config.DEFAULT_SEED`：前端不传 seed，预置 run 一直用它
    for seed in (0, 1, 2, config.DEFAULT_SEED):
        model, success = train(seed=seed)
        path = greedy_path(model)
        assert path[-1] == tuple(ENV["goal"]), f"seed={seed} 贪心策略没到终点"
        assert len(path) - 1 <= best + 4, f"seed={seed} 贪心轨迹绕远：{len(path) - 1} vs {best}"
        assert model.epsilon == pytest.approx(PARAMS["epsilon_end"])
        assert np.mean(success[-20:]) >= 0.8, f"seed={seed} 后 20 个 episode 成功率过低"


def test_display_field_arrows_route_to_goal():
    """画布上那排箭头（`meta.policy` 那张静态场）从起点直达终点——M5 验收 ②。"""
    best = shortest_path(ENV)
    for seed in (0, 1, 2, config.DEFAULT_SEED):
        model, _ = train(seed=seed)
        path = display_path(model)
        assert path[-1] == tuple(ENV["goal"]), f"seed={seed} 箭头场没到终点：{path}"
        assert len(path) - 1 <= best + 4, f"seed={seed} 箭头场绕远：{len(path) - 1} vs {best}"


def test_display_prefers_the_collected_layer():
    """画布优先画「已领层」（奖励领过之后直奔终点），已领层没值的格子回退未领层（奔向奖励格）。"""
    model, _ = train(seed=config.DEFAULT_SEED)
    size = model.env.layer_size
    assert model.env.layer == 0  # 训练结束刚 reset
    cell = 4 * ENV["width"] + 7  # (7,4) 奖励格：领过之后就在已领层
    assert model.q[cell + size].max() > 0
    assert model.values()[cell] == pytest.approx(model.q[cell + size].max())
    # 起点一侧只有未领层有数据（已领层没走到过），显示值取未领层
    assert model.q[size].max() == 0
    assert model.q[0].max() > 0
    assert model.values()[0] == pytest.approx(model.q[0].max())
    # 两层都没值的格子不画箭头（-1），比画一个退化的「第一个合法动作」诚实
    policy = model.policy()
    blank = [cell for cell in range(size) if model.values()[cell] <= 0]
    assert blank, "预置迷宫不该每个格子都学到值"
    for cell in blank:
        x, y = model.env.cell_of(cell)
        if model.env.is_obstacle(x, y) or (x, y) == model.env.goal:
            continue
        assert policy[cell] == -1


def test_epsilon_decays_monotonically():
    model = rl.QLearning(rl.GridWorld(algo_spec.parse_env(ENV), 60), PARAMS, 0)
    values = [model.epsilon]
    seen = 0
    while not model.finished:
        model.step()
        if model.episodes_done > seen:
            seen = model.episodes_done
            values.append(model.epsilon)
    assert len(values) == PARAMS["episodes"] + 1
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[1] == pytest.approx(PARAMS["epsilon_start"] * PARAMS["epsilon_decay"])
    assert values[-1] == pytest.approx(PARAMS["epsilon_end"])


def test_seed_determinism():
    first, second = fit(seed=3), fit(seed=3)
    assert np.array_equal(first.q, second.q)
    assert np.array_equal(first.policy(), second.policy())
    other = fit(seed=4)
    assert not np.array_equal(first.q, other.q)


def test_obstacles_change_the_learned_policy():
    blocked_env = dict(ENV, obstacles=ENV["obstacles"] + [[7, 5], [8, 5]])
    model = fit(env=blocked_env)
    path = greedy_path(model)
    assert path[-1] == tuple(ENV["goal"])
    assert (7, 5) not in path and (8, 5) not in path
    assert len(path) - 1 == shortest_path(blocked_env)
    assert not np.array_equal(model.policy(), fit().policy())


def test_set_lr_overrides_alpha():
    model = rl.QLearning(rl.GridWorld(algo_spec.parse_env(ENV), 60), PARAMS, 0)
    model.set_lr(0.5)
    assert model.alpha == pytest.approx(0.5)
    model.set_lr(3.0)
    assert model.alpha == pytest.approx(1.0)


# ------------------------------------------------------------------ rl_runner


def test_rl_runner_emits_metrics_and_grid_snapshots(tmp_path):
    sink, code = run_rl(every_n=500, tmp_path=tmp_path)
    assert code == 0
    assert sink.of(base.EVENT_ERROR) == []
    log_keys = [event["key"] for event in sink.of(base.EVENT_LOG)]
    assert log_keys[0] == "log.run.starting"
    assert "log.run.env" in log_keys and "log.run.finished" in log_keys
    assert "log.probe.attached" in log_keys

    statuses = sink.statuses()
    assert statuses[0]["status"] == base.STATUS_RUNNING
    assert statuses[0]["device"] == "cpu"
    assert statuses[0]["planned_steps"] == PARAMS["episodes"] * PARAMS["max_steps"]
    assert statuses[-1]["status"] == base.STATUS_FINISHED
    assert statuses[-1]["epoch"] == PARAMS["episodes"]
    assert set(event["status"] for event in statuses) == {base.STATUS_RUNNING, base.STATUS_FINISHED}

    points = sink.metrics()
    names = {name for point in points for name in point["values"]}
    assert names <= set(RL_METRICS)
    assert {"reward", "q_delta"} <= names
    episode_points = [point for point in points if "episode_reward" in point["values"]]
    assert len(episode_points) == PARAMS["episodes"]
    assert {"epsilon", "episode_steps", "success"} <= set(episode_points[-1]["values"])

    rewards = [point["values"]["episode_reward"] for point in episode_points]
    rolling = [float(np.mean(rewards[max(0, index - 19) : index + 1])) for index in range(len(rewards))]
    assert statuses[-1]["best_metric"] == pytest.approx(max(rolling))

    probe_events = sink.of(base.EVENT_PROBE)
    assert len(probe_events) == statuses[-1]["step"] // 500
    last = probe_events[-1]
    assert last["node_id"] == "agent" and last["kind"] == "grid"
    assert last["shape"] == [ENV["height"], ENV["width"]]
    record = json.loads(Path(last["file_path"]).read_text(encoding="utf-8"))
    assert record["dtype"] == "uint8" and record["layout"] == "HW"
    assert record["shape"] == [ENV["height"], ENV["width"]]
    assert len(record["meta"]["policy"]) == ENV["width"] * ENV["height"]
    assert record["meta"]["epsilon"] <= PARAMS["epsilon_start"]
    trajectory = record["meta"]["trajectory"]
    assert trajectory[0] == ENV["start"]
    assert len(trajectory) <= PARAMS["max_steps"] + 1
    grid = np.frombuffer(base64.b64decode(record["data_b64"]), dtype=np.uint8)
    values = grid.reshape(ENV["height"], ENV["width"])
    for x, y in ENV["obstacles"]:
        assert values[y, x] == 0


def test_rl_runner_without_probes_writes_nothing(tmp_path):
    sink, code = run_rl({"episodes": 20, "render_delay_ms": 0}, tmp_path=tmp_path, probes=[])
    assert code == 0
    assert sink.of(base.EVENT_PROBE) == []
    assert "log.probe.attached" not in [event["key"] for event in sink.of(base.EVENT_LOG)]
    assert list(tmp_path.glob("*.json")) == []


def test_rl_runner_requires_gridworld_dataset(tmp_path):
    sink, code = run_rl({"episodes": 5}, tmp_path=tmp_path, dataset_id="mnist")
    assert code == 1
    errors = sink.of(base.EVENT_ERROR)
    assert errors and errors[0]["code"] == "dataset_kind_mismatch"
    assert errors[0]["message_key"] == "errors.dataset.kindMismatch"
    assert errors[0]["args"]["expected"] == "gridworld"


def test_rl_runner_rejects_bad_env(tmp_path):
    bad_env = dict(ENV, goal=[30, 5])
    sink, code = run_rl({"episodes": 5}, tmp_path=tmp_path, env=bad_env)
    assert code == 1
    assert sink.of(base.EVENT_ERROR)[0]["message_key"] == "errors.algo.badEnv"


def test_resolve_probes_rl_single_stream():
    spec = algo_spec.parse_spec(algo_spec_payload())
    default = algo_spec.resolve_probes(None, spec)
    assert len(default) == 1
    assert default[0]["node_id"] == "agent" and default[0]["kind"] == "grid"
    assert default[0]["every_n_steps"] == 80

    with pytest.raises(algo_spec.AlgoSpecError) as excinfo:
        algo_spec.resolve_probes([{"node_id": "model", "kind": "boundary"}], spec)
    assert excinfo.value.code == "invalid_probe"
    assert excinfo.value.message_key == "errors.probe.badStream"


def test_rl_runner_stop_and_ignored_batch_size(tmp_path):
    sink, code = run_rl(
        {"episodes": 250},
        tmp_path=tmp_path,
        control=[{"action": base.CTRL_SET_BATCH_SIZE, "value": 64}, {"action": base.CTRL_SET_LR, "value": 0.4}],
    )
    assert code == 0
    assert "log.run.batchIgnored" in [event["key"] for event in sink.of(base.EVENT_LOG)]
    assert sink.statuses()[-1]["status"] == base.STATUS_FINISHED  # 单流 RL 跑完 250 episode 才停

    sink, code = run_rl({"episodes": 250}, tmp_path=tmp_path, control=[{"action": base.CTRL_STOP}])
    assert code == 0
    assert sink.statuses()[-1]["status"] == base.STATUS_STOPPED
    assert sink.statuses()[-1]["step"] < PARAMS["episodes"] * PARAMS["max_steps"]


def test_grid_payload_round_trip(tmp_path):
    model = fit(parameters={"episodes": 1})
    silent = rl_runner.GridProbe(
        probe_payload(500), model, run_id="r_test", snapshots_dir=str(tmp_path), emit=lambda event: None
    )
    silent.maybe_capture(1, 1)  # every_n=500：本步不采样
    assert list(tmp_path.glob("*.json")) == []

    probe = rl_runner.GridProbe(
        probe_payload(1),
        model,
        run_id="r_test",
        snapshots_dir=str(tmp_path),
        emit=lambda event: None,
    )
    probe.maybe_capture(1, 1)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    payload = {key: record[key] for key in ("shape", "dtype", "layout", "min", "max", "data_b64", "meta")}
    assert payload["layout"] == "HW" and payload["dtype"] == "uint8"
    assert payload["shape"] == [ENV["height"], ENV["width"]]
    assert np.isfinite(payload["min"]) and np.isfinite(payload["max"])

    with pytest.raises(encode.ProbeShapeError):
        encode.encode_grid(np.zeros((3, 4), dtype=np.float32), policy=[0] * 5)
