"""SQLite 存储层（契约见 docs/02 §7.2）：schema、run/metrics/snapshot 读写与批量写。"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
from datetime import datetime
from typing import Any, Iterable, Sequence

from app import config

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
  graph_json TEXT, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
  model_id TEXT, graph_json TEXT,
  dataset_id TEXT, hyperparams_json TEXT, probes_json TEXT, status TEXT NOT NULL,
  device TEXT, seed INTEGER, created_at TEXT, started_at TEXT, finished_at TEXT,
  error TEXT, best_metric REAL, total_steps INTEGER
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL, step INTEGER NOT NULL, epoch INTEGER,
  name TEXT NOT NULL, value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name_step ON metrics(run_id, name, step);
CREATE TABLE IF NOT EXISTS snapshots (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step INTEGER NOT NULL, epoch INTEGER,
  node_id TEXT, kind TEXT NOT NULL, shape_json TEXT,
  min REAL, max REAL, file_path TEXT NOT NULL, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_run_step ON snapshots(run_id, step);
"""

RUN_FIELDS: tuple[str, ...] = (
    "name",
    "kind",
    "model_id",
    "graph_json",
    "dataset_id",
    "hyperparams_json",
    "probes_json",
    "status",
    "device",
    "seed",
    "created_at",
    "started_at",
    "finished_at",
    "error",
    "best_metric",
    "total_steps",
)

RUN_SUMMARY_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "kind",
    "model_id",
    "dataset_id",
    "hyperparams_json",
    "probes_json",
    "status",
    "device",
    "seed",
    "created_at",
    "started_at",
    "finished_at",
    "error",
    "best_metric",
    "total_steps",
)

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id() -> str:
    return f"r_{secrets.token_hex(3)}"


def new_snapshot_id() -> str:
    return f"s_{secrets.token_hex(4)}"


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
        return _conn


def init_db() -> None:
    with _lock:
        conn = connect()
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    log.info("SQLite 就绪：%s", config.DB_PATH)


def _migrate(conn: sqlite3.Connection) -> None:
    """启动期轻量迁移（docs/02 §7.2）：缺列则 ALTER TABLE 补上，老库自动升级。"""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    if "probes_json" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN probes_json TEXT")
        log.info("迁移：runs.probes_json 已补齐")


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    if "graph_json" in data and data["graph_json"]:
        try:
            data["graph"] = json.loads(data["graph_json"])
        except json.JSONDecodeError:
            data["graph"] = None
    if data.get("hyperparams_json"):
        try:
            data["hyperparams"] = json.loads(data["hyperparams_json"])
        except json.JSONDecodeError:
            data["hyperparams"] = {}
    else:
        data["hyperparams"] = {}
    if data.get("probes_json"):
        try:
            probes = json.loads(data["probes_json"])
            data["probes"] = probes if isinstance(probes, list) else []
        except json.JSONDecodeError:
            data["probes"] = []
    else:
        data["probes"] = []
    return data


# ------------------------------------------------------------------ runs


def insert_run(record: dict[str, Any]) -> dict[str, Any]:
    payload = {field: record.get(field) for field in RUN_FIELDS}
    payload["id"] = record["id"]
    columns = ", ".join(payload)
    placeholders = ", ".join("?" for _ in payload)
    with _lock:
        conn = connect()
        conn.execute(
            f"INSERT INTO runs ({columns}) VALUES ({placeholders})", list(payload.values())
        )
        conn.commit()
    return payload


def update_run(run_id: str, **fields: Any) -> None:
    updates = {key: value for key, value in fields.items() if key in RUN_FIELDS}
    if not updates:
        return
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _lock:
        conn = connect()
        conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", [*updates.values(), run_id])
        conn.commit()


def get_run(run_id: str) -> dict[str, Any] | None:
    with _lock:
        row = connect().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _decode_run(row) if row is not None else None


def list_runs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    columns = ", ".join(RUN_SUMMARY_FIELDS)
    with _lock:
        rows = connect().execute(
            f"SELECT {columns} FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_decode_run(row) for row in rows]


def count_runs() -> int:
    with _lock:
        row = connect().execute("SELECT COUNT(*) AS n FROM runs").fetchone()
    return int(row["n"])


def all_run_ids() -> list[str]:
    """全部 run id（占用统计与「清理全部历史」用，不受分页限制）。"""
    with _lock:
        rows = connect().execute("SELECT id FROM runs ORDER BY created_at DESC, rowid DESC").fetchall()
    return [row["id"] for row in rows]


def delete_run(run_id: str) -> bool:
    with _lock:
        conn = connect()
        cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM snapshots WHERE run_id = ?", (run_id,))
        conn.commit()
    return cursor.rowcount > 0


def mark_orphans_interrupted() -> list[str]:
    """服务启动时把上次残留的活动 run 标记为 interrupted（docs/02 §5.3）。"""
    with _lock:
        conn = connect()
        rows = conn.execute(
            "SELECT id FROM runs WHERE status IN ('created', 'running', 'paused')"
        ).fetchall()
        if rows:
            conn.execute(
                "UPDATE runs SET status = 'interrupted', finished_at = ? "
                "WHERE status IN ('created', 'running', 'paused')",
                (now_iso(),),
            )
            conn.commit()
    return [row["id"] for row in rows]


# ------------------------------------------------------------------ metrics


def insert_metrics(run_id: str, rows: Iterable[tuple[int, int | None, str, float]]) -> int:
    payload = [(run_id, step, epoch, name, value) for step, epoch, name, value in rows]
    if not payload:
        return 0
    with _lock:
        conn = connect()
        conn.executemany(
            "INSERT INTO metrics (run_id, step, epoch, name, value) VALUES (?, ?, ?, ?, ?)",
            payload,
        )
        conn.commit()
    return len(payload)


def fetch_metric_series(
    run_id: str, names: Sequence[str] | None = None
) -> dict[str, dict[str, list[float]]]:
    sql = "SELECT name, step, value FROM metrics WHERE run_id = ?"
    params: list[Any] = [run_id]
    if names:
        sql += f" AND name IN ({', '.join('?' for _ in names)})"
        params.extend(names)
    sql += " ORDER BY name, step"
    with _lock:
        rows = connect().execute(sql, params).fetchall()
    series: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        entry = series.setdefault(row["name"], {"steps": [], "values": []})
        entry["steps"].append(int(row["step"]))
        entry["values"].append(float(row["value"]))
    return series


def count_metrics(run_id: str) -> int:
    with _lock:
        row = connect().execute(
            "SELECT COUNT(*) AS n FROM metrics WHERE run_id = ?", (run_id,)
        ).fetchone()
    return int(row["n"])


# ---------------------------------------------------------------- snapshots


def insert_snapshot(record: dict[str, Any]) -> None:
    with _lock:
        conn = connect()
        conn.execute(
            "INSERT OR REPLACE INTO snapshots "
            "(id, run_id, step, epoch, node_id, kind, shape_json, min, max, file_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["id"],
                record["run_id"],
                record["step"],
                record.get("epoch"),
                record.get("node_id"),
                record["kind"],
                json.dumps(record.get("shape") or [], ensure_ascii=False),
                record.get("min"),
                record.get("max"),
                record["file_path"],
                record.get("created_at") or now_iso(),
            ),
        )
        conn.commit()


def list_snapshots(
    run_id: str, step: int | None = None, node_id: str | None = None, kind: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM snapshots WHERE run_id = ?"
    params: list[Any] = [run_id]
    if step is not None:
        sql += " AND step = ?"
        params.append(step)
    if node_id:
        sql += " AND node_id = ?"
        params.append(node_id)
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY step, id"
    with _lock:
        rows = connect().execute(sql, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["shape"] = json.loads(item.pop("shape_json") or "[]")
        except json.JSONDecodeError:
            item["shape"] = []
        out.append(item)
    return out


def get_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    with _lock:
        row = connect().execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    try:
        item["shape"] = json.loads(item.pop("shape_json") or "[]")
    except json.JSONDecodeError:
        item["shape"] = []
    return item
