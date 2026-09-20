"""run 文件目录（契约见 docs/02 §7.1）：snapshots 与 checkpoints 的路径与清理。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app import config


def run_dir(run_id: str) -> Path:
    return config.RUNS_DIR / run_id


def snapshot_dir(run_id: str) -> Path:
    return run_dir(run_id) / "snapshots"


def ensure_run_dir(run_id: str) -> Path:
    snapshot_dir(run_id).mkdir(parents=True, exist_ok=True)
    (run_dir(run_id) / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir(run_id)


def snapshot_path(run_id: str, snapshot_id: str) -> Path:
    return snapshot_dir(run_id) / f"{snapshot_id}.json"


def checkpoint_path(run_id: str) -> Path:
    return run_dir(run_id) / "checkpoints" / "best.pt"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def remove_run_dir(run_id: str) -> None:
    shutil.rmtree(run_dir(run_id), ignore_errors=True)
