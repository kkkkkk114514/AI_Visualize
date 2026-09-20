from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

DATA_DIR = Path(os.environ.get("AI_VISUALIZE_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = DATA_DIR / "runs"
DATASETS_DIR = DATA_DIR / "datasets"
PRESETS_DIR = BACKEND_DIR / "app" / "graph" / "presets"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

HOST = os.environ.get("AI_VISUALIZE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AI_VISUALIZE_PORT", "8410"))

WS_PING_INTERVAL_S = 20.0

SERVICE_NAME = "AI-Visualize"
SERVICE_VERSION = "0.1.0"


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, RUNS_DIR, DATASETS_DIR):
        path.mkdir(parents=True, exist_ok=True)
