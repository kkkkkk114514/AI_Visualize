from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

DATA_DIR = Path(os.environ.get("AI_VISUALIZE_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = DATA_DIR / "runs"
DATASETS_DIR = DATA_DIR / "datasets"
DB_PATH = DATA_DIR / "app.db"
PRESETS_DIR = BACKEND_DIR / "app" / "graph" / "presets"
CORPORA_DIR = BACKEND_DIR / "app" / "datasets" / "corpora"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

HOST = os.environ.get("AI_VISUALIZE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AI_VISUALIZE_PORT", "8410"))

WS_PING_INTERVAL_S = 20.0

# run 生命周期（docs/02 §5.3 / §8.5）
DEFAULT_SEED = 42
RUN_STOP_GRACE_S = 3.0
RUN_KILL_GRACE_S = 1.0
WATCHDOG_POLL_S = 5.0
WATCHDOG_MIN_STALE_S = 30.0

# 训练子进程 CPU 线程数（docs/02 §5.2）：小批量下线程超订反而更慢，且会挤占渲染进程
TORCH_THREADS = int(
    os.environ.get("AI_VISUALIZE_TORCH_THREADS", "") or max(1, min(8, (os.cpu_count() or 2) // 2))
)

# 指标写库节流（docs/02 §2）
METRICS_FLUSH_INTERVAL_S = 2.0
METRICS_FLUSH_ROWS = 200
METRICS_MAX_POINTS = 2000

# 探针采样（docs/02 §6.1）
PROBE_DEFAULT_EVERY_N = 50
PROBE_MAX_PER_STREAM = 200
PROBE_MAX_ITEMS = 32

# 数据集镜像（按顺序回退；官方 ossci 镜像与 fgnt/mnist 内容同 md5）
DATASET_TIMEOUT_S = 30.0
_MIRROR_OVERRIDE = os.environ.get("AI_VISUALIZE_DATASET_MIRRORS", "")
DATASET_MIRRORS: tuple[str, ...] = tuple(
    mirror.strip() for mirror in _MIRROR_OVERRIDE.split(",") if mirror.strip()
) or (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
)

# 数据集上传（docs/02 §7.5）：单文件大小上限（MB）
UPLOAD_ZIP_MAX_MB = int(os.environ.get("AI_VISUALIZE_MAX_UPLOAD_MB", "") or 32)
UPLOAD_CSV_MAX_MB = 2
UPLOAD_TXT_MAX_MB = 5

SERVICE_NAME = "AI-Visualize"
SERVICE_VERSION = "0.1.0"


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, RUNS_DIR, DATASETS_DIR, DATASETS_DIR / "manual"):
        path.mkdir(parents=True, exist_ok=True)
