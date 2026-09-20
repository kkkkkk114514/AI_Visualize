#!/usr/bin/env bash
# 开发模式：后端热重载（8410） + 前端 Vite dev（5180，/api 与 /ws 代理到后端）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -x backend/.venv/Scripts/python.exe ]; then
  PY=backend/.venv/Scripts/python.exe
elif [ -x backend/.venv/bin/python ]; then
  PY=backend/.venv/bin/python
else
  echo "未找到 backend/.venv：请先 python -m venv backend/.venv 并安装依赖" >&2
  exit 1
fi

"$PY" -m uvicorn app.main:app --app-dir backend --reload --port 8410 &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT

cd frontend
npm run dev
