#!/usr/bin/env bash
# 生产模式：构建前端后由后端单端口托管（http://localhost:8410）
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

npm --prefix frontend run build
exec "$PY" -m uvicorn app.main:app --app-dir backend --port 8410
