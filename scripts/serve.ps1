# 生产模式：构建前端后由后端单端口托管（http://localhost:8410）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "未找到 backend\.venv：请先 python -m venv backend\.venv 并安装依赖"
}

npm --prefix frontend run build
& $python -m uvicorn app.main:app --app-dir backend --port 8410
