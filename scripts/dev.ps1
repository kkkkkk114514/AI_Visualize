# 开发模式：后端热重载（8410） + 前端 Vite dev（5180，/api 与 /ws 代理到后端）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "未找到 backend\.venv：请先 python -m venv backend\.venv 并安装依赖"
}

$backend = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--reload", "--port", "8410" `
    -PassThru
try {
    Set-Location (Join-Path $root "frontend")
    npm run dev
}
finally {
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
}
