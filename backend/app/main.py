from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.api import graph, health, models, ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_data_dirs()
    log.info("数据目录就绪：%s", config.DATA_DIR)
    yield


app = FastAPI(title=config.SERVICE_NAME, version=config.SERVICE_VERSION, lifespan=lifespan)
app.include_router(health.router)
app.include_router(models.router)
app.include_router(graph.router)
app.include_router(ws.router)

if config.FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    if not config.FRONTEND_DIST.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "frontend_missing",
                    "message_key": "errors.frontend_missing",
                    "detail": "前端未构建：请先在 frontend/ 执行 npm run build",
                }
            },
        )
    candidate = config.FRONTEND_DIST / full_path
    if full_path and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(config.FRONTEND_DIST / "index.html")
