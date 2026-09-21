from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.api import algos, datasets, graph, health, models, runs, ws
from app.api.ws import hub
from app.runners.manager import manager
from app.store import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_data_dirs()
    db.init_db()
    loop = asyncio.get_running_loop()
    hub.bind_loop(loop)
    manager.bind_loop(loop)
    orphans = db.mark_orphans_interrupted()
    if orphans:
        log.warning("上次残留的活动 run 已标记为 interrupted：%s", ", ".join(orphans))
    log.info("数据目录就绪：%s", config.DATA_DIR)
    try:
        yield
    finally:
        await manager.shutdown()
        db.close()


app = FastAPI(title=config.SERVICE_NAME, version=config.SERVICE_VERSION, lifespan=lifespan)
app.include_router(health.router)
app.include_router(algos.router)
app.include_router(models.router)
app.include_router(graph.router)
app.include_router(datasets.router)
app.include_router(runs.router)
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
