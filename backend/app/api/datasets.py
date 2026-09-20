from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.ws import hub
from app.datasets import downloaders, registry

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets")
async def list_datasets() -> dict[str, Any]:
    """数据集列表（含 md5 校验后的缓存状态与放置路径）。"""
    return {"datasets": await run_in_threadpool(registry.list_payload)}


@router.post("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str) -> Any:
    spec = registry.get_spec(dataset_id)
    if spec is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "dataset_not_found",
                    "message_key": "errors.dataset.notFound",
                    "args": {"id": dataset_id},
                }
            },
        )
    if await run_in_threadpool(registry.is_cached, spec):
        return {"dataset": registry.cache_state(spec), "started": False}
    if downloaders.is_downloading(dataset_id):
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "dataset_downloading",
                    "message_key": "errors.dataset.downloading",
                    "args": {"id": dataset_id},
                }
            },
        )

    loop = asyncio.get_running_loop()

    def on_progress(phase: str, progress: float, extra: dict[str, Any]) -> None:
        event: dict[str, Any] = {
            "type": "dataset.progress",
            "dataset_id": dataset_id,
            "phase": phase,
            "progress": round(progress, 4),
        }
        event.update(extra)
        hub.emit_threadsafe(event)

    started = downloaders.start_download(dataset_id, on_progress)
    if not started:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "dataset_downloading",
                    "message_key": "errors.dataset.downloading",
                    "args": {"id": dataset_id},
                }
            },
        )
    log.info("数据集 %s 开始下载", dataset_id)
    return JSONResponse(status_code=202, content={"dataset_id": dataset_id, "started": True})
