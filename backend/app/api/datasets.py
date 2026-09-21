from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.ws import hub
from app.datasets import downloaders, registry, synth2d

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["datasets"])

POINT_SPLITS = ("train", "val")


def _error(status_code: int, code: str, message_key: str, args: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message_key": message_key, "args": args or {}}},
    )


@router.get("/datasets")
async def list_datasets() -> dict[str, Any]:
    """数据集列表（含 md5 校验后的缓存状态与放置路径）。"""
    return {"datasets": await run_in_threadpool(registry.list_payload)}


@router.get("/datasets/{dataset_id}/points")
async def dataset_points(dataset_id: str, split: str = "train") -> Any:
    """二维合成数据集的点集（决策边界画布的散点底图，docs/02 §13.3）。"""
    spec = registry.get_spec(dataset_id)
    if spec is None:
        return _error(404, "dataset_not_found", "errors.dataset.notFound", {"id": dataset_id})
    if spec.loader != "synth2d":
        return _error(404, "dataset_no_points", "errors.dataset.noPoints", {"id": dataset_id})
    if split not in POINT_SPLITS:
        return _error(
            422, "dataset_bad_split", "errors.dataset.badSplit", {"split": split, "choices": list(POINT_SPLITS)}
        )
    return await run_in_threadpool(synth2d.points_payload, dataset_id, split)


@router.post("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str) -> Any:
    spec = registry.get_spec(dataset_id)
    if spec is None:
        return _error(404, "dataset_not_found", "errors.dataset.notFound", {"id": dataset_id})
    if await run_in_threadpool(registry.is_cached, spec):
        return {"dataset": registry.cache_state(spec), "started": False}
    if downloaders.is_downloading(dataset_id):
        return _error(409, "dataset_downloading", "errors.dataset.downloading", {"id": dataset_id})

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
        return _error(409, "dataset_downloading", "errors.dataset.downloading", {"id": dataset_id})
    log.info("数据集 %s 开始下载", dataset_id)
    return JSONResponse(status_code=202, content={"dataset_id": dataset_id, "started": True})
