from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.ws import hub
from app.datasets import downloaders, points2d, registry, upload

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["datasets"])

POINT_SPLITS = ("train", "val")
READ_CHUNK = 1 << 20


def _error(
    status_code: int,
    code: str,
    message_key: str,
    args: dict[str, Any] | None = None,
    detail: str = "",
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message_key": message_key, "args": args or {}}
    if detail:
        error["detail"] = detail
    return JSONResponse(status_code=status_code, content={"error": error})


@router.get("/datasets")
async def list_datasets() -> dict[str, Any]:
    """数据集列表（含 md5 校验后的缓存状态与放置路径）。"""
    return {"datasets": await run_in_threadpool(registry.list_payload)}


@router.get("/datasets/{dataset_id}/points")
async def dataset_points(dataset_id: str, split: str = "train") -> Any:
    """二维点集（决策边界画布的散点底图，docs/02 §13.3）：`synth2d` 与上传的 `csv2d`。"""
    spec = registry.get_spec(dataset_id)
    if spec is None:
        return _error(404, "dataset_not_found", "errors.dataset.notFound", {"id": dataset_id})
    if spec.loader not in points2d.POINT_LOADERS:
        return _error(404, "dataset_no_points", "errors.dataset.noPoints", {"id": dataset_id})
    if split not in POINT_SPLITS:
        return _error(
            422, "dataset_bad_split", "errors.dataset.badSplit", {"split": split, "choices": list(POINT_SPLITS)}
        )
    try:
        return await run_in_threadpool(points2d.points_payload, spec, split)
    except (OSError, ValueError) as exc:
        return _error(
            409, "dataset_not_cached", "errors.dataset.notCached", {"id": dataset_id}, str(exc)
        )


@router.post("/datasets/upload")
async def upload_dataset(file: Annotated[UploadFile | None, File()] = None) -> Any:
    """上传自定义数据集（multipart 字段 `file`，docs/02 §7.5）：zip 图片集 / csv 点集 / txt 语料。"""
    if file is None:
        return _error(400, "upload_no_file", "errors.upload.noFile")
    filename = file.filename or ""
    try:
        fmt = upload.resolve_format(filename)
    except upload.UploadError as exc:
        return _error(exc.status_code, exc.code, exc.message_key, exc.error_args, exc.detail)

    chunks: list[bytes] = []
    total = 0
    limit_bytes = upload.limit_mb(fmt) << 20
    while True:
        chunk = await file.read(READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            return _error(
                413,
                "upload_too_large",
                "errors.upload.tooLarge",
                {"format": fmt, "limit_mb": upload.limit_mb(fmt)},
            )
        chunks.append(chunk)

    try:
        spec = await run_in_threadpool(upload.handle_upload, filename, b"".join(chunks))
    except upload.UploadError as exc:
        log.info("数据集上传被拒绝（%s）：%s", exc.code, exc.detail)
        return _error(exc.status_code, exc.code, exc.message_key, exc.error_args, exc.detail)
    log.info("数据集 %s 上传成功（来源 %s）", spec.id, filename)
    return {"dataset": registry.cache_state(spec)}


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> Any:
    """删除上传的数据集（内置集与未知 id 分别 403 / 404，docs/02 §7.5）。"""
    if registry.get_spec(dataset_id) is None:
        return _error(404, "dataset_not_found", "errors.dataset.notFound", {"id": dataset_id})
    if not upload.is_uploaded(dataset_id):
        return _error(403, "dataset_builtin", "errors.dataset.builtin", {"id": dataset_id})
    await run_in_threadpool(upload.remove_uploaded, dataset_id)
    return {"deleted": dataset_id}


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
