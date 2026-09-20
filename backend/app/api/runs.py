from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.api.ws import hub
from app.metrics import METRIC_NAMES, series_payload
from app.runners import base
from app.runners.manager import RunError, manager
from app.store import db, files

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/runs", tags=["runs"])


def _error(
    status_code: int, code: str, message_key: str, args: dict[str, Any] | None = None, detail: str = ""
) -> JSONResponse:
    payload: dict[str, Any] = {"code": code, "message_key": message_key, "args": args or {}}
    if detail:
        payload["detail"] = detail
    return JSONResponse(status_code=status_code, content={"error": payload})


def _run_error(exc: RunError) -> JSONResponse:
    return _error(exc.status_code, exc.code, exc.message_key, exc.error_args, exc.detail)


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    live = manager.live_info(record["id"])
    payload = {
        "id": record["id"],
        "name": record["name"],
        "kind": record["kind"],
        "model_id": record.get("model_id"),
        "dataset_id": record.get("dataset_id"),
        "hyperparams": record.get("hyperparams") or {},
        "probes": record.get("probes") or [],
        "status": record["status"],
        "device": record.get("device"),
        "seed": record.get("seed"),
        "created_at": record.get("created_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "error": record.get("error"),
        "best_metric": record.get("best_metric"),
        "total_steps": record.get("total_steps") or 0,
        "elapsed_s": None,
        "planned_steps": None,
    }
    if live:
        payload.update(
            {
                "status": live["status"],
                "step": live["step"],
                "epoch": live["epoch"],
                "device": live["device"],
                "elapsed_s": live["elapsed_s"],
                "planned_steps": live["planned_steps"],
                "best_metric": live["best_metric"],
                "total_steps": max(payload["total_steps"], live["step"]),
            }
        )
    return payload


@router.post("", status_code=201)
async def create_run(payload: dict[str, Any]) -> Any:
    try:
        record = await manager.start(payload)
    except RunError as exc:
        return _run_error(exc)
    return {"run": _summary(record)}


@router.get("")
async def list_runs(limit: int = 20, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    records = db.list_runs(limit, offset)
    return {"runs": [_summary(record) for record in records], "total": db.count_runs()}


@router.get("/{run_id}")
async def get_run(run_id: str) -> Any:
    record = db.get_run(run_id)
    if record is None:
        return _error(404, "run_not_found", "errors.run.notFound", {"id": run_id})
    payload = _summary(record)
    payload["graph"] = record.get("graph")
    return {"run": payload}


@router.post("/{run_id}/control")
async def control_run(run_id: str, payload: dict[str, Any]) -> Any:
    action = payload.get("action")
    if not isinstance(action, str):
        return _error(400, "run_bad_action", "errors.run.badAction", {"action": action})
    try:
        result = await manager.control(run_id, action, payload.get("value"))
    except RunError as exc:
        return _run_error(exc)
    return result


@router.get("/{run_id}/metrics")
async def run_metrics(
    run_id: str, names: str | None = None, max_points: int = config.METRICS_MAX_POINTS
) -> Any:
    record = db.get_run(run_id)
    if record is None:
        return _error(404, "run_not_found", "errors.run.notFound", {"id": run_id})
    wanted = tuple(item.strip() for item in names.split(",") if item.strip()) if names else METRIC_NAMES
    invalid = [name for name in wanted if name not in METRIC_NAMES]
    if invalid:
        return _error(422, "run_bad_metric", "errors.run.badMetric", {"names": invalid})

    series = db.fetch_metric_series(run_id, wanted)
    for step, epoch, name, value in manager.pending_metrics(run_id):
        if name not in wanted:
            continue
        entry = series.setdefault(name, {"steps": [], "values": []})
        if entry["steps"] and entry["steps"][-1] >= step:
            continue
        entry["steps"].append(step)
        entry["values"].append(value)
    return {
        "run_id": run_id,
        "status": record["status"],
        "total_steps": record.get("total_steps") or 0,
        "max_points": max(1, min(int(max_points), 20000)),
        "series": series_payload(series, max(1, min(int(max_points), 20000))),
    }


@router.get("/{run_id}/snapshots")
async def list_snapshots(
    run_id: str, step: int | None = None, node_id: str | None = None, kind: str | None = None
) -> Any:
    if db.get_run(run_id) is None:
        return _error(404, "run_not_found", "errors.run.notFound", {"id": run_id})
    return {"snapshots": db.list_snapshots(run_id, step=step, node_id=node_id, kind=kind)}


@router.get("/{run_id}/snapshots/{snapshot_id}")
async def get_snapshot(run_id: str, snapshot_id: str, request: Request) -> Any:
    record = db.get_snapshot(snapshot_id)
    if record is None or record["run_id"] != run_id:
        return _error(404, "snapshot_not_found", "errors.snapshot.notFound", {"id": snapshot_id})
    path = files.snapshot_path(run_id, snapshot_id)
    if not path.is_file():
        return _error(404, "snapshot_not_found", "errors.snapshot.notFound", {"id": snapshot_id})
    # 快照一次写定、内容不可变（docs/02 §6.3）：强 ETag + 长缓存，命中即 304
    etag = f'"{snapshot_id}"'
    headers = {"ETag": etag, "Cache-Control": "public, max-age=31536000, immutable"}
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type="application/json", headers=headers)


def _etag_matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    candidates = [item.strip() for item in header.split(",")]
    return etag in candidates or f"W/{etag}" in candidates or "*" in candidates


@router.delete("/{run_id}")
async def delete_run(run_id: str) -> Any:
    if manager.live_info(run_id) is not None:
        return _error(409, "run_active", "errors.run.deleteActive", {"id": run_id})
    if db.get_run(run_id) is None:
        return _error(404, "run_not_found", "errors.run.notFound", {"id": run_id})
    files.remove_run_dir(run_id)
    db.delete_run(run_id)
    await hub.broadcast({"type": base.WIRE_DELETED, "run_id": run_id})
    return {"run_id": run_id, "deleted": True}
