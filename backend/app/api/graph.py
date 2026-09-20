from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.graph import shapes
from app.graph.ir import IRParseError, parse_graph

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/graph", tags=["graph"])


def _parse_error_response(exc: IRParseError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_ir",
                "message_key": exc.message_key,
                "args": exc.error_args,
                "detail": exc.detail,
            }
        },
    )


@router.post("/infer")
async def infer_graph(payload: dict[str, Any]) -> Any:
    """形状与参数量推导（结构校验 + dry-run），返回逐节点结果。"""
    try:
        graph = parse_graph(payload)
    except IRParseError as exc:
        return _parse_error_response(exc)
    dataset_id = payload.get("dataset_id")
    result = await run_in_threadpool(
        shapes.analyze, graph, shapes.DRY_RUN_BATCH, dataset_id if isinstance(dataset_id, str) else None
    )
    return {
        "ok": result["ok"],
        "nodes": result["nodes"],
        "total_params": result["total_params"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "elapsed_ms": result["elapsed_ms"],
    }


@router.post("/validate")
async def validate_graph_ir(payload: dict[str, Any]) -> Any:
    """校验图 IR（结构 + dry-run），返回 errors / warnings。"""
    try:
        graph = parse_graph(payload)
    except IRParseError as exc:
        return _parse_error_response(exc)
    dataset_id = payload.get("dataset_id")
    result = await run_in_threadpool(
        shapes.analyze, graph, shapes.DRY_RUN_BATCH, dataset_id if isinstance(dataset_id, str) else None
    )
    return {
        "ok": result["ok"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "total_params": result["total_params"],
        "elapsed_ms": result["elapsed_ms"],
    }
