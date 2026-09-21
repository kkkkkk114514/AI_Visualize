from __future__ import annotations

from fastapi import APIRouter

from app.algos import schema as algo_schema

router = APIRouter(prefix="/api", tags=["algos"])


@router.get("/algos")
def list_algos() -> dict:
    """ML / RL 算法参数 schema（契约见 docs/02 §13.2）：前端表单与服务端校验共用同一份表。"""
    return algo_schema.catalog()
