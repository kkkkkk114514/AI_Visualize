from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["models"])

KINDS = ("ml", "dl", "rl")


def _read_preset(model_id: str) -> dict | None:
    if "/" in model_id or "\\" in model_id or model_id.startswith("."):
        return None
    path = config.PRESETS_DIR / f"{model_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("预置模型 %s 读取失败：%s", path.name, exc)
        return None


@router.get("/models")
def list_models() -> dict:
    groups: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    for path in sorted(config.PRESETS_DIR.glob("*.json")):
        preset = _read_preset(path.stem)
        if preset is None:
            continue
        kind = preset.get("kind", "dl")
        groups.setdefault(kind, []).append(
            {
                "id": preset.get("id", path.stem),
                "kind": kind,
                "task": preset.get("task"),
                "name": preset.get("name"),
                "desc": preset.get("desc"),
                "source": "preset",
            }
        )
    return {"groups": groups}


@router.get("/models/{model_id}")
def get_model(model_id: str):
    preset = _read_preset(model_id)
    if preset is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "model_not_found",
                    "message_key": "errors.model.notFound",
                    "args": {"id": model_id},
                }
            },
        )
    return {"source": "preset", "graph": preset}
