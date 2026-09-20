from __future__ import annotations

import json
import logging

from fastapi import APIRouter

from app import config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["models"])

KINDS = ("ml", "dl", "rl")


@router.get("/models")
def list_models() -> dict:
    groups: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    for path in sorted(config.PRESETS_DIR.glob("*.json")):
        try:
            preset = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("预置模型 %s 读取失败：%s", path.name, exc)
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
