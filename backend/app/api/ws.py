from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config

log = logging.getLogger(__name__)
router = APIRouter()


async def _ping_loop(websocket: WebSocket) -> None:
    while True:
        await asyncio.sleep(config.WS_PING_INTERVAL_S)
        await websocket.send_json({"type": "ping"})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("ws 连接建立：%s", websocket.client)
    subscriptions: set[str] = set()
    ping_task = asyncio.create_task(_ping_loop(websocket))
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type == "subscribe":
                run_id = message.get("run_id")
                if isinstance(run_id, str) and run_id:
                    subscriptions.add(run_id)
            elif message_type == "unsubscribe":
                subscriptions.discard(message.get("run_id"))
            elif message_type == "pong":
                continue
            else:
                log.warning("ws 收到未知消息类型：%r", message_type)
    except WebSocketDisconnect:
        log.info("ws 连接断开：%s", websocket.client)
    except (ValueError, TypeError) as exc:
        log.warning("ws 收到非法消息：%s", exc)
    finally:
        ping_task.cancel()
