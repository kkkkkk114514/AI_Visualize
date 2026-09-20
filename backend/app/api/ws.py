"""WS Hub（契约见 docs/02 §8.2）：按 run 订阅分发 + 心跳；命令一律走 REST（D6）。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config

log = logging.getLogger(__name__)
router = APIRouter()


class Hub:
    """事件广播中心：带 run_id 的事件只发给订阅者，其余（如 dataset.progress）发给全部连接。"""

    def __init__(self) -> None:
        self._clients: dict[WebSocket, set[str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def register(self, websocket: WebSocket) -> None:
        self._clients[websocket] = set()

    def unregister(self, websocket: WebSocket) -> None:
        self._clients.pop(websocket, None)

    def subscribe(self, websocket: WebSocket, run_id: str) -> None:
        if websocket in self._clients:
            self._clients[websocket].add(run_id)

    def unsubscribe(self, websocket: WebSocket, run_id: str) -> None:
        if websocket in self._clients:
            self._clients[websocket].discard(run_id)

    def subscribed_runs(self) -> set[str]:
        return {run_id for subs in self._clients.values() for run_id in subs}

    async def broadcast(self, event: dict[str, Any]) -> None:
        if not self._clients:
            return
        run_id = event.get("run_id")
        payload = json.dumps(event, ensure_ascii=False)
        targets = [
            websocket
            for websocket, subscriptions in list(self._clients.items())
            if run_id is None or run_id in subscriptions
        ]
        for websocket in targets:
            try:
                await websocket.send_text(payload)
            except Exception:  # noqa: BLE001 - 连接已断，交给端点收尾
                self.unregister(websocket)

    def emit_threadsafe(self, event: dict[str, Any]) -> None:
        """供工作线程（如数据集下载）调用。"""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._emit_on_loop, event)

    def _emit_on_loop(self, event: dict[str, Any]) -> None:
        asyncio.ensure_future(self.broadcast(event))


hub = Hub()


async def _ping_loop(websocket: WebSocket) -> None:
    while True:
        await asyncio.sleep(config.WS_PING_INTERVAL_S)
        await websocket.send_json({"type": "ping"})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("ws 连接建立：%s", websocket.client)
    hub.register(websocket)
    ping_task = asyncio.create_task(_ping_loop(websocket))
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type == "subscribe":
                run_id = message.get("run_id")
                if isinstance(run_id, str) and run_id:
                    hub.subscribe(websocket, run_id)
            elif message_type == "unsubscribe":
                run_id = message.get("run_id")
                if isinstance(run_id, str) and run_id:
                    hub.unsubscribe(websocket, run_id)
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
        hub.unregister(websocket)
