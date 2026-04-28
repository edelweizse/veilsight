from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket


class TelemetryCache:
    def __init__(self) -> None:
        self.latest_metrics: dict[str, Any] = {}
        self.latest_status: dict[str, Any] = {}
        self.latest_analytics: dict[str, dict[str, Any]] = {}
        self.last_receive_timestamp_ms: int | None = None
        self.connection_state = "disconnected"
        self._analytics_sockets: set[WebSocket] = set()
        self._metrics_sockets: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        async with self._lock:
            if channel == "metrics":
                self._metrics_sockets.add(websocket)
            else:
                self._analytics_sockets.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._analytics_sockets.discard(websocket)
            self._metrics_sockets.discard(websocket)

    async def handle_event(self, event: dict[str, Any]) -> None:
        self.connection_state = "connected"
        self.last_receive_timestamp_ms = int(time.time() * 1000)
        if "metrics" in event:
            self.latest_metrics = event["metrics"]
            await self._broadcast(self._metrics_sockets, {"metrics": self.latest_metrics})
        elif "frame" in event:
            frame = event["frame"]
            stream = frame.get("stream", {})
            key = f"{stream.get('stream_id', '')}/{stream.get('profile', '')}"
            self.latest_analytics[key] = frame
            await self._broadcast(self._analytics_sockets, frame)
        elif "status" in event:
            self.latest_status = event["status"]
            await self._broadcast(self._metrics_sockets, {"status": self.latest_status})

    async def _broadcast(self, sockets: set[WebSocket], payload: dict[str, Any]) -> None:
        message = json.dumps(payload)
        async with self._lock:
            targets = list(sockets)
        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_text(message)
            except RuntimeError:
                stale.append(websocket)
        if stale:
            async with self._lock:
                for websocket in stale:
                    sockets.discard(websocket)
