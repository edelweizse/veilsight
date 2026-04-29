from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

from .analytics_engine import AnalyticsEngine, AnalyticsFrameResult
from .analytics_models import (
    AnalyticsEvent,
    AnalyticsQueueStats,
    AnalyticsRetention,
    AnalyticsRule,
    AnalyticsRuleCreate,
    AnalyticsRuleUpdate,
    AnalyticsSnapshot,
    AnalyticsSummary,
    StreamIdentity,
    model_copy_with,
    model_to_dict,
)
from .analytics_store import AnalyticsStore


def _model_json(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True)
    if hasattr(model, "dict"):
        return model.dict(by_alias=True)
    return model


class AnalyticsService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.store = AnalyticsStore(settings.db_path)
        self.engine = AnalyticsEngine(settings)
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=int(settings.ingest_queue_capacity))
        self.ingest_dropped = 0
        self._worker_task: asyncio.Task[None] | None = None
        self._latest_snapshots: dict[tuple[str, str], AnalyticsSnapshot] = {}
        self._sockets: set[WebSocket] = set()
        self._socket_lock = asyncio.Lock()
        self._last_prune_ms = 0

    async def start(self) -> None:
        rules = await asyncio.to_thread(self.store.list_rules)
        self.engine.set_rules(rules)
        if self.settings.enabled and self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def close(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        self.store.close()

    async def handle_event(self, event: dict[str, Any]) -> None:
        if not self.settings.enabled or "frame" not in event:
            return
        frame = event["frame"]
        try:
            self.queue.put_nowait(frame)
            return
        except asyncio.QueueFull:
            self.ingest_dropped += 1

        if self.settings.ingest_drop_policy == "drop_oldest":
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
                self.queue.task_done()
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(frame)
        else:
            return

    async def list_rules(self, stream_id: str | None, profile: str | None) -> list[AnalyticsRule]:
        return await asyncio.to_thread(self.store.list_rules, stream_id, profile)

    async def create_rule(self, payload: AnalyticsRuleCreate) -> AnalyticsRule:
        ts_ms = int(time.time() * 1000)
        rule = AnalyticsRule(
            id=str(uuid4()),
            stream_id=payload.stream_id,
            profile=payload.profile,
            kind=payload.kind,
            name=payload.name,
            enabled=payload.enabled,
            geometry=payload.geometry,
            settings=payload.settings,
            created_at_ms=ts_ms,
            updated_at_ms=ts_ms,
        )
        created = await asyncio.to_thread(self.store.create_rule, rule)
        self.engine.upsert_rule(created)
        await self._refresh_rule_snapshots(created.stream_id, created.profile)
        return created

    async def update_rule(self, rule_id: str, payload: AnalyticsRuleUpdate) -> AnalyticsRule | None:
        current = await asyncio.to_thread(self.store.get_rule, rule_id)
        if current is None:
            return None
        patch = model_to_dict(payload, exclude_unset=True)
        merged = model_copy_with(
            current,
            {
                **patch,
                "updated_at_ms": int(time.time() * 1000),
            },
        )
        updated = await asyncio.to_thread(self.store.update_rule, merged)
        self.engine.upsert_rule(updated)
        await self._refresh_rule_snapshots(updated.stream_id, updated.profile)
        return updated

    async def delete_rule(self, rule_id: str) -> bool:
        current = await asyncio.to_thread(self.store.get_rule, rule_id)
        deleted = await asyncio.to_thread(self.store.delete_rule, rule_id)
        if deleted:
            self.engine.delete_rule(rule_id)
            if current is not None:
                await self._refresh_rule_snapshots(current.stream_id, current.profile)
        return deleted

    async def list_events(
        self,
        *,
        stream_id: str | None,
        profile: str | None,
        rule_id: str | None,
        limit: int,
    ) -> list[AnalyticsEvent]:
        return await asyncio.to_thread(
            self.store.list_events,
            stream_id=stream_id,
            profile=profile,
            rule_id=rule_id,
            limit=limit,
        )

    async def latest_snapshot(self, stream_id: str, profile: str = "ui") -> AnalyticsSnapshot:
        key = (stream_id, profile)
        if key in self._latest_snapshots:
            return self._latest_snapshots[key]
        stored = await asyncio.to_thread(self.store.latest_snapshot, stream_id, profile)
        if stored is not None:
            self._latest_snapshots[key] = stored
            return stored
        return self.engine.empty_snapshot(stream_id, profile)

    async def summary(self, stream_id: str | None, profile: str | None, window_s: int) -> AnalyticsSummary:
        since = int(time.time() * 1000) - max(1, window_s) * 1000
        event_count = await asyncio.to_thread(self.store.count_events_since, stream_id, profile, since)
        snapshot: AnalyticsSnapshot | None = None
        if stream_id is not None:
            snapshot = await self.latest_snapshot(stream_id, profile or "ui")
        return AnalyticsSummary(
            stream=StreamIdentity(stream_id=stream_id, profile=profile or "ui") if stream_id else None,
            window_s=window_s,
            retention=AnalyticsRetention(
                raw_track_samples_days=self.settings.retention.raw_track_samples_days,
                events_days=self.settings.retention.events_days,
                aggregate_snapshots_days=self.settings.retention.aggregate_snapshots_days,
            ),
            queue=self.queue_stats(),
            latest_snapshot=snapshot,
            recent_event_count=event_count,
        )

    def queue_stats(self) -> AnalyticsQueueStats:
        return AnalyticsQueueStats(
            capacity=int(self.settings.ingest_queue_capacity),
            size=self.queue.qsize(),
            dropped=self.ingest_dropped,
            drop_policy=self.settings.ingest_drop_policy,
        )

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._socket_lock:
            self._sockets.add(websocket)
            snapshots = list(self._latest_snapshots.values())
        for snapshot in snapshots:
            await websocket.send_text(json.dumps(_model_json(snapshot)))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._socket_lock:
            self._sockets.discard(websocket)

    async def _worker(self) -> None:
        while True:
            frame = await self.queue.get()
            try:
                result = self.engine.process_frame(frame)
                persisted_events = await asyncio.to_thread(self._persist_result, result)
                if persisted_events:
                    by_key = {(event.stream_id, event.profile): [] for event in persisted_events}
                    for event in persisted_events:
                        by_key[(event.stream_id, event.profile)].append(event)
                    for key, events in by_key.items():
                        snapshot = result.snapshot
                        if (snapshot.stream.stream_id, snapshot.stream.profile) == key:
                            snapshot = model_copy_with(
                                snapshot,
                                {"recent_events": events + snapshot.recent_events[len(result.events) :]},
                            )
                            result.snapshot = snapshot
                key = (result.snapshot.stream.stream_id, result.snapshot.stream.profile)
                self._latest_snapshots[key] = result.snapshot
                await self._broadcast(result.snapshot)
            finally:
                self.queue.task_done()

    def _persist_result(self, result: AnalyticsFrameResult) -> list[AnalyticsEvent]:
        snapshot = result.snapshot if result.persist_snapshot else None
        persisted_events = self.store.insert_frame_records(
            samples=result.samples,
            segments=result.segments,
            events=result.events,
            snapshot=snapshot,
        )
        now = int(time.time() * 1000)
        if now - self._last_prune_ms > 600_000:
            self.store.prune(now_ms=now, retention=self.settings.retention)
            self._last_prune_ms = now
        return persisted_events

    async def _broadcast(self, snapshot: AnalyticsSnapshot) -> None:
        message = json.dumps(_model_json(snapshot))
        async with self._socket_lock:
            targets = list(self._sockets)
        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_text(message)
            except RuntimeError:
                stale.append(websocket)
        if stale:
            async with self._socket_lock:
                for websocket in stale:
                    self._sockets.discard(websocket)

    async def _refresh_rule_snapshots(self, stream_id: str, profile: str) -> None:
        key = (stream_id, profile)
        snapshot = self._latest_snapshots.get(key)
        if snapshot is None:
            return
        updated = model_copy_with(snapshot, {"rules": await self.list_rules(stream_id, profile)})
        self._latest_snapshots[key] = updated
        await self._broadcast(updated)
