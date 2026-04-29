from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.veilsight_controller.analytics_engine import AnalyticsEngine
from controller.veilsight_controller.analytics_models import AnalyticsEvent, AnalyticsRule
from controller.veilsight_controller.analytics_service import AnalyticsService
from controller.veilsight_controller.analytics_store import AnalyticsStore
from controller.veilsight_controller.api import RunnerClientRegistry, router


class HeatmapSettings:
    rows = 4
    cols = 4
    decay_half_life_s = 900.0


class RoutesSettings:
    max_points_per_track = 16
    simplify_epsilon_px = 1.0
    min_route_support = 1


class RetentionSettings:
    raw_track_samples_days = None
    events_days = None
    aggregate_snapshots_days = None


class AnalyticsTestSettings:
    enabled = True
    ingest_queue_capacity = 2
    ingest_drop_policy = "drop_oldest"
    snapshot_interval_ms = 1

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.heatmap = HeatmapSettings()
        self.routes = RoutesSettings()
        self.retention = RetentionSettings()


def rule(
    *,
    rule_id: str = "rule-1",
    kind: str = "line",
    geometry: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> AnalyticsRule:
    return AnalyticsRule(
        id=rule_id,
        stream_id="cam0",
        profile="ui",
        kind=kind,  # type: ignore[arg-type]
        name=rule_id,
        enabled=True,
        geometry=geometry or {},
        settings=settings or {},
        created_at_ms=1,
        updated_at_ms=1,
    )


def frame(frame_id: int, x: float, *, pts_ns: int, track_id: int = 1) -> dict[str, Any]:
    return {
        "stream": {"stream_id": "cam0", "profile": "ui"},
        "frame_id": frame_id,
        "pts_ns": pts_ns,
        "width": 120,
        "height": 100,
        "tracks": [{"id": track_id, "x": x, "y": 20, "w": 10, "h": 20, "score": 0.9}],
    }


def test_line_crossing_direction_and_debounce(tmp_path: Path) -> None:
    engine = AnalyticsEngine(AnalyticsTestSettings(tmp_path / "analytics.sqlite3"))
    engine.set_rules(
        [
            rule(
                geometry={"points": [{"x": 50, "y": 0}, {"x": 50, "y": 100}]},
                settings={"min_gap_ms": 5000},
            )
        ]
    )

    events: list[AnalyticsEvent] = []
    events.extend(engine.process_frame(frame(1, 35, pts_ns=0), receive_ts_ms=1000).events)
    events.extend(engine.process_frame(frame(2, 55, pts_ns=1_000_000_000), receive_ts_ms=2000).events)
    events.extend(engine.process_frame(frame(3, 35, pts_ns=2_000_000_000), receive_ts_ms=2500).events)
    events.extend(engine.process_frame(frame(4, 55, pts_ns=3_000_000_000), receive_ts_ms=3000).events)

    line_events = [event for event in events if event.kind == "line_cross"]
    assert [event.direction for event in line_events] == ["positive_to_negative", "negative_to_positive"]


def test_area_enter_exit_and_dwell_threshold(tmp_path: Path) -> None:
    engine = AnalyticsEngine(AnalyticsTestSettings(tmp_path / "analytics.sqlite3"))
    engine.set_rules(
        [
            rule(
                kind="area",
                geometry={"points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]},
                settings={"dwell_threshold_s": 1.0},
            )
        ]
    )

    events: list[AnalyticsEvent] = []
    events.extend(engine.process_frame(frame(1, 125, pts_ns=0), receive_ts_ms=0).events)
    events.extend(engine.process_frame(frame(2, 20, pts_ns=1_000_000_000), receive_ts_ms=1000).events)
    events.extend(engine.process_frame(frame(3, 22, pts_ns=2_000_000_000), receive_ts_ms=2500).events)
    events.extend(engine.process_frame(frame(4, 125, pts_ns=3_000_000_000), receive_ts_ms=3000).events)

    assert [event.kind for event in events] == ["area_enter", "area_dwell_threshold", "area_exit"]
    assert events[1].payload["dwell_s"] >= 1.0
    assert events[2].payload["total_dwell_s"] >= 2.0


def test_heatmap_density_uses_clamped_frame_delta(tmp_path: Path) -> None:
    engine = AnalyticsEngine(AnalyticsTestSettings(tmp_path / "analytics.sqlite3"))
    engine.process_frame(frame(1, 10, pts_ns=0), receive_ts_ms=0)
    result = engine.process_frame(frame(2, 10, pts_ns=10_000_000_000), receive_ts_ms=10_000)

    assert result.snapshot.heatmap.max_value <= 1.0
    assert result.snapshot.heatmap.max_value > 0.99
    assert result.snapshot.density.max_value == 1.0


def test_route_simplification_and_clustering(tmp_path: Path) -> None:
    settings = AnalyticsTestSettings(tmp_path / "analytics.sqlite3")
    settings.routes.min_route_support = 2
    engine = AnalyticsEngine(settings)

    engine.process_frame(frame(1, 5, pts_ns=0, track_id=1), receive_ts_ms=0)
    engine.process_frame(frame(2, 85, pts_ns=1_000_000_000, track_id=1), receive_ts_ms=1000)
    engine.process_frame({**frame(3, 0, pts_ns=2_000_000_000), "tracks": []}, receive_ts_ms=2000)
    engine.process_frame(frame(4, 6, pts_ns=3_000_000_000, track_id=2), receive_ts_ms=3000)
    engine.process_frame(frame(5, 84, pts_ns=4_000_000_000, track_id=2), receive_ts_ms=4000)
    result = engine.process_frame({**frame(6, 0, pts_ns=5_000_000_000), "tracks": []}, receive_ts_ms=5000)

    assert len(result.snapshot.routes) == 1
    assert result.snapshot.routes[0].support == 2
    assert len(result.snapshot.routes[0].points) == 2


def test_sqlite_migration_and_rule_event_persistence(tmp_path: Path) -> None:
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")
    created = store.create_rule(rule(geometry={"points": [{"x": 0, "y": 0}, {"x": 10, "y": 10}]}))
    events = store.insert_frame_records(
        samples=[],
        segments=[],
        events=[
            AnalyticsEvent(
                stream_id="cam0",
                profile="ui",
                rule_id=created.id,
                kind="line_cross",
                track_id=1,
                direction="positive_to_negative",
                frame_id=1,
                pts_ns=0,
                ts_ms=100,
                position={"x": 5, "y": 5},
                payload={},
            )
        ],
    )

    with sqlite3.connect(tmp_path / "analytics.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 1
    assert store.list_rules("cam0", "ui")[0].id == created.id
    assert events[0].id is not None
    assert store.list_events(stream_id="cam0", profile="ui", rule_id=created.id)[0].kind == "line_cross"
    store.close()


def test_bounded_queue_drop_behavior_is_nonblocking(tmp_path: Path) -> None:
    settings = AnalyticsTestSettings(tmp_path / "analytics.sqlite3")
    settings.ingest_queue_capacity = 1
    service = AnalyticsService(settings)

    async def run() -> None:
        await service.handle_event({"frame": frame(1, 10, pts_ns=0)})
        await service.handle_event({"frame": frame(2, 20, pts_ns=1_000_000)})

    asyncio.run(run())
    queued = service.queue.get_nowait()
    assert queued["frame_id"] == 2
    assert service.ingest_dropped == 1
    service.store.close()


def test_controller_analytics_api_rules_snapshot_and_event_filters(tmp_path: Path) -> None:
    service = AnalyticsService(AnalyticsTestSettings(tmp_path / "analytics.sqlite3"))
    previous = RunnerClientRegistry.analytics_service
    previous_registered = RunnerClientRegistry._analytics_callback_registered
    RunnerClientRegistry.analytics_service = service
    RunnerClientRegistry._analytics_callback_registered = False
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        response = client.post(
            "/api/analytics/rules",
            json={
                "stream_id": "cam0",
                "profile": "ui",
                "kind": "line",
                "name": "Door",
                "geometry": {"points": [{"x": 0, "y": 0}, {"x": 10, "y": 10}]},
            },
        )
        assert response.status_code == 200
        rule_id = response.json()["id"]
        assert client.get("/api/analytics/rules?stream_id=cam0&profile=ui").json()[0]["id"] == rule_id

        update = client.put(f"/api/analytics/rules/{rule_id}", json={"name": "Entry"})
        assert update.status_code == 200
        assert update.json()["name"] == "Entry"

        service.store.insert_frame_records(
            samples=[],
            segments=[],
            events=[
                AnalyticsEvent(
                    stream_id="cam0",
                    profile="ui",
                    rule_id=rule_id,
                    kind="line_cross",
                    track_id=7,
                    direction="positive_to_negative",
                    frame_id=1,
                    pts_ns=0,
                    ts_ms=100,
                    position={"x": 5, "y": 5},
                    payload={},
                )
            ],
        )
        events = client.get(f"/api/analytics/events?stream_id=cam0&profile=ui&rule_id={rule_id}").json()
        assert len(events) == 1
        assert events[0]["track_id"] == 7

        snapshot = client.get("/api/analytics/snapshot?stream_id=cam0&profile=ui").json()
        assert snapshot["heatmap"]["rows"] == 4
        assert snapshot["density"]["cols"] == 4
        assert snapshot["rules"][0]["id"] == rule_id

        delete = client.delete(f"/api/analytics/rules/{rule_id}")
        assert delete.status_code == 200
        assert delete.json() == {"deleted": True}
    finally:
        service.store.close()
        RunnerClientRegistry.analytics_service = previous
        RunnerClientRegistry._analytics_callback_registered = previous_registered
