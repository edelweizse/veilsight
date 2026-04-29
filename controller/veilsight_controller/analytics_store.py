from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .analytics_models import (
    AnalyticsEvent,
    AnalyticsRule,
    AnalyticsSnapshot,
    Counts,
    DensityLayer,
    HeatmapLayer,
    Point,
    RouteSummary,
    StreamIdentity,
    model_copy_with,
    model_to_dict,
)


SCHEMA_VERSION = "analytics_schema_v1"


class AnalyticsStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    applied_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analytics_rules (
                    id TEXT PRIMARY KEY,
                    stream_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    geometry_json TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS track_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    pts_ns INTEGER NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    w REAL NOT NULL,
                    h REAL NOT NULL,
                    foot_x REAL NOT NULL,
                    foot_y REAL NOT NULL,
                    score REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS track_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    from_ts_ms INTEGER NOT NULL,
                    to_ts_ms INTEGER NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    speed_px_s REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    rule_id TEXT,
                    kind TEXT NOT NULL,
                    track_id INTEGER,
                    direction TEXT,
                    frame_id INTEGER NOT NULL,
                    pts_ns INTEGER NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    position_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aggregate_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    window_s INTEGER NOT NULL,
                    counts_json TEXT NOT NULL,
                    heatmap_json TEXT NOT NULL,
                    density_json TEXT NOT NULL,
                    routes_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_analytics_rules_stream_profile
                    ON analytics_rules (stream_id, profile);
                CREATE INDEX IF NOT EXISTS idx_track_samples_stream_profile_ts
                    ON track_samples (stream_id, profile, ts_ms);
                CREATE INDEX IF NOT EXISTS idx_track_samples_stream_profile_track_ts
                    ON track_samples (stream_id, profile, track_id, ts_ms);
                CREATE INDEX IF NOT EXISTS idx_track_segments_stream_profile_track_ts
                    ON track_segments (stream_id, profile, track_id, from_ts_ms);
                CREATE INDEX IF NOT EXISTS idx_analytics_events_stream_profile_ts
                    ON analytics_events (stream_id, profile, ts_ms);
                CREATE INDEX IF NOT EXISTS idx_analytics_events_rule_ts
                    ON analytics_events (rule_id, ts_ms);
                CREATE INDEX IF NOT EXISTS idx_aggregate_snapshots_stream_profile_ts
                    ON aggregate_snapshots (stream_id, profile, ts_ms);
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO migrations (name, applied_at_ms) VALUES (?, ?)",
                (SCHEMA_VERSION, int(time.time() * 1000)),
            )
            self._conn.commit()

    def list_rules(self, stream_id: str | None = None, profile: str | None = None) -> list[AnalyticsRule]:
        sql = "SELECT * FROM analytics_rules"
        params: list[Any] = []
        where: list[str] = []
        if stream_id is not None:
            where.append("stream_id = ?")
            params.append(stream_id)
        if profile is not None:
            where.append("profile = ?")
            params.append(profile)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at_ms ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_rule(row) for row in rows]

    def get_rule(self, rule_id: str) -> AnalyticsRule | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM analytics_rules WHERE id = ?", (rule_id,)).fetchone()
        return self._row_to_rule(row) if row else None

    def create_rule(self, rule: AnalyticsRule) -> AnalyticsRule:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO analytics_rules (
                    id, stream_id, profile, kind, name, enabled, geometry_json, settings_json,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.id,
                    rule.stream_id,
                    rule.profile,
                    rule.kind,
                    rule.name,
                    int(rule.enabled),
                    json.dumps(rule.geometry, separators=(",", ":")),
                    json.dumps(rule.settings, separators=(",", ":")),
                    rule.created_at_ms,
                    rule.updated_at_ms,
                ),
            )
            self._conn.commit()
        return rule

    def update_rule(self, rule: AnalyticsRule) -> AnalyticsRule:
        with self._lock:
            self._conn.execute(
                """
                UPDATE analytics_rules
                   SET stream_id = ?, profile = ?, kind = ?, name = ?, enabled = ?,
                       geometry_json = ?, settings_json = ?, updated_at_ms = ?
                 WHERE id = ?
                """,
                (
                    rule.stream_id,
                    rule.profile,
                    rule.kind,
                    rule.name,
                    int(rule.enabled),
                    json.dumps(rule.geometry, separators=(",", ":")),
                    json.dumps(rule.settings, separators=(",", ":")),
                    rule.updated_at_ms,
                    rule.id,
                ),
            )
            self._conn.commit()
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM analytics_rules WHERE id = ?", (rule_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def insert_frame_records(
        self,
        *,
        samples: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        events: list[AnalyticsEvent],
        snapshot: AnalyticsSnapshot | None = None,
        window_s: int = 300,
    ) -> list[AnalyticsEvent]:
        with self._lock:
            if samples:
                self._conn.executemany(
                    """
                    INSERT INTO track_samples (
                        stream_id, profile, track_id, frame_id, pts_ns, ts_ms, x, y, w, h,
                        foot_x, foot_y, score
                    ) VALUES (
                        :stream_id, :profile, :track_id, :frame_id, :pts_ns, :ts_ms, :x, :y, :w, :h,
                        :foot_x, :foot_y, :score
                    )
                    """,
                    samples,
                )
            if segments:
                self._conn.executemany(
                    """
                    INSERT INTO track_segments (
                        stream_id, profile, track_id, from_ts_ms, to_ts_ms, x1, y1, x2, y2, speed_px_s
                    ) VALUES (
                        :stream_id, :profile, :track_id, :from_ts_ms, :to_ts_ms, :x1, :y1, :x2, :y2, :speed_px_s
                    )
                    """,
                    segments,
                )
            persisted_events: list[AnalyticsEvent] = []
            for event in events:
                cur = self._conn.execute(
                    """
                    INSERT INTO analytics_events (
                        stream_id, profile, rule_id, kind, track_id, direction, frame_id,
                        pts_ns, ts_ms, position_json, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.stream_id,
                        event.profile,
                        event.rule_id,
                        event.kind,
                        event.track_id,
                        event.direction,
                        event.frame_id,
                        event.pts_ns,
                        event.ts_ms,
                        json.dumps(event.position, separators=(",", ":")),
                        json.dumps(event.payload, separators=(",", ":")),
                    ),
                )
                persisted_events.append(model_copy_with(event, {"id": int(cur.lastrowid)}))
            if snapshot is not None:
                self._conn.execute(
                    """
                    INSERT INTO aggregate_snapshots (
                        stream_id, profile, ts_ms, window_s, counts_json, heatmap_json, density_json, routes_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.stream.stream_id,
                        snapshot.stream.profile,
                        snapshot.timestamp_ms,
                        window_s,
                        json.dumps(model_to_dict(snapshot.counts), separators=(",", ":")),
                        json.dumps(model_to_dict(snapshot.heatmap), separators=(",", ":")),
                        json.dumps(model_to_dict(snapshot.density), separators=(",", ":")),
                        json.dumps([model_to_dict(route) for route in snapshot.routes], separators=(",", ":")),
                    ),
                )
            self._conn.commit()
        return persisted_events

    def list_events(
        self,
        *,
        stream_id: str | None = None,
        profile: str | None = "ui",
        rule_id: str | None = None,
        limit: int = 100,
    ) -> list[AnalyticsEvent]:
        sql = "SELECT * FROM analytics_events"
        params: list[Any] = []
        where: list[str] = []
        if stream_id is not None:
            where.append("stream_id = ?")
            params.append(stream_id)
        if profile is not None:
            where.append("profile = ?")
            params.append(profile)
        if rule_id is not None:
            where.append("rule_id = ?")
            params.append(rule_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_ms DESC, id DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def latest_snapshot(self, stream_id: str, profile: str = "ui") -> AnalyticsSnapshot | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM aggregate_snapshots
                 WHERE stream_id = ? AND profile = ?
                 ORDER BY ts_ms DESC, id DESC
                 LIMIT 1
                """,
                (stream_id, profile),
            ).fetchone()
        if row is None:
            return None
        return AnalyticsSnapshot(
            stream=StreamIdentity(stream_id=row["stream_id"], profile=row["profile"]),
            timestamp_ms=int(row["ts_ms"]),
            counts=Counts(**json.loads(row["counts_json"])),
            heatmap=HeatmapLayer(**json.loads(row["heatmap_json"])),
            density=DensityLayer(**json.loads(row["density_json"])),
            routes=[RouteSummary(points=[Point(**point) for point in route.get("points", [])], **{k: v for k, v in route.items() if k != "points"}) for route in json.loads(row["routes_json"])],
            rules=self.list_rules(row["stream_id"], row["profile"]),
            recent_events=self.list_events(stream_id=row["stream_id"], profile=row["profile"], limit=25),
        )

    def count_events_since(self, stream_id: str | None, profile: str | None, since_ts_ms: int) -> int:
        sql = "SELECT COUNT(*) FROM analytics_events WHERE ts_ms >= ?"
        params: list[Any] = [since_ts_ms]
        if stream_id is not None:
            sql += " AND stream_id = ?"
            params.append(stream_id)
        if profile is not None:
            sql += " AND profile = ?"
            params.append(profile)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def prune(self, *, now_ms: int, retention: Any) -> None:
        deletions: list[tuple[str, int]] = []
        raw_days = getattr(retention, "raw_track_samples_days", None)
        events_days = getattr(retention, "events_days", None)
        snapshots_days = getattr(retention, "aggregate_snapshots_days", None)
        if raw_days is not None:
            cutoff = now_ms - int(raw_days * 86_400_000)
            deletions.append(("DELETE FROM track_samples WHERE ts_ms < ?", cutoff))
            deletions.append(("DELETE FROM track_segments WHERE to_ts_ms < ?", cutoff))
        if events_days is not None:
            deletions.append(("DELETE FROM analytics_events WHERE ts_ms < ?", now_ms - int(events_days * 86_400_000)))
        if snapshots_days is not None:
            deletions.append(
                ("DELETE FROM aggregate_snapshots WHERE ts_ms < ?", now_ms - int(snapshots_days * 86_400_000))
            )
        if not deletions:
            return
        with self._lock:
            for sql, cutoff in deletions:
                self._conn.execute(sql, (cutoff,))
            self._conn.commit()

    def _row_to_rule(self, row: sqlite3.Row) -> AnalyticsRule:
        return AnalyticsRule(
            id=row["id"],
            stream_id=row["stream_id"],
            profile=row["profile"],
            kind=row["kind"],
            name=row["name"],
            enabled=bool(row["enabled"]),
            geometry=json.loads(row["geometry_json"]),
            settings=json.loads(row["settings_json"]),
            created_at_ms=int(row["created_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
        )

    def _row_to_event(self, row: sqlite3.Row) -> AnalyticsEvent:
        return AnalyticsEvent(
            id=int(row["id"]),
            stream_id=row["stream_id"],
            profile=row["profile"],
            rule_id=row["rule_id"],
            kind=row["kind"],
            track_id=row["track_id"],
            direction=row["direction"],
            frame_id=int(row["frame_id"]),
            pts_ns=int(row["pts_ns"]),
            ts_ms=int(row["ts_ms"]),
            position=json.loads(row["position_json"]),
            payload=json.loads(row["payload_json"]),
        )
