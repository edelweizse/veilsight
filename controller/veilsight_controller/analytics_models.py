from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 fallback
    ConfigDict = None  # type: ignore[assignment]


def model_to_dict(model: BaseModel, *, exclude_unset: bool = False, by_alias: bool = False) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if dump is not None:
        return dump(exclude_unset=exclude_unset, by_alias=by_alias)
    return model.dict(exclude_unset=exclude_unset, by_alias=by_alias)


def model_copy_with(model: BaseModel, update: dict[str, Any]) -> Any:
    copier = getattr(model, "model_copy", None)
    if copier is not None:
        return copier(update=update)
    return model.copy(update=update)


class Point(BaseModel):
    x: float
    y: float


class Rect(BaseModel):
    x: float
    y: float
    w: float
    h: float


class StreamIdentity(BaseModel):
    stream_id: str
    profile: str = "ui"


class AnalyticsRuleCreate(BaseModel):
    stream_id: str
    profile: str = "ui"
    kind: Literal["line", "area"]
    name: str
    enabled: bool = True
    geometry: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class AnalyticsRuleUpdate(BaseModel):
    stream_id: str | None = None
    profile: str | None = None
    kind: Literal["line", "area"] | None = None
    name: str | None = None
    enabled: bool | None = None
    geometry: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None


class AnalyticsRule(BaseModel):
    id: str
    stream_id: str
    profile: str = "ui"
    kind: Literal["line", "area"]
    name: str
    enabled: bool = True
    geometry: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at_ms: int
    updated_at_ms: int


class AnalyticsEvent(BaseModel):
    id: int | None = None
    stream_id: str
    profile: str = "ui"
    rule_id: str | None = None
    kind: str
    track_id: int | None = None
    direction: str | None = None
    frame_id: int
    pts_ns: int
    ts_ms: int
    position: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class Counts(BaseModel):
    active_tracks: int = 0
    unique_tracks: int = 0


class HeatmapLayer(BaseModel):
    rows: int
    cols: int
    values: list[float] = Field(default_factory=list)
    max_value: float = 0.0


class DensityLayer(BaseModel):
    rows: int
    cols: int
    values: list[float] = Field(default_factory=list)
    max_value: float = 0.0


class AnalyticsTrack(BaseModel):
    id: int
    bbox: Rect
    foot: Point
    dwell_s: float = 0.0
    velocity_px_s: float = 0.0


class DirectionVector(BaseModel):
    track_id: int
    from_: Point = Field(alias="from")
    to: Point
    speed_px_s: float = 0.0

    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - pydantic v1 fallback
        class Config:
            allow_population_by_field_name = True


class RouteSummary(BaseModel):
    id: str
    points: list[Point] = Field(default_factory=list)
    support: int = 0
    last_seen_ms: int = 0


class AnalyticsSnapshot(BaseModel):
    stream: StreamIdentity
    frame_id: int = 0
    timestamp_ms: int = 0
    width: int = 0
    height: int = 0
    counts: Counts = Field(default_factory=Counts)
    tracks: list[AnalyticsTrack] = Field(default_factory=list)
    heatmap: HeatmapLayer
    density: DensityLayer
    directions: list[DirectionVector] = Field(default_factory=list)
    routes: list[RouteSummary] = Field(default_factory=list)
    rules: list[AnalyticsRule] = Field(default_factory=list)
    recent_events: list[AnalyticsEvent] = Field(default_factory=list)


class AnalyticsRetention(BaseModel):
    raw_track_samples_days: int | None = None
    events_days: int | None = None
    aggregate_snapshots_days: int | None = None


class AnalyticsQueueStats(BaseModel):
    capacity: int
    size: int
    dropped: int
    drop_policy: str


class AnalyticsSummary(BaseModel):
    stream: StreamIdentity | None = None
    window_s: int
    retention: AnalyticsRetention
    queue: AnalyticsQueueStats
    latest_snapshot: AnalyticsSnapshot | None = None
    recent_event_count: int = 0
