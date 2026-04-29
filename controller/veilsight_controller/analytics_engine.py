from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Any

from .analytics_models import (
    AnalyticsEvent,
    AnalyticsRule,
    AnalyticsSnapshot,
    AnalyticsTrack,
    Counts,
    DensityLayer,
    DirectionVector,
    HeatmapLayer,
    Point,
    Rect,
    RouteSummary,
    StreamIdentity,
)


MAX_FRAME_DELTA_S = 1.0
MIN_FRAME_DELTA_S = 0.0


@dataclass
class TrackState:
    track_id: int
    foot: Point
    bbox: Rect
    last_ts_ms: int
    dwell_s: float = 0.0
    velocity_px_s: float = 0.0
    route_points: list[Point] = field(default_factory=list)


@dataclass
class AreaTrackState:
    inside: bool = False
    entered_at_ms: int | None = None
    total_dwell_s: float = 0.0
    threshold_emitted: bool = False


@dataclass
class RouteCluster:
    id: str
    points: list[Point]
    support: int
    last_seen_ms: int


@dataclass
class StreamState:
    stream_id: str
    profile: str
    width: int = 0
    height: int = 0
    frame_id: int = 0
    last_pts_ns: int | None = None
    last_ts_ms: int | None = None
    tracks: dict[int, TrackState] = field(default_factory=dict)
    active_track_ids: set[int] = field(default_factory=set)
    unique_track_ids: set[int] = field(default_factory=set)
    heatmap_values: list[float] = field(default_factory=list)
    density_values: list[float] = field(default_factory=list)
    area_states: dict[tuple[str, int], AreaTrackState] = field(default_factory=dict)
    line_debounce: dict[tuple[str, int, str], int] = field(default_factory=dict)
    routes: list[RouteCluster] = field(default_factory=list)
    route_counter: int = 0
    recent_events: deque[AnalyticsEvent] = field(default_factory=lambda: deque(maxlen=50))
    last_snapshot_persist_ms: int = 0


@dataclass
class AnalyticsFrameResult:
    samples: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    events: list[AnalyticsEvent]
    snapshot: AnalyticsSnapshot
    persist_snapshot: bool


def now_ms() -> int:
    return int(time.time() * 1000)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def side_of_line(a: Point, b: Point, p: Point) -> float:
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)


def segment_intersects(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return side_of_line(p, q, r)

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p.x, r.x) - 1e-9 <= q.x <= max(p.x, r.x) + 1e-9
            and min(p.y, r.y) - 1e-9 <= q.y <= max(p.y, r.y) + 1e-9
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-9 and on_segment(a, c, b):
        return True
    if abs(o2) < 1e-9 and on_segment(a, d, b):
        return True
    if abs(o3) < 1e-9 and on_segment(c, a, d):
        return True
    if abs(o4) < 1e-9 and on_segment(c, b, d):
        return True
    return False


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        intersects = (current.y > point.y) != (previous.y > point.y)
        if intersects:
            x_at_y = (previous.x - current.x) * (point.y - current.y) / (previous.y - current.y) + current.x
            if point.x < x_at_y:
                inside = not inside
        j = i
    return inside


def simplify_polyline(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) <= 2 or epsilon <= 0:
        return points[:]

    start = points[0]
    end = points[-1]
    max_distance = 0.0
    max_index = 0
    line_length = distance(start, end)
    for index, point in enumerate(points[1:-1], start=1):
        if line_length == 0:
            point_distance = distance(point, start)
        else:
            point_distance = abs(side_of_line(start, end, point)) / line_length
        if point_distance > max_distance:
            max_distance = point_distance
            max_index = index

    if max_distance > epsilon:
        left = simplify_polyline(points[: max_index + 1], epsilon)
        right = simplify_polyline(points[max_index:], epsilon)
        return left[:-1] + right
    return [start, end]


def parse_point(raw: Any) -> Point | None:
    if isinstance(raw, dict) and "x" in raw and "y" in raw:
        return Point(x=float(raw["x"]), y=float(raw["y"]))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return Point(x=float(raw[0]), y=float(raw[1]))
    return None


def line_points(geometry: dict[str, Any]) -> tuple[Point, Point] | None:
    raw_points = geometry.get("points")
    if isinstance(raw_points, list) and len(raw_points) >= 2:
        first = parse_point(raw_points[0])
        second = parse_point(raw_points[1])
        if first and second:
            return first, second
    first = parse_point(geometry.get("from") or geometry.get("p1") or geometry.get("start"))
    second = parse_point(geometry.get("to") or geometry.get("p2") or geometry.get("end"))
    if first and second:
        return first, second
    return None


def polygon_points(geometry: dict[str, Any]) -> list[Point]:
    raw_points = geometry.get("points") or geometry.get("polygon") or []
    if not isinstance(raw_points, list):
        return []
    points = [parse_point(item) for item in raw_points]
    return [point for point in points if point is not None]


class AnalyticsEngine:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.rows = int(settings.heatmap.rows)
        self.cols = int(settings.heatmap.cols)
        self.rules: dict[str, AnalyticsRule] = {}
        self._states: dict[tuple[str, str], StreamState] = {}

    def set_rules(self, rules: list[AnalyticsRule]) -> None:
        self.rules = {rule.id: rule for rule in rules}

    def upsert_rule(self, rule: AnalyticsRule) -> None:
        self.rules[rule.id] = rule

    def delete_rule(self, rule_id: str) -> None:
        self.rules.pop(rule_id, None)

    def process_frame(self, frame: dict[str, Any], receive_ts_ms: int | None = None) -> AnalyticsFrameResult:
        ts_ms = receive_ts_ms if receive_ts_ms is not None else now_ms()
        stream = frame.get("stream") or {}
        stream_id = str(stream.get("stream_id") or "")
        profile = str(stream.get("profile") or "ui")
        state = self._state(stream_id, profile)
        frame_id = int(frame.get("frame_id") or 0)
        pts_ns = int(frame.get("pts_ns") or 0)
        width = int(frame.get("width") or state.width or 0)
        height = int(frame.get("height") or state.height or 0)
        tracks = frame.get("tracks") if isinstance(frame.get("tracks"), list) else []

        state.frame_id = frame_id
        state.width = width
        state.height = height
        self._ensure_grid(state)
        delta_s = self._frame_delta_s(state, pts_ns, ts_ms)
        self._decay_heatmap(state, delta_s)

        current_ids = {int(track.get("id")) for track in tracks if track.get("id") is not None}
        events: list[AnalyticsEvent] = []
        for missing_id in state.active_track_ids - current_ids:
            events.extend(self._finalize_missing_track(state, missing_id, frame_id, pts_ns, ts_ms))

        density_values = [0.0] * (self.rows * self.cols)
        samples: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        snapshot_tracks: list[AnalyticsTrack] = []
        directions: list[DirectionVector] = []

        for track in tracks:
            track_id = int(track.get("id") or 0)
            bbox = Rect(
                x=float(track.get("x") or 0.0),
                y=float(track.get("y") or 0.0),
                w=float(track.get("w") or 0.0),
                h=float(track.get("h") or 0.0),
            )
            foot = Point(x=bbox.x + bbox.w / 2.0, y=bbox.y + bbox.h * 0.95)
            previous = state.tracks.get(track_id)
            velocity_px_s = 0.0
            if previous is None:
                previous = TrackState(
                    track_id=track_id,
                    foot=foot,
                    bbox=bbox,
                    last_ts_ms=ts_ms,
                    route_points=[foot],
                )
                state.tracks[track_id] = previous
            else:
                if delta_s > 0:
                    velocity_px_s = distance(previous.foot, foot) / delta_s
                    previous.dwell_s += delta_s
                if distance(previous.foot, foot) >= max(1.0, float(self.settings.routes.simplify_epsilon_px)):
                    previous.route_points.append(foot)
                    max_points = int(self.settings.routes.max_points_per_track)
                    if len(previous.route_points) > max_points:
                        previous.route_points = previous.route_points[-max_points:]
                segments.append(
                    {
                        "stream_id": stream_id,
                        "profile": profile,
                        "track_id": track_id,
                        "from_ts_ms": previous.last_ts_ms,
                        "to_ts_ms": ts_ms,
                        "x1": previous.foot.x,
                        "y1": previous.foot.y,
                        "x2": foot.x,
                        "y2": foot.y,
                        "speed_px_s": velocity_px_s,
                    }
                )
                if delta_s > 0:
                    directions.append(
                        DirectionVector(
                            track_id=track_id,
                            from_=previous.foot,
                            to=foot,
                            speed_px_s=velocity_px_s,
                        )
                    )
                events.extend(self._line_events(state, previous.foot, foot, track_id, frame_id, pts_ns, ts_ms))

            samples.append(
                {
                    "stream_id": stream_id,
                    "profile": profile,
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "pts_ns": pts_ns,
                    "ts_ms": ts_ms,
                    "x": bbox.x,
                    "y": bbox.y,
                    "w": bbox.w,
                    "h": bbox.h,
                    "foot_x": foot.x,
                    "foot_y": foot.y,
                    "score": float(track.get("score") or 0.0),
                }
            )
            cell = self._cell_index(foot, width, height)
            if cell is not None:
                state.heatmap_values[cell] += delta_s
                density_values[cell] += 1.0
            events.extend(self._area_events(state, foot, track_id, frame_id, pts_ns, ts_ms))

            previous.foot = foot
            previous.bbox = bbox
            previous.last_ts_ms = ts_ms
            previous.velocity_px_s = velocity_px_s
            state.unique_track_ids.add(track_id)
            snapshot_tracks.append(
                AnalyticsTrack(
                    id=track_id,
                    bbox=bbox,
                    foot=foot,
                    dwell_s=round(previous.dwell_s, 3),
                    velocity_px_s=round(velocity_px_s, 3),
                )
            )

        state.density_values = density_values
        state.active_track_ids = current_ids
        state.last_pts_ns = pts_ns
        state.last_ts_ms = ts_ms
        for event in events:
            state.recent_events.appendleft(event)

        snapshot = self._snapshot(state, snapshot_tracks, directions, ts_ms)
        persist_snapshot = ts_ms - state.last_snapshot_persist_ms >= int(self.settings.snapshot_interval_ms)
        if persist_snapshot:
            state.last_snapshot_persist_ms = ts_ms
        return AnalyticsFrameResult(
            samples=samples,
            segments=segments,
            events=events,
            snapshot=snapshot,
            persist_snapshot=persist_snapshot,
        )

    def empty_snapshot(self, stream_id: str, profile: str = "ui") -> AnalyticsSnapshot:
        layer_size = self.rows * self.cols
        return AnalyticsSnapshot(
            stream=StreamIdentity(stream_id=stream_id, profile=profile),
            heatmap=HeatmapLayer(rows=self.rows, cols=self.cols, values=[0.0] * layer_size),
            density=DensityLayer(rows=self.rows, cols=self.cols, values=[0.0] * layer_size),
            rules=self._rules_for_stream(stream_id, profile),
        )

    def _state(self, stream_id: str, profile: str) -> StreamState:
        key = (stream_id, profile)
        if key not in self._states:
            self._states[key] = StreamState(stream_id=stream_id, profile=profile)
            self._ensure_grid(self._states[key])
        return self._states[key]

    def _ensure_grid(self, state: StreamState) -> None:
        size = self.rows * self.cols
        if len(state.heatmap_values) != size:
            state.heatmap_values = [0.0] * size
        if len(state.density_values) != size:
            state.density_values = [0.0] * size

    def _frame_delta_s(self, state: StreamState, pts_ns: int, ts_ms: int) -> float:
        if state.last_pts_ns is not None and pts_ns > state.last_pts_ns:
            raw = (pts_ns - state.last_pts_ns) / 1_000_000_000.0
        elif state.last_ts_ms is not None and ts_ms > state.last_ts_ms:
            raw = (ts_ms - state.last_ts_ms) / 1000.0
        else:
            raw = 0.0
        return min(MAX_FRAME_DELTA_S, max(MIN_FRAME_DELTA_S, raw))

    def _decay_heatmap(self, state: StreamState, delta_s: float) -> None:
        half_life = float(self.settings.heatmap.decay_half_life_s)
        if delta_s <= 0 or half_life <= 0:
            return
        factor = 0.5 ** (delta_s / half_life)
        state.heatmap_values = [value * factor for value in state.heatmap_values]

    def _cell_index(self, point: Point, width: int, height: int) -> int | None:
        if width <= 0 or height <= 0:
            return None
        x = min(max(point.x, 0.0), max(0.0, float(width) - 1e-6))
        y = min(max(point.y, 0.0), max(0.0, float(height) - 1e-6))
        col = min(self.cols - 1, max(0, int(x / width * self.cols)))
        row = min(self.rows - 1, max(0, int(y / height * self.rows)))
        return row * self.cols + col

    def _rules_for_stream(self, stream_id: str, profile: str, *, enabled_only: bool = False) -> list[AnalyticsRule]:
        rules = [
            rule
            for rule in self.rules.values()
            if rule.stream_id == stream_id and rule.profile == profile and (rule.enabled or not enabled_only)
        ]
        return sorted(rules, key=lambda rule: rule.created_at_ms)

    def _line_events(
        self,
        state: StreamState,
        previous: Point,
        current: Point,
        track_id: int,
        frame_id: int,
        pts_ns: int,
        ts_ms: int,
    ) -> list[AnalyticsEvent]:
        events: list[AnalyticsEvent] = []
        if previous.x == current.x and previous.y == current.y:
            return events
        for rule in self._rules_for_stream(state.stream_id, state.profile, enabled_only=True):
            if rule.kind != "line":
                continue
            points = line_points(rule.geometry)
            if points is None:
                continue
            a, b = points
            previous_side = side_of_line(a, b, previous)
            current_side = side_of_line(a, b, current)
            if previous_side == 0 or current_side == 0:
                direction = "on_line"
            elif previous_side < 0 < current_side:
                direction = "negative_to_positive"
            elif previous_side > 0 > current_side:
                direction = "positive_to_negative"
            else:
                continue
            if not segment_intersects(previous, current, a, b):
                continue
            debounce_ms = int(rule.settings.get("min_gap_ms") or rule.settings.get("debounce_ms") or 1000)
            debounce_key = (rule.id, track_id, direction)
            last_event_ms = state.line_debounce.get(debounce_key, -debounce_ms)
            if ts_ms - last_event_ms < debounce_ms:
                continue
            state.line_debounce[debounce_key] = ts_ms
            events.append(
                AnalyticsEvent(
                    stream_id=state.stream_id,
                    profile=state.profile,
                    rule_id=rule.id,
                    kind="line_cross",
                    track_id=track_id,
                    direction=direction,
                    frame_id=frame_id,
                    pts_ns=pts_ns,
                    ts_ms=ts_ms,
                    position={"x": current.x, "y": current.y},
                    payload={"from": {"x": previous.x, "y": previous.y}, "to": {"x": current.x, "y": current.y}},
                )
            )
        return events

    def _area_events(
        self,
        state: StreamState,
        foot: Point,
        track_id: int,
        frame_id: int,
        pts_ns: int,
        ts_ms: int,
    ) -> list[AnalyticsEvent]:
        events: list[AnalyticsEvent] = []
        for rule in self._rules_for_stream(state.stream_id, state.profile, enabled_only=True):
            if rule.kind != "area":
                continue
            polygon = polygon_points(rule.geometry)
            inside = point_in_polygon(foot, polygon)
            key = (rule.id, track_id)
            area_state = state.area_states.setdefault(key, AreaTrackState())
            if inside and not area_state.inside:
                area_state.inside = True
                area_state.entered_at_ms = ts_ms
                area_state.threshold_emitted = False
                events.append(self._area_event(rule, "area_enter", track_id, frame_id, pts_ns, ts_ms, foot, {}))
            elif not inside and area_state.inside:
                dwell_s = self._area_current_dwell(area_state, ts_ms)
                area_state.total_dwell_s += dwell_s
                area_state.inside = False
                area_state.entered_at_ms = None
                area_state.threshold_emitted = False
                events.append(
                    self._area_event(
                        rule,
                        "area_exit",
                        track_id,
                        frame_id,
                        pts_ns,
                        ts_ms,
                        foot,
                        {"dwell_s": round(dwell_s, 3), "total_dwell_s": round(area_state.total_dwell_s, 3)},
                    )
                )
            elif inside and area_state.inside:
                threshold_s = rule.settings.get("dwell_threshold_s")
                if threshold_s is not None and not area_state.threshold_emitted:
                    dwell_s = self._area_current_dwell(area_state, ts_ms)
                    if dwell_s >= float(threshold_s):
                        area_state.threshold_emitted = True
                        events.append(
                            self._area_event(
                                rule,
                                "area_dwell_threshold",
                                track_id,
                                frame_id,
                                pts_ns,
                                ts_ms,
                                foot,
                                {"dwell_s": round(dwell_s, 3), "threshold_s": float(threshold_s)},
                            )
                        )
        return events

    def _area_event(
        self,
        rule: AnalyticsRule,
        kind: str,
        track_id: int,
        frame_id: int,
        pts_ns: int,
        ts_ms: int,
        foot: Point,
        payload: dict[str, Any],
    ) -> AnalyticsEvent:
        return AnalyticsEvent(
            stream_id=rule.stream_id,
            profile=rule.profile,
            rule_id=rule.id,
            kind=kind,
            track_id=track_id,
            frame_id=frame_id,
            pts_ns=pts_ns,
            ts_ms=ts_ms,
            position={"x": foot.x, "y": foot.y},
            payload=payload,
        )

    def _area_current_dwell(self, state: AreaTrackState, ts_ms: int) -> float:
        if state.entered_at_ms is None:
            return 0.0
        return max(0.0, (ts_ms - state.entered_at_ms) / 1000.0)

    def _finalize_missing_track(
        self,
        state: StreamState,
        track_id: int,
        frame_id: int,
        pts_ns: int,
        ts_ms: int,
    ) -> list[AnalyticsEvent]:
        track_state = state.tracks.get(track_id)
        if track_state is not None:
            self._learn_route(state, track_state.route_points, ts_ms)
            track_state.route_points = [track_state.foot]
        events: list[AnalyticsEvent] = []
        for rule in self._rules_for_stream(state.stream_id, state.profile, enabled_only=True):
            if rule.kind != "area":
                continue
            area_key = (rule.id, track_id)
            area_state = state.area_states.get(area_key)
            if area_state is None or not area_state.inside or track_state is None:
                continue
            dwell_s = self._area_current_dwell(area_state, ts_ms)
            area_state.total_dwell_s += dwell_s
            area_state.inside = False
            area_state.entered_at_ms = None
            area_state.threshold_emitted = False
            events.append(
                self._area_event(
                    rule,
                    "area_exit",
                    track_id,
                    frame_id,
                    pts_ns,
                    ts_ms,
                    track_state.foot,
                    {"dwell_s": round(dwell_s, 3), "total_dwell_s": round(area_state.total_dwell_s, 3)},
                )
            )
        return events

    def _learn_route(self, state: StreamState, raw_points: list[Point], ts_ms: int) -> None:
        if len(raw_points) < 2:
            return
        points = simplify_polyline(raw_points, float(self.settings.routes.simplify_epsilon_px))
        if len(points) < 2:
            return
        cluster = self._matching_route(state.routes, points)
        if cluster is None:
            state.route_counter += 1
            state.routes.append(
                RouteCluster(
                    id=f"route-{state.route_counter}",
                    points=points,
                    support=1,
                    last_seen_ms=ts_ms,
                )
            )
            return
        cluster.support += 1
        cluster.last_seen_ms = ts_ms

    def _matching_route(self, routes: list[RouteCluster], points: list[Point]) -> RouteCluster | None:
        threshold = max(20.0, float(self.settings.routes.simplify_epsilon_px) * 10.0)
        start = points[0]
        end = points[-1]
        for route in routes:
            if not route.points:
                continue
            forward = distance(route.points[0], start) + distance(route.points[-1], end)
            reverse = distance(route.points[0], end) + distance(route.points[-1], start)
            if min(forward, reverse) <= threshold:
                return route
        return None

    def _snapshot(
        self,
        state: StreamState,
        tracks: list[AnalyticsTrack],
        directions: list[DirectionVector],
        ts_ms: int,
    ) -> AnalyticsSnapshot:
        routes = [
            RouteSummary(id=route.id, points=route.points, support=route.support, last_seen_ms=route.last_seen_ms)
            for route in state.routes
            if route.support >= int(self.settings.routes.min_route_support)
        ]
        heatmap_max = max(state.heatmap_values) if state.heatmap_values else 0.0
        density_max = max(state.density_values) if state.density_values else 0.0
        return AnalyticsSnapshot(
            stream=StreamIdentity(stream_id=state.stream_id, profile=state.profile),
            frame_id=state.frame_id,
            timestamp_ms=ts_ms,
            width=state.width,
            height=state.height,
            counts=Counts(active_tracks=len(state.active_track_ids), unique_tracks=len(state.unique_track_ids)),
            tracks=tracks,
            heatmap=HeatmapLayer(
                rows=self.rows,
                cols=self.cols,
                values=[round(value, 6) for value in state.heatmap_values],
                max_value=round(heatmap_max, 6),
            ),
            density=DensityLayer(
                rows=self.rows,
                cols=self.cols,
                values=[round(value, 6) for value in state.density_values],
                max_value=round(density_max, 6),
            ),
            directions=directions,
            routes=routes,
            rules=self._rules_for_stream(state.stream_id, state.profile),
            recent_events=list(state.recent_events),
        )
