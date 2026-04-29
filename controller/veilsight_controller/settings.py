from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "" or raw.strip().lower() == "null":
        return None
    return int(raw)


class AnalyticsHeatmapSettings:
    def __init__(self) -> None:
        self.rows = _env_int("VEILSIGHT_ANALYTICS_HEATMAP_ROWS", 36)
        self.cols = _env_int("VEILSIGHT_ANALYTICS_HEATMAP_COLS", 64)
        self.decay_half_life_s = _env_float("VEILSIGHT_ANALYTICS_HEATMAP_DECAY_HALF_LIFE_S", 900.0)


class AnalyticsRoutesSettings:
    def __init__(self) -> None:
        self.max_points_per_track = _env_int("VEILSIGHT_ANALYTICS_ROUTES_MAX_POINTS_PER_TRACK", 256)
        self.simplify_epsilon_px = _env_float("VEILSIGHT_ANALYTICS_ROUTES_SIMPLIFY_EPSILON_PX", 3.0)
        self.min_route_support = _env_int("VEILSIGHT_ANALYTICS_ROUTES_MIN_ROUTE_SUPPORT", 3)


class AnalyticsRetentionSettings:
    def __init__(self) -> None:
        self.raw_track_samples_days = _env_optional_int("VEILSIGHT_ANALYTICS_RETENTION_RAW_TRACK_SAMPLES_DAYS")
        self.events_days = _env_optional_int("VEILSIGHT_ANALYTICS_RETENTION_EVENTS_DAYS")
        self.aggregate_snapshots_days = _env_optional_int("VEILSIGHT_ANALYTICS_RETENTION_AGGREGATE_SNAPSHOTS_DAYS")


class AnalyticsSettings:
    def __init__(self) -> None:
        self.enabled = _env_bool("VEILSIGHT_ANALYTICS_ENABLED", True)
        self.db_path = Path(os.getenv("VEILSIGHT_ANALYTICS_DB_PATH", "data/veilsight_analytics.sqlite3")).resolve()
        self.ingest_queue_capacity = _env_int("VEILSIGHT_ANALYTICS_INGEST_QUEUE_CAPACITY", 2000)
        self.ingest_drop_policy = os.getenv("VEILSIGHT_ANALYTICS_INGEST_DROP_POLICY", "drop_oldest")
        self.snapshot_interval_ms = _env_int("VEILSIGHT_ANALYTICS_SNAPSHOT_INTERVAL_MS", 500)
        self.heatmap = AnalyticsHeatmapSettings()
        self.routes = AnalyticsRoutesSettings()
        self.retention = AnalyticsRetentionSettings()


class ControllerSettings:
    controller_id: str = "controller"

    def __init__(self) -> None:
        self.config_path = Path(os.getenv("VEILSIGHT_CONFIG", "configs/dual_example.yaml")).resolve()
        self.runner_grpc = os.getenv("VEILSIGHT_RUNNER_GRPC", "unix:///tmp/veilsight-runner.sock")
        self.runner_public_base_url = os.getenv("VEILSIGHT_RUNNER_PUBLIC_BASE_URL", "http://localhost:8080")
        self.web_dist = Path(os.getenv("VEILSIGHT_WEB_DIST", "web/dist")).resolve()
        self.analytics = AnalyticsSettings()


settings = ControllerSettings()
