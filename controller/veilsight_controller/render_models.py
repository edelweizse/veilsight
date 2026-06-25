from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


RenderLayer = Literal["tracks", "faces", "directions", "rules", "events"]
RenderTimingMode = Literal["source", "custom"]


class RenderPoint(BaseModel):
    x: float
    y: float


class RenderRule(BaseModel):
    id: str
    kind: Literal["line", "area"]
    name: str
    enabled: bool = True
    geometry: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class RenderPreviewRequest(BaseModel):
    input_path: str


class RenderJobCreate(BaseModel):
    config_path: str
    input_path: str
    output_path: str
    gallery_db: str | None = None
    layers: list[RenderLayer] = Field(default_factory=list)
    rules: list[RenderRule] = Field(default_factory=list)
    overwrite: bool = False
    timing_mode: RenderTimingMode = "source"
    render_mode: str = "face+body"
    fps: float | None = None
    source_fps: float | None = None
    no_gallery: bool = False
    preview_every_frames: int = Field(default=5, ge=1, le=300)
    stream_id: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> "RenderJobCreate":
        if self.timing_mode == "custom" and self.fps is None:
            raise ValueError("fps is required when timing_mode is custom")
        return self


class RenderJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at_ms: int
    updated_at_ms: int
    config_path: str
    input_path: str
    output_path: str
    gallery_db: str | None = None
    layers: list[RenderLayer] = Field(default_factory=list)
    rules_yaml_path: str | None = None
    events_jsonl_path: str | None = None
    manifest_path: str | None = None
    progress_frame: int | None = None
    total_frames: int | None = None
    progress_percent: float | None = None
    preview_jpeg_path: str | None = None
    preview_updated_at_ms: int | None = None
    timing_mode: RenderTimingMode = "source"
    render_mode: str = "face+body"
    no_gallery: bool = False
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    returncode: int | None = None
    stderr_tail: str = ""
    message: str = ""


class RenderSettingsResponse(BaseModel):
    binary_path: str
    default_gallery_db: str
