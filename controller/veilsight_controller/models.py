from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConfigPayload(BaseModel):
    yaml_text: str | None = None
    config: dict[str, Any] | None = None


class ConfigResponse(BaseModel):
    path: str
    yaml_text: str
    config: dict[str, Any] | None = None


class SaveConfigResponse(BaseModel):
    path: str
    saved: bool
    validation: dict[str, Any] | None = None


class ControllerHealth(BaseModel):
    ok: bool = True
    runner: dict[str, Any]
    connection_state: str


class StreamInfo(BaseModel):
    stream_id: str
    profile: str
    webrtc_available: bool = False
    webrtc_offer_url: str
    mjpeg_url: str
    snapshot_url: str
    metadata_url: str


class StreamsResponse(BaseModel):
    streams: list[StreamInfo] = Field(default_factory=list)
    public_base_url: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
