from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import grpc
import yaml
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .analytics_models import (
    AnalyticsEvent,
    AnalyticsRule,
    AnalyticsRuleCreate,
    AnalyticsRuleUpdate,
    AnalyticsSnapshot,
    AnalyticsSummary,
)
from .analytics_service import AnalyticsService
from .gallery_models import (
    EnrollmentCandidateResponse,
    EnrollmentStreamCapture,
    GalleryEmbedding,
    GalleryEmbeddingCreate,
    GalleryIdentity,
    GalleryIdentityCreate,
    GalleryIdentityUpdate,
    GalleryRefreshResponse,
)
from .gallery_service import GalleryService
from .gallery_store import GalleryStore
from .models import (
    ConfigFileInfo,
    ConfigListResponse,
    ConfigPayload,
    ConfigResponse,
    ConfigSelectPayload,
    ControllerHealth,
    SaveConfigResponse,
    StreamsResponse,
)
from .render_models import RenderJobCreate, RenderJobStatus, RenderPreviewRequest, RenderSettingsResponse
from .render_service import RenderService
from .runner_client import RunnerClient, read_text, write_text
from .settings import ControllerSettings, settings
from .telemetry_cache import TelemetryCache


router = APIRouter()


def get_settings() -> ControllerSettings:
    return settings


def get_runner(request_settings: Annotated[ControllerSettings, Depends(get_settings)]) -> RunnerClient:
    return RunnerClientRegistry.instance(request_settings)


def get_cache() -> TelemetryCache:
    return RunnerClientRegistry.cache


def get_analytics_service() -> AnalyticsService:
    return RunnerClientRegistry.analytics(settings)


def get_gallery_service() -> GalleryService:
    return RunnerClientRegistry.gallery(settings)


def get_render_service() -> RenderService:
    return RunnerClientRegistry.render(settings)


SettingsDep = Annotated[ControllerSettings, Depends(get_settings)]
RunnerDep = Annotated[RunnerClient, Depends(get_runner)]
CacheDep = Annotated[TelemetryCache, Depends(get_cache)]
AnalyticsDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
GalleryDep = Annotated[GalleryService, Depends(get_gallery_service)]
RenderDep = Annotated[RenderService, Depends(get_render_service)]


class RunnerClientRegistry:
    client: RunnerClient | None = None
    cache = TelemetryCache()
    analytics_service: AnalyticsService | None = None
    gallery_service: GalleryService | None = None
    render_service: RenderService | None = None
    _analytics_callback_registered = False

    @classmethod
    def instance(cls, request_settings: ControllerSettings) -> RunnerClient:
        if cls.client is None:
            cls.client = RunnerClient(request_settings.runner_grpc)
            cls.client.add_telemetry_callback(cls.cache.handle_event)
        if cls.analytics_service is not None and not cls._analytics_callback_registered:
            cls.client.add_telemetry_callback(cls.analytics_service.handle_event)
            cls._analytics_callback_registered = True
        return cls.client

    @classmethod
    def analytics(cls, request_settings: ControllerSettings) -> AnalyticsService:
        if cls.analytics_service is None:
            cls.analytics_service = AnalyticsService(request_settings.analytics)
        if cls.client is not None and not cls._analytics_callback_registered:
            cls.client.add_telemetry_callback(cls.analytics_service.handle_event)
            cls._analytics_callback_registered = True
        return cls.analytics_service

    @classmethod
    def gallery(cls, request_settings: ControllerSettings) -> GalleryService:
        if cls.gallery_service is None:
            cls.gallery_service = GalleryService(request_settings.gallery, GalleryStore(request_settings.gallery.db_path))
        return cls.gallery_service

    @classmethod
    def render(cls, request_settings: ControllerSettings) -> RenderService:
        if cls.render_service is None:
            cls.render_service = RenderService(request_settings)
        return cls.render_service


def payload_to_yaml(payload: ConfigPayload) -> str:
    if payload.yaml_text is not None:
        return payload.yaml_text
    if payload.config is not None:
        return yaml.safe_dump(payload.config, sort_keys=False)
    raise HTTPException(status_code=400, detail={"error": "yaml_text or config is required"})


def load_config(path: Path) -> ConfigResponse:
    yaml_text = read_text(path)
    parsed = yaml.safe_load(yaml_text) or {}
    return ConfigResponse(path=str(path), yaml_text=yaml_text, config=parsed)


def available_config_paths(request_settings: ControllerSettings) -> list[Path]:
    paths: set[Path] = set()
    if request_settings.config_dir.exists():
        for pattern in ("*.yaml", "*.yml"):
            paths.update(path.resolve() for path in request_settings.config_dir.glob(pattern) if path.is_file())
    if request_settings.config_path.exists():
        paths.add(request_settings.config_path.resolve())
    return sorted(paths, key=lambda path: (path.name.lower(), str(path)))


def resolve_config_choice(raw_path: str, request_settings: ControllerSettings) -> Path:
    requested = Path(raw_path)
    if not requested.is_absolute():
        requested = request_settings.config_dir / requested
    resolved = requested.resolve()
    if resolved not in set(available_config_paths(request_settings)):
        raise HTTPException(status_code=400, detail={"error": "config_file_not_available", "path": raw_path})
    return resolved


def normalize_base_url(raw: str | None, fallback: str) -> str:
    base = (raw or fallback).rstrip("/")
    return base or fallback.rstrip("/")


def translate_streams(raw: dict[str, Any], fallback_base_url: str) -> StreamsResponse:
    base_url = normalize_base_url(raw.get("public_base_url"), fallback_base_url)
    streams = []
    for stream in raw.get("streams", []):
        stream_id = stream.get("stream_id", "")
        profile = stream.get("profile", "ui")
        suffix = f"{stream_id}/{profile}"
        streams.append(
            {
                "stream_id": stream_id,
                "profile": profile,
                "webrtc_available": bool(stream.get("webrtc_available", True)),
                "webrtc_offer_url": f"{base_url}/webrtc/whep/{suffix}",
                "mjpeg_url": f"{base_url}/video/{suffix}",
                "snapshot_url": f"{base_url}/snapshot/{suffix}",
                "metadata_url": f"{base_url}/meta/{suffix}",
            }
        )
    return StreamsResponse(streams=streams, public_base_url=base_url)


def snapshot_url_for_stream(raw: dict[str, Any], fallback_base_url: str, stream_id: str, profile: str) -> str:
    translated = translate_streams(raw, fallback_base_url)
    for stream in translated.streams:
        if stream.stream_id == stream_id and stream.profile == profile:
            return stream.snapshot_url
    raise HTTPException(status_code=404, detail={"error": "stream_profile_not_found"})


def grpc_error(exc: grpc.aio.AioRpcError) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={"error": "runner_grpc_error", "detail": exc.details() or str(exc)},
    )


@router.get("/api/health", response_model=ControllerHealth)
async def health(client: RunnerDep, cache: CacheDep) -> ControllerHealth:
    try:
        runner = await client.health()
        cache.connection_state = "connected" if runner.get("ok") else "error"
    except grpc.aio.AioRpcError as exc:
        cache.connection_state = "disconnected"
        runner = {"ok": False, "error": exc.details() or str(exc)}
    return ControllerHealth(ok=True, runner=runner, connection_state=cache.connection_state)


@router.get("/api/config", response_model=ConfigResponse)
async def get_config(request_settings: SettingsDep) -> ConfigResponse:
    return load_config(request_settings.config_path)


@router.get("/api/configs", response_model=ConfigListResponse)
async def list_configs(request_settings: SettingsDep) -> ConfigListResponse:
    active_path = request_settings.config_path.resolve()
    files = [
        ConfigFileInfo(path=str(path), name=path.name, active=path == active_path)
        for path in available_config_paths(request_settings)
    ]
    return ConfigListResponse(active_path=str(active_path), files=files)


@router.post("/api/config/select", response_model=ConfigResponse)
async def select_config(payload: ConfigSelectPayload, request_settings: SettingsDep) -> ConfigResponse:
    request_settings.config_path = resolve_config_choice(payload.path, request_settings)
    return load_config(request_settings.config_path)


@router.put("/api/config", response_model=SaveConfigResponse)
async def put_config(
    payload: ConfigPayload,
    client: RunnerDep,
    request_settings: SettingsDep,
    validate: Annotated[bool, Query()] = True,
) -> SaveConfigResponse:
    yaml_text = payload_to_yaml(payload)
    validation: dict[str, Any] | None = None
    if validate:
        try:
            validation = await client.validate_config(yaml_text)
        except grpc.aio.AioRpcError as exc:
            raise grpc_error(exc) from exc
        if not validation.get("valid", False):
            raise HTTPException(status_code=422, detail=validation)
    write_text(request_settings.config_path, yaml_text)
    return SaveConfigResponse(path=str(request_settings.config_path), saved=True, validation=validation)


@router.post("/api/config/validate")
async def validate_config(payload: ConfigPayload, client: RunnerDep) -> dict[str, Any]:
    try:
        return await client.validate_config(payload_to_yaml(payload))
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.post("/api/pipeline/start")
async def start_pipeline(client: RunnerDep, request_settings: SettingsDep) -> dict[str, Any]:
    try:
        return await client.start_pipeline(read_text(request_settings.config_path))
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.post("/api/pipeline/stop")
async def stop_pipeline(client: RunnerDep) -> dict[str, Any]:
    try:
        return await client.stop_pipeline()
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.post("/api/pipeline/reload")
async def reload_pipeline(client: RunnerDep, request_settings: SettingsDep) -> dict[str, Any]:
    try:
        return await client.reload_pipeline(read_text(request_settings.config_path))
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.get("/api/pipeline/status")
async def pipeline_status(client: RunnerDep, cache: CacheDep) -> dict[str, Any]:
    try:
        status = await client.status()
        cache.latest_status = status
        return status
    except grpc.aio.AioRpcError as exc:
        cache.connection_state = "disconnected"
        return {"running": False, "state": "disconnected", "message": exc.details() or str(exc)}


@router.get("/api/streams", response_model=StreamsResponse)
async def streams(client: RunnerDep, request_settings: SettingsDep) -> StreamsResponse:
    try:
        raw = await client.streams()
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc
    return translate_streams(raw, request_settings.runner_public_base_url)


@router.get("/api/gallery/identities", response_model=list[GalleryIdentity])
async def gallery_identities(gallery: GalleryDep) -> list[GalleryIdentity]:
    return gallery.store.list_identities()


@router.post("/api/gallery/identities", response_model=GalleryIdentity)
async def create_gallery_identity(payload: GalleryIdentityCreate, gallery: GalleryDep) -> GalleryIdentity:
    try:
        return gallery.store.create_identity(
            display_name=payload.display_name,
            identity_key=payload.identity_key,
            active=payload.active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_identity", "detail": str(exc)}) from exc


@router.patch("/api/gallery/identities/{identity_key}", response_model=GalleryIdentity)
async def update_gallery_identity(
    identity_key: str,
    payload: GalleryIdentityUpdate,
    gallery: GalleryDep,
    client: RunnerDep,
    request_settings: SettingsDep,
) -> GalleryIdentity:
    try:
        identity = gallery.store.update_identity(identity_key, display_name=payload.display_name, active=payload.active)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_identity", "detail": str(exc)}) from exc
    if identity is None:
        raise HTTPException(status_code=404, detail={"error": "identity_not_found"})
    if request_settings.gallery.auto_refresh and payload.active is not None:
        try:
            await client.reload_gallery()
        except grpc.aio.AioRpcError:
            pass
    return identity


@router.delete("/api/gallery/identities/{identity_key}")
async def delete_gallery_identity(
    identity_key: str,
    gallery: GalleryDep,
    client: RunnerDep,
    request_settings: SettingsDep,
) -> dict[str, bool]:
    if not gallery.store.delete_identity(identity_key):
        raise HTTPException(status_code=404, detail={"error": "identity_not_found"})
    if request_settings.gallery.auto_refresh:
        try:
            await client.reload_gallery()
        except grpc.aio.AioRpcError:
            pass
    return {"deleted": True}


@router.get("/api/gallery/identities/{identity_key}/embeddings", response_model=list[GalleryEmbedding])
async def gallery_identity_embeddings(identity_key: str, gallery: GalleryDep) -> list[GalleryEmbedding]:
    if gallery.store.get_identity(identity_key) is None:
        raise HTTPException(status_code=404, detail={"error": "identity_not_found"})
    return gallery.store.list_embeddings(identity_key)


@router.delete("/api/gallery/embeddings/{embedding_id}", response_model=GalleryEmbedding)
async def delete_gallery_embedding(
    embedding_id: int,
    gallery: GalleryDep,
    client: RunnerDep,
    request_settings: SettingsDep,
) -> GalleryEmbedding:
    embedding = gallery.store.set_embedding_active(embedding_id, False)
    if embedding is None:
        raise HTTPException(status_code=404, detail={"error": "embedding_not_found"})
    if request_settings.gallery.auto_refresh:
        try:
            await client.reload_gallery()
        except grpc.aio.AioRpcError as exc:
            raise grpc_error(exc) from exc
    return embedding


@router.post("/api/gallery/enrollment-candidates", response_model=EnrollmentCandidateResponse)
async def gallery_enrollment_candidates(
    gallery: GalleryDep,
    client: RunnerDep,
    file: Annotated[UploadFile, File()],
) -> EnrollmentCandidateResponse:
    data = await file.read()
    try:
        return await gallery.analyze_image(
            image_bytes=data,
            mime_type=file.content_type or "application/octet-stream",
            source_type="upload",
            source_ref=file.filename,
            runner=client,
        )
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.post("/api/gallery/enrollment-candidates/from-stream", response_model=EnrollmentCandidateResponse)
async def gallery_enrollment_candidates_from_stream(
    payload: EnrollmentStreamCapture,
    gallery: GalleryDep,
    client: RunnerDep,
    request_settings: SettingsDep,
) -> EnrollmentCandidateResponse:
    try:
        raw_streams = await client.streams()
        snapshot_url = snapshot_url_for_stream(
            raw_streams,
            request_settings.runner_public_base_url,
            payload.stream_id,
            payload.profile,
        )
        return await gallery.analyze_stream_snapshot(
            stream_id=payload.stream_id,
            profile=payload.profile,
            runner=client,
            snapshot_url=snapshot_url,
        )
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.get("/api/gallery/enrollment-candidates/{enrollment_id}/image/{image_id}")
async def gallery_enrollment_image(enrollment_id: str, image_id: str, gallery: GalleryDep) -> Response:
    data, mime_type = gallery.get_image(enrollment_id, image_id)
    return Response(content=data, media_type=mime_type)


@router.post("/api/gallery/identities/{identity_key}/embeddings", response_model=list[GalleryEmbedding])
async def commit_gallery_embeddings(
    identity_key: str,
    payload: GalleryEmbeddingCreate,
    gallery: GalleryDep,
    client: RunnerDep,
) -> list[GalleryEmbedding]:
    try:
        return await gallery.commit_candidates(
            identity_key=identity_key,
            enrollment_id=payload.enrollment_id,
            candidate_ids=payload.candidate_ids,
            source_ref=payload.source_ref,
            runner=client,
        )
    except grpc.aio.AioRpcError:
        return gallery.store.list_embeddings(identity_key)[: len(payload.candidate_ids)]


@router.post("/api/gallery/refresh", response_model=GalleryRefreshResponse)
async def refresh_gallery(client: RunnerDep) -> GalleryRefreshResponse:
    try:
        raw = await client.reload_gallery()
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc
    return GalleryRefreshResponse(
        accepted=bool(raw.get("accepted", False)),
        message=str(raw.get("message", "")),
    )


@router.get("/api/renders/settings", response_model=RenderSettingsResponse)
async def render_settings(request_settings: SettingsDep) -> RenderSettingsResponse:
    return RenderSettingsResponse(
        binary_path=str(request_settings.render.binary_path),
        default_gallery_db=str(request_settings.gallery.db_path),
    )


@router.post("/api/renders/preview")
async def render_preview(payload: RenderPreviewRequest, renders: RenderDep) -> Response:
    data = await renders.preview_frame(payload.input_path)
    return Response(content=data, media_type="image/jpeg")


@router.post("/api/renders/jobs", response_model=RenderJobStatus)
async def create_render_job(payload: RenderJobCreate, renders: RenderDep) -> RenderJobStatus:
    return await renders.create_job(payload)


@router.get("/api/renders/jobs", response_model=list[RenderJobStatus])
async def list_render_jobs(renders: RenderDep) -> list[RenderJobStatus]:
    return renders.list_jobs()


@router.get("/api/renders/jobs/{job_id}", response_model=RenderJobStatus)
async def get_render_job(job_id: str, renders: RenderDep) -> RenderJobStatus:
    return renders.get_job(job_id)


@router.get("/api/renders/jobs/{job_id}/preview.jpg")
async def get_render_job_preview(job_id: str, renders: RenderDep) -> FileResponse:
    path = renders.preview_jpeg(job_id)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/api/renders/jobs/{job_id}/cancel", response_model=RenderJobStatus)
async def cancel_render_job(job_id: str, renders: RenderDep) -> RenderJobStatus:
    return await renders.cancel_job(job_id)


@router.get("/api/metrics")
async def metrics(client: RunnerDep, cache: CacheDep) -> dict[str, Any]:
    if cache.latest_metrics:
        return cache.latest_metrics
    try:
        return await client.metrics_snapshot()
    except grpc.aio.AioRpcError as exc:
        raise grpc_error(exc) from exc


@router.get("/api/analytics/latest")
async def analytics_latest(cache: CacheDep) -> dict[str, Any]:
    return {"analytics": cache.latest_analytics, "last_receive_timestamp_ms": cache.last_receive_timestamp_ms}


@router.get("/api/analytics/rules", response_model=list[AnalyticsRule])
async def analytics_rules(
    analytics: AnalyticsDep,
    stream_id: Annotated[str | None, Query()] = None,
    profile: Annotated[str | None, Query()] = "ui",
) -> list[AnalyticsRule]:
    return await analytics.list_rules(stream_id, profile)


@router.post("/api/analytics/rules", response_model=AnalyticsRule)
async def create_analytics_rule(payload: AnalyticsRuleCreate, analytics: AnalyticsDep) -> AnalyticsRule:
    return await analytics.create_rule(payload)


@router.put("/api/analytics/rules/{rule_id}", response_model=AnalyticsRule)
async def update_analytics_rule(
    rule_id: str,
    payload: AnalyticsRuleUpdate,
    analytics: AnalyticsDep,
) -> AnalyticsRule:
    rule = await analytics.update_rule(rule_id, payload)
    if rule is None:
        raise HTTPException(status_code=404, detail={"error": "analytics_rule_not_found"})
    return rule


@router.delete("/api/analytics/rules/{rule_id}")
async def delete_analytics_rule(rule_id: str, analytics: AnalyticsDep) -> dict[str, bool]:
    deleted = await analytics.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"error": "analytics_rule_not_found"})
    return {"deleted": True}


@router.get("/api/analytics/summary", response_model=AnalyticsSummary)
async def analytics_summary(
    analytics: AnalyticsDep,
    stream_id: Annotated[str | None, Query()] = None,
    profile: Annotated[str | None, Query()] = "ui",
    window_s: Annotated[int, Query(ge=1, le=86400)] = 300,
) -> AnalyticsSummary:
    return await analytics.summary(stream_id, profile, window_s)


@router.get("/api/analytics/events", response_model=list[AnalyticsEvent])
async def analytics_events(
    analytics: AnalyticsDep,
    stream_id: Annotated[str | None, Query()] = None,
    profile: Annotated[str | None, Query()] = "ui",
    rule_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[AnalyticsEvent]:
    return await analytics.list_events(stream_id=stream_id, profile=profile, rule_id=rule_id, limit=limit)


@router.get("/api/analytics/snapshot", response_model=AnalyticsSnapshot)
async def analytics_snapshot(
    analytics: AnalyticsDep,
    stream_id: Annotated[str, Query()],
    profile: Annotated[str, Query()] = "ui",
) -> AnalyticsSnapshot:
    return await analytics.latest_snapshot(stream_id, profile)


async def websocket_loop(websocket: WebSocket, channel: str, cache: TelemetryCache) -> None:
    await cache.connect(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await cache.disconnect(websocket)


@router.websocket("/ws/analytics")
async def analytics_ws(websocket: WebSocket, cache: CacheDep) -> None:
    await websocket_loop(websocket, "analytics", cache)


@router.websocket("/ws/analytics/overlays")
async def analytics_overlays_ws(websocket: WebSocket, analytics: AnalyticsDep) -> None:
    await analytics.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await analytics.disconnect(websocket)


@router.websocket("/ws/metrics")
async def metrics_ws(websocket: WebSocket, cache: CacheDep) -> None:
    await websocket_loop(websocket, "metrics", cache)
