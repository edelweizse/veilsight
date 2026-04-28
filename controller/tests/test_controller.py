from __future__ import annotations

import asyncio

from controller.veilsight_controller.api import translate_streams
from controller.veilsight_controller.runner_client import grpc_target
from controller.veilsight_controller.telemetry_cache import TelemetryCache


def test_grpc_target_unix_uri() -> None:
    assert grpc_target("unix:///tmp/x.sock") == "unix:/tmp/x.sock"


def test_stream_translation_preserves_snake_case_urls() -> None:
    response = translate_streams(
        {
            "public_base_url": "http://localhost:8080/",
            "streams": [{"stream_id": "cam0", "profile": "ui", "webrtc_available": False}],
        },
        "http://fallback:8080",
    )
    assert response.public_base_url == "http://localhost:8080"
    assert response.streams[0].stream_id == "cam0"
    assert response.streams[0].webrtc_available is False
    assert response.streams[0].webrtc_offer_url == "http://localhost:8080/webrtc/whep/cam0/ui"
    assert response.streams[0].mjpeg_url == "http://localhost:8080/video/cam0/ui"


def test_telemetry_cache_latest_values() -> None:
    cache = TelemetryCache()
    async def run() -> None:
        await cache.handle_event({"metrics": {"timestamp_ms": 1}})
        await cache.handle_event({"status": {"state": "running"}})
        await cache.handle_event({"frame": {"stream": {"stream_id": "cam0", "profile": "ui"}, "frame_id": 7}})

    asyncio.run(run())
    assert cache.latest_metrics["timestamp_ms"] == 1
    assert cache.latest_status["state"] == "running"
    assert cache.latest_analytics["cam0/ui"]["frame_id"] == 7
