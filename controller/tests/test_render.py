from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.veilsight_controller.api import RunnerClientRegistry, router
from controller.veilsight_controller.render_models import RenderJobCreate, RenderJobStatus
from controller.veilsight_controller.render_service import RenderService
from controller.veilsight_controller.settings import ControllerSettings


class FakeProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"progress frame=30\n", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.stdout = FakeStream(stdout)
        self.stderr = FakeStream(stderr)
        self.terminated = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(0)
        return self._stdout, self._stderr

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        await asyncio.sleep(0)
        return self.returncode


class FakeStream:
    def __init__(self, data: bytes) -> None:
        self.lines = data.splitlines(keepends=True)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        if not self.lines:
            return b""
        return self.lines.pop(0)


def render_settings(tmp_path: Path) -> ControllerSettings:
    settings = ControllerSettings()
    settings.render.binary_path = tmp_path / "veilsight_render_video"
    settings.render.binary_path.write_text("#!/bin/sh\n")
    settings.gallery.db_path = tmp_path / "gallery.sqlite3"
    return settings


def test_preview_endpoint_returns_503_if_binary_missing(tmp_path: Path) -> None:
    settings = ControllerSettings()
    settings.render.binary_path = tmp_path / "missing-render-binary"
    service = RenderService(settings)
    previous = RunnerClientRegistry.render_service
    RunnerClientRegistry.render_service = service
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        response = client.post("/api/renders/preview", json={"input_path": str(tmp_path / "input.mp4")})
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "render_binary_missing"
    finally:
        RunnerClientRegistry.render_service = previous


def test_job_creation_validates_missing_input_path(tmp_path: Path) -> None:
    async def run() -> None:
        settings = render_settings(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("streams: []\n")
        service = RenderService(settings)

        with pytest.raises(Exception) as exc:
            await service.create_job(
                RenderJobCreate(
                    config_path=str(config),
                    input_path=str(tmp_path / "missing.mp4"),
                    output_path=str(tmp_path / "out.mp4"),
                )
            )
        assert getattr(exc.value, "status_code", None) == 400
        assert exc.value.detail["error"] == "input_not_found"

    asyncio.run(run())


def test_job_creation_writes_rules_yaml_with_geometry(tmp_path: Path) -> None:
    async def run() -> None:
        settings = render_settings(tmp_path)
        config = tmp_path / "config.yaml"
        source = tmp_path / "input.mp4"
        config.write_text("streams: []\n")
        source.write_bytes(b"video")
        service = RenderService(settings)

        async def fake_subprocess(*args: Any, **kwargs: Any) -> FakeProcess:
            return FakeProcess()

        service.subprocess_factory = fake_subprocess
        job = await service.create_job(
            RenderJobCreate(
                config_path=str(config),
                input_path=str(source),
                output_path=str(tmp_path / "out.mp4"),
                rules=[
                    {
                        "id": "rule_1",
                        "kind": "line",
                        "name": "Line 1",
                        "geometry": {"points": [{"x": 120.0, "y": 300.0}, {"x": 620.0, "y": 300.0}]},
                        "settings": {"min_gap_ms": 1000},
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        data = yaml.safe_load(Path(job.rules_yaml_path or "").read_text())
        assert data["rules"][0]["geometry"]["points"][1] == {"x": 620.0, "y": 300.0}

    asyncio.run(run())


def test_no_gallery_job_omits_gallery_db_and_adds_flag(tmp_path: Path) -> None:
    async def run() -> None:
        settings = render_settings(tmp_path)
        config = tmp_path / "config.yaml"
        source = tmp_path / "input.mp4"
        config.write_text("streams: []\n")
        source.write_bytes(b"video")
        captured: list[str] = []
        service = RenderService(settings)

        async def fake_subprocess(*args: Any, **kwargs: Any) -> FakeProcess:
            captured.extend(str(arg) for arg in args)
            return FakeProcess()

        service.subprocess_factory = fake_subprocess
        await service.create_job(
            RenderJobCreate(config_path=str(config), input_path=str(source), output_path=str(tmp_path / "out.mp4"), no_gallery=True)
        )
        await asyncio.sleep(0)
        assert "--no-gallery" in captured
        assert "--gallery-db" not in captured

    asyncio.run(run())


def test_timing_mode_command_flags(tmp_path: Path) -> None:
    async def run() -> None:
        settings = render_settings(tmp_path)
        config = tmp_path / "config.yaml"
        source = tmp_path / "input.mp4"
        config.write_text("streams: []\n")
        source.write_bytes(b"video")
        captured: list[str] = []
        service = RenderService(settings)

        async def fake_subprocess(*args: Any, **kwargs: Any) -> FakeProcess:
            captured.extend(str(arg) for arg in args)
            return FakeProcess()

        service.subprocess_factory = fake_subprocess
        await service.create_job(
            RenderJobCreate(
                config_path=str(config),
                input_path=str(source),
                output_path=str(tmp_path / "out.mp4"),
                timing_mode="custom",
                fps=12.5,
            )
        )
        await asyncio.sleep(0)
        assert captured[captured.index("--timing-mode") + 1] == "custom"
        assert captured[captured.index("--fps") + 1] == "12.5"

        captured.clear()
        await service.create_job(RenderJobCreate(config_path=str(config), input_path=str(source), output_path=str(tmp_path / "out2.mp4")))
        await asyncio.sleep(0)
        assert captured[captured.index("--timing-mode") + 1] == "source"
        assert "--fps" not in captured

    asyncio.run(run())


def test_output_overwrite_guard(tmp_path: Path) -> None:
    async def run() -> None:
        settings = render_settings(tmp_path)
        config = tmp_path / "config.yaml"
        source = tmp_path / "input.mp4"
        output = tmp_path / "out.mp4"
        config.write_text("streams: []\n")
        source.write_bytes(b"video")
        output.write_bytes(b"existing")
        service = RenderService(settings)

        with pytest.raises(Exception) as exc:
            await service.create_job(
                RenderJobCreate(config_path=str(config), input_path=str(source), output_path=str(output))
            )
        assert getattr(exc.value, "status_code", None) == 409
        assert exc.value.detail["error"] == "output_exists"

    asyncio.run(run())


def test_job_status_transitions_for_mocked_subprocess(tmp_path: Path) -> None:
    async def run() -> None:
        settings = render_settings(tmp_path)
        config = tmp_path / "config.yaml"
        source = tmp_path / "input.mp4"
        config.write_text("streams: []\n")
        source.write_bytes(b"video")
        service = RenderService(settings)

        async def fake_subprocess(*args: Any, **kwargs: Any) -> FakeProcess:
            return FakeProcess(stdout=b"progress frame=30 total=120 preview=/tmp/missing.jpg\n", stderr=b"warn\n")

        service.subprocess_factory = fake_subprocess
        job = await service.create_job(
            RenderJobCreate(config_path=str(config), input_path=str(source), output_path=str(tmp_path / "out.mp4"))
        )
        assert job.status in {"queued", "running"}
        for _ in range(10):
            await asyncio.sleep(0)
            if service.get_job(job.job_id).status == "succeeded":
                break
        updated = service.get_job(job.job_id)
        assert updated.status == "succeeded"
        assert updated.progress_frame == 30
        assert updated.total_frames == 120
        assert updated.progress_percent == 25.0
        assert updated.returncode == 0
        assert "warn" in updated.stderr_tail

    asyncio.run(run())


def test_preview_endpoint_lifecycle(tmp_path: Path) -> None:
    settings = render_settings(tmp_path)
    service = RenderService(settings)
    ts = 1
    preview = tmp_path / "latest.jpg"
    service.jobs["job-1"] = RenderJobStatus(
        job_id="job-1",
        status="running",
        created_at_ms=ts,
        updated_at_ms=ts,
        config_path=str(tmp_path / "config.yaml"),
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(tmp_path / "out.mp4"),
        preview_jpeg_path=str(preview),
    )
    previous = RunnerClientRegistry.render_service
    RunnerClientRegistry.render_service = service
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        missing = client.get("/api/renders/jobs/job-1/preview.jpg")
        assert missing.status_code == 404
        assert missing.json()["detail"]["error"] == "preview_not_ready"

        preview.write_bytes(b"\xff\xd8\xff\xd9")
        response = client.get("/api/renders/jobs/job-1/preview.jpg")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"].startswith("image/jpeg")
    finally:
        RunnerClientRegistry.render_service = previous


def test_custom_timing_without_fps_returns_422(tmp_path: Path) -> None:
    settings = render_settings(tmp_path)
    service = RenderService(settings)
    previous = RunnerClientRegistry.render_service
    RunnerClientRegistry.render_service = service
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        response = client.post(
            "/api/renders/jobs",
            json={
                "config_path": str(tmp_path / "config.yaml"),
                "input_path": str(tmp_path / "input.mp4"),
                "output_path": str(tmp_path / "out.mp4"),
                "timing_mode": "custom",
            },
        )
        assert response.status_code == 422
    finally:
        RunnerClientRegistry.render_service = previous
