from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

from .analytics_models import model_to_dict
from .gallery_store import now_ms
from .render_models import RenderJobCreate, RenderJobStatus
from .settings import ControllerSettings


class RenderService:
    def __init__(self, settings: ControllerSettings) -> None:
        self.settings = settings
        self.jobs: dict[str, RenderJobStatus] = {}
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.subprocess_factory = asyncio.create_subprocess_exec

    async def preview_frame(self, input_path: str) -> bytes:
        self._ensure_binary()
        source = self._resolve_existing_file(input_path, "input_not_found")
        with tempfile.TemporaryDirectory(prefix="veilsight-preview-") as tmp:
            output = Path(tmp) / "preview.jpg"
            process = await self.subprocess_factory(
                str(self.settings.render.binary_path),
                "--preview-frame",
                "--input",
                str(source),
                "--output",
                str(output),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            if process.returncode != 0:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "preview_failed",
                        "stdout": stdout.decode("utf-8", errors="replace")[-2000:],
                        "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
                    },
                )
            if not output.exists():
                raise HTTPException(status_code=502, detail={"error": "preview_not_written"})
            return output.read_bytes()

    async def create_job(self, payload: RenderJobCreate) -> RenderJobStatus:
        self._ensure_binary()
        config = self._resolve_existing_file(payload.config_path, "config_not_found")
        source = self._resolve_existing_file(payload.input_path, "input_not_found")
        output = Path(payload.output_path).expanduser().resolve()
        if output.exists() and not payload.overwrite:
            raise HTTPException(status_code=409, detail={"error": "output_exists", "path": str(output)})
        output.parent.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job_dir = Path(tempfile.gettempdir()) / f"veilsight-render-{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        gallery_db = None if payload.no_gallery else Path(payload.gallery_db).expanduser().resolve() if payload.gallery_db else self.settings.gallery.db_path
        rules_yaml = self._write_rules_yaml(payload, job_dir)
        preview_jpeg = job_dir / "latest.jpg"
        events_jsonl = output.with_suffix(".events.jsonl")
        manifest = output.with_suffix(".manifest.json")
        ts = now_ms()

        job = RenderJobStatus(
            job_id=job_id,
            status="queued",
            created_at_ms=ts,
            updated_at_ms=ts,
            config_path=str(config),
            input_path=str(source),
            output_path=str(output),
            gallery_db=str(gallery_db) if gallery_db is not None else None,
            layers=payload.layers,
            rules_yaml_path=str(rules_yaml),
            events_jsonl_path=str(events_jsonl),
            manifest_path=str(manifest),
            preview_jpeg_path=str(preview_jpeg),
            timing_mode=payload.timing_mode,
            render_mode=payload.render_mode,
            no_gallery=payload.no_gallery,
        )
        self.jobs[job_id] = job

        command = [
            str(self.settings.render.binary_path),
            "--config",
            str(config),
            "--input",
            str(source),
            "--output",
            str(output),
            "--layers",
            ",".join(payload.layers),
            "--rules-yaml",
            str(rules_yaml),
            "--events-jsonl",
            str(events_jsonl),
            "--manifest",
            str(manifest),
            "--preview-jpeg",
            str(preview_jpeg),
            "--preview-every-frames",
            str(payload.preview_every_frames),
            "--timing-mode",
            payload.timing_mode,
        ]
        if payload.no_gallery:
            command.append("--no-gallery")
        elif gallery_db is not None:
            command.extend(["--gallery-db", str(gallery_db)])
        if payload.overwrite:
            command.append("--overwrite")
        if payload.timing_mode == "custom" and payload.fps is not None:
            command.extend(["--fps", str(payload.fps)])
        if payload.timing_mode == "custom" and payload.source_fps is not None:
            command.extend(["--source-fps", str(payload.source_fps)])
        if payload.render_mode:
            command.extend(["--mode", payload.render_mode])
        if payload.stream_id:
            command.extend(["--stream-id", payload.stream_id])

        asyncio.create_task(self._run_job(job_id, command))
        return self.jobs[job_id]

    def list_jobs(self) -> list[RenderJobStatus]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at_ms, reverse=True)

    def get_job(self, job_id: str) -> RenderJobStatus:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "render_job_not_found"})
        return job

    def preview_jpeg(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        if not job.preview_jpeg_path:
            raise HTTPException(status_code=404, detail={"error": "preview_not_ready"})
        path = Path(job.preview_jpeg_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail={"error": "preview_not_ready"})
        return path

    async def cancel_job(self, job_id: str) -> RenderJobStatus:
        job = self.get_job(job_id)
        process = self.processes.get(job_id)
        if process is not None and job.status == "running":
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self._update_job(job_id, status="cancelled", message="cancelled by user")
        return self.jobs[job_id]

    async def _run_job(self, job_id: str, command: list[str]) -> None:
        self._update_job(job_id, status="running", started_at_ms=now_ms(), message="render process started")
        try:
            process = await self.subprocess_factory(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.processes[job_id] = process
            stdout_task = asyncio.create_task(self._read_stdout(job_id, process))
            stderr_task = asyncio.create_task(self._read_stderr(job_id, process))
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
            if self.jobs[job_id].status == "cancelled":
                self._update_job(job_id, returncode=process.returncode, finished_at_ms=now_ms())
            elif process.returncode == 0:
                self._update_job(
                    job_id,
                    status="succeeded",
                    returncode=process.returncode,
                    finished_at_ms=now_ms(),
                    message="render complete",
                )
            else:
                self._update_job(
                    job_id,
                    status="failed",
                    returncode=process.returncode,
                    finished_at_ms=now_ms(),
                    message="render failed",
                )
        except Exception as exc:  # pragma: no cover - defensive boundary around subprocess state
            self._update_job(job_id, status="failed", stderr_tail=str(exc), finished_at_ms=now_ms(), message="render failed to start")
        finally:
            self.processes.pop(job_id, None)

    def _update_job(self, job_id: str, **updates: Any) -> None:
        job = self.jobs[job_id]
        updates["updated_at_ms"] = now_ms()
        copier = getattr(job, "model_copy", None)
        self.jobs[job_id] = copier(update=updates) if copier else job.copy(update=updates)

    def _ensure_binary(self) -> None:
        binary = self.settings.render.binary_path
        if not binary.exists():
            raise HTTPException(status_code=503, detail={"error": "render_binary_missing", "path": str(binary)})

    @staticmethod
    def _resolve_existing_file(raw_path: str, error: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail={"error": error, "path": str(path)})
        return path

    async def _read_stdout(self, job_id: str, process: asyncio.subprocess.Process) -> None:
        async for line in self._stream_lines(getattr(process, "stdout", None)):
            self._handle_stdout_line(job_id, line.strip())

    async def _read_stderr(self, job_id: str, process: asyncio.subprocess.Process) -> None:
        tail = self.jobs[job_id].stderr_tail
        async for line in self._stream_lines(getattr(process, "stderr", None)):
            tail = (tail + line)[-8000:]
            self._update_job(job_id, stderr_tail=tail)

    @staticmethod
    async def _stream_lines(stream: Any):
        if stream is None:
            return
        if isinstance(stream, bytes):
            for line in stream.decode("utf-8", errors="replace").splitlines(True):
                yield line
            return
        if isinstance(stream, str):
            for line in stream.splitlines(True):
                yield line
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            yield line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)

    def _handle_stdout_line(self, job_id: str, line: str) -> None:
        parsed = self._parse_progress_line(line)
        if not parsed:
            return
        updates: dict[str, Any] = {}
        if "frame" in parsed:
            try:
                updates["progress_frame"] = int(parsed["frame"])
            except ValueError:
                pass
        if "total" in parsed:
            try:
                total = int(parsed["total"])
                if total > 0:
                    updates["total_frames"] = total
            except ValueError:
                pass
        effective_total = updates.get("total_frames", self.jobs[job_id].total_frames)
        effective_frame = updates.get("progress_frame", self.jobs[job_id].progress_frame)
        if effective_total and effective_frame is not None:
                updates["progress_percent"] = round(min(100.0, max(0.0, float(effective_frame) / float(effective_total) * 100.0)), 2)
        preview = parsed.get("preview")
        if preview:
            updates["preview_jpeg_path"] = preview
            path = Path(preview)
            if path.is_file():
                updates["preview_updated_at_ms"] = int(path.stat().st_mtime_ns / 1_000_000)
        elif self.jobs[job_id].preview_jpeg_path:
            path = Path(self.jobs[job_id].preview_jpeg_path or "")
            if path.is_file():
                updates["preview_updated_at_ms"] = int(path.stat().st_mtime_ns / 1_000_000)
        if updates:
            self._update_job(job_id, **updates)

    @staticmethod
    def _parse_progress_line(line: str) -> dict[str, str]:
        if not line.startswith("progress "):
            return {}
        parsed: dict[str, str] = {}
        for item in line.split()[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            parsed[key] = value
        return parsed

    @staticmethod
    def _write_rules_yaml(payload: RenderJobCreate, job_dir: Path) -> Path:
        path = job_dir / "rules.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"rules": [model_to_dict(rule) for rule in payload.rules]}, handle, sort_keys=False)
        return path
