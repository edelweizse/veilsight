from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
import sys
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict

GENERATED = Path(__file__).resolve().parents[1] / "generated"
if str(GENERATED) not in sys.path:
    sys.path.insert(0, str(GENERATED))

from veilsight.runner.v1 import runner_pb2, runner_pb2_grpc

TelemetryCallback = Callable[[dict[str, Any]], Awaitable[None]]


def grpc_target(address: str) -> str:
    if address.startswith("unix://"):
        return "unix:" + address.removeprefix("unix://")
    return address


def message_to_dict(message: Any) -> dict[str, Any]:
    return MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )


class RunnerClient:
    def __init__(self, address: str, timeout_s: float = 3.0) -> None:
        self.address = address
        self.timeout_s = timeout_s
        self._channel: grpc.aio.Channel | None = None
        self._control: runner_pb2_grpc.PipelineControlServiceStub | None = None
        self._telemetry: runner_pb2_grpc.RunnerTelemetryServiceStub | None = None
        self._telemetry_task: asyncio.Task[None] | None = None
        self._callbacks: list[TelemetryCallback] = []
        self._healthy = False
        self._lock = asyncio.Lock()

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def start(self) -> None:
        await self._connect()
        if self._telemetry_task is None:
            self._telemetry_task = asyncio.create_task(self._watch_telemetry())

    async def close(self) -> None:
        if self._telemetry_task:
            self._telemetry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._telemetry_task
        if self._channel:
            await self._channel.close()

    def add_telemetry_callback(self, callback: TelemetryCallback) -> None:
        self._callbacks.append(callback)

    async def health(self) -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.Health(
            runner_pb2.HealthRequest(controller_id="controller"),
            timeout=self.timeout_s,
        )
        self._healthy = bool(response.ok)
        return message_to_dict(response)

    async def status(self) -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.GetStatus(runner_pb2.GetStatusRequest(), timeout=self.timeout_s)
        return message_to_dict(response)

    async def streams(self) -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.GetStreams(runner_pb2.GetStreamsRequest(), timeout=self.timeout_s)
        return message_to_dict(response)

    async def validate_config(self, config_yaml: str) -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.ValidateConfig(
            runner_pb2.ValidateConfigRequest(config_yaml=config_yaml),
            timeout=self.timeout_s,
        )
        return message_to_dict(response)

    async def start_pipeline(self, config_yaml: str = "") -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.Start(
            runner_pb2.StartRequest(config_yaml=config_yaml),
            timeout=self.timeout_s,
        )
        return message_to_dict(response)

    async def stop_pipeline(self) -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.Stop(runner_pb2.StopRequest(), timeout=self.timeout_s)
        return message_to_dict(response)

    async def reload_pipeline(self, config_yaml: str) -> dict[str, Any]:
        control, _ = await self._stubs()
        response = await control.Reload(
            runner_pb2.ReloadRequest(config_yaml=config_yaml),
            timeout=max(self.timeout_s, 10.0),
        )
        return message_to_dict(response)

    async def metrics_snapshot(self) -> dict[str, Any]:
        _, telemetry = await self._stubs()
        response = await telemetry.GetMetricsSnapshot(
            runner_pb2.GetMetricsSnapshotRequest(),
            timeout=self.timeout_s,
        )
        return message_to_dict(response)

    async def _connect(self) -> None:
        async with self._lock:
            if self._channel is not None:
                return
            self._channel = grpc.aio.insecure_channel(grpc_target(self.address))
            self._control = runner_pb2_grpc.PipelineControlServiceStub(self._channel)
            self._telemetry = runner_pb2_grpc.RunnerTelemetryServiceStub(self._channel)

    async def _stubs(
        self,
    ) -> tuple[runner_pb2_grpc.PipelineControlServiceStub, runner_pb2_grpc.RunnerTelemetryServiceStub]:
        await self._connect()
        if self._control is None or self._telemetry is None:
            raise RuntimeError("runner client is not connected")
        return self._control, self._telemetry

    async def _watch_telemetry(self) -> None:
        while True:
            try:
                _, telemetry = await self._stubs()
                request = runner_pb2.WatchTelemetryRequest(
                    include_frame_analytics=True,
                    include_metrics=True,
                    include_logs=True,
                )
                async for event in telemetry.WatchTelemetry(request):
                    self._healthy = True
                    payload = message_to_dict(event)
                    await asyncio.gather(
                        *(callback(payload) for callback in self._callbacks),
                        return_exceptions=True,
                    )
            except asyncio.CancelledError:
                raise
            except grpc.aio.AioRpcError:
                self._healthy = False
                await asyncio.sleep(1.0)


class RunnerClientError(RuntimeError):
    pass


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
