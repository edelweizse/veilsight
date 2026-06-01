"""ChokePoint runtime metrics.

Implements formulas from the methodology chapter:
  FPS, mean latency, p50/p95/p99.
Reads data from frame_runtime_log.csv produced by the C++ eval app.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class RuntimeRecord:
    frame_id: int
    latency_ms: float
    output_frame_emitted: bool
    dropped_frame: bool
    deadline_ms: float
    deadline_missed: bool


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no", "n", "none", "nan"}


def load_runtime_log(csv_path: Path) -> list[RuntimeRecord]:
    records: list[RuntimeRecord] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(RuntimeRecord(
                frame_id=int(row.get("frame_id", 0)),
                latency_ms=float(row.get("latency_ms", 0)),
                output_frame_emitted=_parse_bool(row.get("output_frame_emitted", "0")),
                dropped_frame=_parse_bool(row.get("dropped_frame", "0")),
                deadline_ms=float(row.get("deadline_ms", 0)),
                deadline_missed=_parse_bool(row.get("deadline_missed", "0")),
            ))
    return records


def fps_metric(processed_frames: int, processing_time_seconds: float) -> float:
    if processing_time_seconds <= 0:
        return 0.0
    return processed_frames / processing_time_seconds


def mean_latency(latencies_ms: Sequence[float]) -> float:
    if not latencies_ms:
        return 0.0
    return statistics.mean(latencies_ms)


def percentile_latency(latencies_ms: Sequence[float], p: float) -> float:
    if not latencies_ms:
        return 0.0
    sorted_lat = sorted(latencies_ms)
    k = (len(sorted_lat) - 1) * (p / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_lat):
        return sorted_lat[f] * (1.0 - c) + sorted_lat[f + 1] * c
    return sorted_lat[f]


def compute_all_runtime(records: list[RuntimeRecord], processing_time_seconds: float) -> dict:
    latencies = [r.latency_ms for r in records if r.output_frame_emitted]
    emitted = [r for r in records if r.output_frame_emitted]

    return {
        "fps": fps_metric(len(emitted), processing_time_seconds),
        "mean_latency_ms": mean_latency(latencies),
        "p50_ms": percentile_latency(latencies, 50),
        "p95_ms": percentile_latency(latencies, 95),
        "p99_ms": percentile_latency(latencies, 99),
    }
