#!/usr/bin/env python3
"""Compute Veilsight ChokePoint metrics from an existing result-log directory."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_chokepoint_eval import compute_metrics_for_sequence


def _load_enrolled_ids(args: argparse.Namespace) -> set[str]:
    enrolled_ids = {v.strip() for v in args.enrolled_ids.split(",") if v.strip()} if args.enrolled_ids else set()
    if enrolled_ids:
        return enrolled_ids

    if not args.gallery_db:
        return set()

    gallery_db = args.gallery_db if args.gallery_db.is_absolute() else REPO_ROOT / args.gallery_db
    if not gallery_db.exists():
        raise FileNotFoundError(f"Gallery DB does not exist: {gallery_db}")

    conn = sqlite3.connect(gallery_db)
    try:
        rows = conn.execute("SELECT identity_key FROM identities WHERE active = 1").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def _sequence_from_config(config: dict, sequence_id: str) -> dict:
    for sequence in config.get("sequences", []):
        if sequence.get("id") == sequence_id:
            return sequence
    raise ValueError(f"Sequence {sequence_id!r} is not listed in the config")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chokepoint_eval.yaml"))
    parser.add_argument("--sequence", required=True, help="Sequence ID from the ChokePoint config")
    parser.add_argument("--result-dir", type=Path, required=True, help="Directory with face_log.csv/anonymization_log.csv/frame_runtime_log.csv")
    parser.add_argument("--gallery-db", type=Path, default=None, help="SQLite gallery DB with active enrolled IDs")
    parser.add_argument("--enrolled-ids", default="", help="Comma-separated allowlist/gallery IDs, e.g. 0001,0003")
    parser.add_argument("--processing-time-seconds", type=float, default=0.0, help="Wall processing time for FPS; latency percentiles do not need this")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    result_dir = args.result_dir if args.result_dir.is_absolute() else REPO_ROOT / args.result_dir
    required = ["face_log.csv", "anonymization_log.csv", "frame_runtime_log.csv"]
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required log files in {result_dir}: {', '.join(missing)}")

    sequence = _sequence_from_config(config, args.sequence)
    enrolled_ids = _load_enrolled_ids(args)
    thresholds = config.get("thresholds", {})
    fps = float(config.get("fps", {}).get("target", 5.0))

    result = compute_metrics_for_sequence(
        sequence=sequence,
        output_dir=result_dir,
        enrolled_ids=enrolled_ids,
        fps=fps,
        thresholds=thresholds,
        processing_time_seconds=args.processing_time_seconds,
    )
    if result.error:
        raise RuntimeError(result.error)

    payload = {
        "sequence": args.sequence,
        "result_dir": str(result_dir),
        "enrolled_ids": sorted(enrolled_ids),
        "metrics": result.metrics,
    }

    if args.output_json:
        output_json = args.output_json if args.output_json.is_absolute() else REPO_ROOT / args.output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {output_json}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
