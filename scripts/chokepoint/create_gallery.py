#!/usr/bin/env python3
"""Create MobileFaceNet gallery database for ChokePoint evaluation.

Reads groundtruth XML to determine person IDs, selects the configured enrollment
ratio, generates embeddings using the C++ enrollment tool, and creates a SQLite
gallery DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import subprocess
import tempfile
from pathlib import Path

from scripts.chokepoint.data_loader import get_all_person_ids

SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
  identity_key TEXT PRIMARY KEY,
  display_name TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at_ms INTEGER,
  updated_at_ms INTEGER
);

CREATE TABLE IF NOT EXISTS face_embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  identity_key TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT 'mobilefacenet',
  dim INTEGER NOT NULL DEFAULT 128,
  embedding BLOB NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at_ms INTEGER,
  source_type TEXT,
  FOREIGN KEY(identity_key) REFERENCES identities(identity_key)
);
"""


def encode_embedding(values: list[float]) -> bytes:
    if len(values) != 128:
        raise ValueError(f"MobileFaceNet embeddings must have exactly 128 dimensions, got {len(values)}")
    return struct.pack("<128f", *(float(v) for v in values))


def create_gallery(
    enrollment_binary: Path,
    config_path: Path,
    groundtruth_xmls: list[Path],
    faces_dirs: list[Path],
    output_db: Path,
    enroll_ratio: float,
    angles_step: int,
    min_embeddings_per_id: int,
) -> None:
    # Collect all unique person IDs from all sequences
    all_ids: list[str] = []
    seen: set[str] = set()
    for xml_path in groundtruth_xmls:
        for pid in get_all_person_ids(xml_path):
            if pid not in seen:
                seen.add(pid)
                all_ids.append(pid)
    all_ids.sort()
    if not all_ids:
        raise ValueError(f"No person IDs found in {len(groundtruth_xmls)} groundtruth XML files")

    num_enroll = max(1, int(len(all_ids) * enroll_ratio))
    enrolled = all_ids[:num_enroll]
    print(f"Total {len(all_ids)} people (from {len(groundtruth_xmls)} sequences), enrolling {len(enrolled)}: {enrolled}")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    output_db.unlink(missing_ok=True)

    conn = sqlite3.connect(str(output_db))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)

    import time
    now_ms = int(time.time() * 1000)

    total_embeddings = 0
    identities_with_embeddings = 0
    for identity_key in enrolled:
        # Gather .pgm files for this identity from ALL sequences
        pgm_files: list[Path] = []
        for faces_dir in faces_dirs:
            person_faces_dir = faces_dir / identity_key
            if person_faces_dir.is_dir():
                pgm_files.extend(
                    sorted(p for p in person_faces_dir.iterdir() if p.suffix.lower() == ".pgm")
                )

        if not pgm_files:
            print(f"  WARNING: no faces directory or .pgm files for {identity_key} in any sequence")
            continue

        selected = pgm_files
        print(f"  {identity_key}: enrolling all {len(selected)} faces from {len(faces_dirs)} sequences")

        # Copy selected files to temp dir for enrollment tool
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_person = Path(tmpdir) / identity_key
            tmp_person.mkdir()
            for pf in selected:
                dest = tmp_person / pf.name
                dest.write_bytes(pf.read_bytes())

            json_path = Path(tmpdir) / "embeddings.json"
            cmd = [
                str(enrollment_binary),
                "--config", str(config_path),
                "--faces-dir", str(tmp_person),
                "--output", str(json_path),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"    Enrollment tool failed: {result.stderr.strip()}")
                    continue
            except FileNotFoundError:
                print(f"    Enrollment binary not found: {enrollment_binary}")
                print("    Build it first: cmake --build build --target enroll_faces")
                conn.close()
                output_db.unlink(missing_ok=True)
                raise
            except subprocess.TimeoutExpired:
                print(f"    Enrollment tool timed out for {identity_key}")
                continue

            if not json_path.exists():
                print(f"    No embeddings output for {identity_key}")
                continue

            with open(json_path) as f:
                data = json.load(f)

            identity_embeddings = 0
            for entry in data:
                emb = entry.get("embedding", [])
                if len(emb) != 128:
                    continue
                key = entry.get("identity_key", identity_key)
                display_name = entry.get("display_name", key)
                conn.execute(
                    "INSERT OR IGNORE INTO identities (identity_key, display_name, active, created_at_ms) VALUES (?, ?, 1, ?)",
                    (key, display_name, now_ms),
                )
                blob = encode_embedding(emb)
                conn.execute(
                    "INSERT INTO face_embeddings (identity_key, model, dim, embedding, active, created_at_ms, source_type) VALUES (?, 'mobilefacenet', 128, ?, 1, ?, 'chokepoint_enrollment')",
                    (key, blob, now_ms),
                )
                identity_embeddings += 1

            print(f"    Enrolled {identity_embeddings} embeddings")
            total_embeddings += identity_embeddings
            if identity_embeddings:
                identities_with_embeddings += 1

    conn.commit()
    conn.close()

    print(f"\nGallery created: {output_db}")
    print(f"  {identities_with_embeddings}/{len(enrolled)} identities, {total_embeddings} total embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ChokePoint gallery database")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config YAML")
    parser.add_argument("--groundtruth-xmls", type=Path, nargs="+", required=True,
                        help="One or more groundtruth XML files with person IDs")
    parser.add_argument("--faces-dirs", type=Path, nargs="+", required=True,
                        help="One or more faces directories (e.g., faces/P1E_S2_C1)")
    parser.add_argument("--output-db", type=Path, required=True, help="Output SQLite gallery DB path")
    parser.add_argument("--enroll-ratio", type=float, default=0.5, help="Fraction of people to enroll")
    parser.add_argument("--angles-step", type=int, default=5, help="Deprecated; all .pgm files are enrolled")
    parser.add_argument("--min-embeddings", type=int, default=2, help="Deprecated; all .pgm files are enrolled")
    parser.add_argument("--enrollment-binary", type=Path, default=Path("build/apps/enroll_faces/enroll_faces"),
                        help="Path to C++ enrollment binary")
    args = parser.parse_args()

    create_gallery(
        enrollment_binary=args.enrollment_binary,
        config_path=args.config,
        groundtruth_xmls=args.groundtruth_xmls,
        faces_dirs=args.faces_dirs,
        output_db=args.output_db,
        enroll_ratio=args.enroll_ratio,
        angles_step=args.angles_step,
        min_embeddings_per_id=args.min_embeddings,
    )


if __name__ == "__main__":
    main()
