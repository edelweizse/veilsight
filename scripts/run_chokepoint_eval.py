#!/usr/bin/env python3
"""ChokePoint evaluation orchestrator.

Runs the complete evaluation pipeline:
  1. Builds C++ targets
  2. Creates gallery DB with half of subjects enrolled
  3. Runs C++ eval app for each video sequence
  4. Computes all non-body metrics per video and combined
  5. Outputs metrics_per_video.json, metrics_combined.json, metrics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.chokepoint.data_loader import (
    get_groundtruth_faces,
)
from scripts.chokepoint.metrics_detection import (
    FaceDecision,
    _Bbox,
    _coverage,
    face_detection_recall,
    face_detection_recall_center,
    face_false_allow_rate,
    non_gallery_face_anonymization_recall,
)
from scripts.chokepoint.metrics_gallery import (
    GalleryFaceObs,
    allow_stability,
    real_enrolled_gallery_allow_rate,
    time_to_first_allow,
)
from scripts.chokepoint.metrics_attack import (
    AttackRecord,
    _Roi,
    effective_face_region_coverage_ratio,
    face_detection_after_anonymization,
    face_protection_error_rate,
    face_reidentification_attack_success_rate,
    face_region_coverage_ratio,
)
from scripts.chokepoint.metrics_runtime import (
    compute_all_runtime,
    load_runtime_log,
)
from scripts.chokepoint.metrics_visual import (
    background_lpips,
    background_ssim,
)


@dataclass
class SequenceResult:
    sequence_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _read_face_log(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_anon_log(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_attack_log(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _face_log_to_detections(rows: list[dict]) -> dict[int, list[_Bbox]]:
    det: dict[int, list[_Bbox]] = {}
    for row in rows:
        fid = int(row.get("frame_id", 0))
        x = float(row.get("face_bbox_x", 0))
        y = float(row.get("face_bbox_y", 0))
        w = float(row.get("face_bbox_w", 0))
        h = float(row.get("face_bbox_h", 0))
        if w <= 0 or h <= 0:
            continue
        det.setdefault(fid, []).append(_Bbox(x, y, w, h))
    return det


def _face_log_to_decisions(rows: list[dict]) -> dict[int, list[FaceDecision]]:
    decisions: dict[int, list[FaceDecision]] = {}
    for row in rows:
        fid = int(row.get("frame_id", 0))
        x = float(row.get("face_bbox_x", 0))
        y = float(row.get("face_bbox_y", 0))
        w = float(row.get("face_bbox_w", 0))
        h = float(row.get("face_bbox_h", 0))
        bbox = _Bbox(x, y, w, h) if w > 0 and h > 0 else None
        decisions.setdefault(fid, []).append(FaceDecision(
            frame_id=fid,
            bbox=bbox,
            recognition_state=row.get("recognition_state", ""),
            privacy_action=row.get("privacy_action", ""),
        ))
    return decisions


def _anon_log_to_regions(rows: list[dict], target_types: set[str] | None = None) -> dict[int, list[_Bbox]]:
    regions: dict[int, list[_Bbox]] = {}
    for row in rows:
        target_type = row.get("target_type", "")
        if target_types is not None and target_type not in target_types:
            continue
        fid = int(row.get("frame_id", 0))
        x = float(row.get("roi_x", 0))
        y = float(row.get("roi_y", 0))
        w = float(row.get("roi_w", 0))
        h = float(row.get("roi_h", 0))
        if w <= 0 or h <= 0:
            continue
        regions.setdefault(fid, []).append(_Bbox(x, y, w, h))
    return regions


def compute_metrics_for_sequence(
    sequence: dict,
    output_dir: Path,
    enrolled_ids: set[str],
    fps: float,
    source_fps: float,
    thresholds: dict,
    processing_time_seconds: float,
) -> SequenceResult:
    seq_id = sequence["id"]
    result = SequenceResult(sequence_id=seq_id)

    try:
        gt_xml = REPO_ROOT / sequence["groundtruth_xml"]
        labels_json = REPO_ROOT / sequence["labels_json"]
        gt_faces = get_groundtruth_faces(
            gt_xml,
            labels_json,
            label_join=sequence.get("label_join", "task-index"),
        )

        face_log_path = output_dir / "face_log.csv"
        anon_log_path = output_dir / "anonymization_log.csv"
        runtime_log_path = output_dir / "frame_runtime_log.csv"
        attack_log_path = output_dir / "attack_log.csv"
        frame_out_dir = output_dir / "output_frames"
        mask_dir = output_dir / "masks"

        face_rows = _read_face_log(face_log_path) if face_log_path.exists() else []
        anon_rows = _read_anon_log(anon_log_path) if anon_log_path.exists() else []
        attack_rows = _read_attack_log(attack_log_path) if attack_log_path.exists() else []
        runtime_records = load_runtime_log(runtime_log_path) if runtime_log_path.exists() else []

        detected_bboxes = _face_log_to_detections(face_rows)
        face_decisions = _face_log_to_decisions(face_rows)
        anon_regions = _anon_log_to_regions(anon_rows)
        face_anon_regions = _anon_log_to_regions(anon_rows, {"face"})
        body_anon_regions = _anon_log_to_regions(anon_rows, {"body", "person"})

        iou_thr = thresholds.get("face_iou", 0.5)
        tau = thresholds.get("anonymization_coverage", 0.8)

        processed_fids: set[int] = {
            record.frame_id for record in runtime_records if record.output_frame_emitted
        } or set(detected_bboxes.keys())

        gt_bboxes_frame: dict[int, list[_Bbox]] = {}
        gt_face_rois_frame: dict[int, list[_Roi]] = {}
        non_gallery_faces_frame: dict[int, list[_Bbox]] = {}
        gallery_obs: list[GalleryFaceObs] = []
        non_gallery_rois_frame: dict[int, list[_Roi]] = {}
        gt_attack_face_dict: dict[int, list[dict]] = {}
        all_gt_person_ids: set[str] = set()

        for fid, gfs in gt_faces.items():
            if fid not in processed_fids:
                continue
            for gf in gfs:
                all_gt_person_ids.add(gf.person_id)
                if gf.bbox is not None:
                    fb = gf.bbox
                    roi = _Bbox(fb.x, fb.y, fb.width, fb.height)
                    gt_bboxes_frame.setdefault(fid, []).append(roi)
                    gt_face_rois_frame.setdefault(fid, []).append(
                        _Roi(fb.x, fb.y, fb.width, fb.height)
                    )
                    gt_attack_face_dict.setdefault(fid, []).append({
                        "person_id": gf.person_id,
                        "bbox": fb,
                    })
                    if gf.person_id in enrolled_ids and gf.bbox is not None:
                        regions = anon_regions.get(fid, [])
                        gallery_obs.append(GalleryFaceObs(
                            frame_id=fid,
                            person_id=gf.person_id,
                            # In the cross-system ChokePoint comparison, a face
                            # is in scope for gallery metrics when it appears in
                            # the GT labels. The old recognition_eligible label
                            # is intentionally not used as a denominator gate.
                            recognition_eligible=True,
                            allowed_raw=not any(_coverage(r, roi) >= tau for r in regions),
                        ))
                    else:
                        non_gallery_faces_frame.setdefault(fid, []).append(roi)
                        non_gallery_rois_frame.setdefault(fid, []).append(
                            _Roi(fb.x, fb.y, fb.width, fb.height)
                        )

        m = {}

        match_mode = thresholds.get("match_mode", "iou")
        if match_mode == "center":
            center_thr = thresholds.get("face_center_dist", 0.5)
            m["face_detection_recall"] = face_detection_recall_center(gt_bboxes_frame, detected_bboxes, center_thr)
        else:
            m["face_detection_recall"] = face_detection_recall(gt_bboxes_frame, detected_bboxes, iou_thr)

        # Protection metrics use non-gallery privacy targets. Gallery subjects
        # are intentionally allowed by policy and are measured by REGAR/TTFA.
        m["non_gallery_face_anonymization_recall"] = non_gallery_face_anonymization_recall(
            non_gallery_faces_frame, anon_regions, tau)
        m["face_false_allow_rate"] = face_false_allow_rate(
            non_gallery_faces_frame, face_decisions, anon_regions, tau, iou_thr)

        m["regar"] = real_enrolled_gallery_allow_rate(gallery_obs)
        ttfa = time_to_first_allow(gallery_obs, source_fps)
        m["ttfa_frames"] = ttfa["ttfa_frames"]
        m["ttfa_ms"] = ttfa["ttfa_ms"]
        m["allow_stability"] = allow_stability(gallery_obs)

        face_anon_rois = {
            fid: [_Roi(region.x, region.y, region.w, region.h) for region in regions]
            for fid, regions in face_anon_regions.items()
        }
        body_anon_rois = {
            fid: [_Roi(region.x, region.y, region.w, region.h) for region in regions]
            for fid, regions in body_anon_regions.items()
        }
        m["face_rcr"] = face_region_coverage_ratio(non_gallery_rois_frame, face_anon_rois)
        m["effective_face_rcr"] = effective_face_region_coverage_ratio(
            non_gallery_rois_frame, face_anon_rois, body_anon_rois, set())
        m["face_per"] = face_protection_error_rate(
            non_gallery_rois_frame, face_anon_rois, body_anon_rois, tau, set())

        if attack_log_path.exists():
            attack_records: list[AttackRecord] = []
            for row in attack_rows:
                fid = int(row.get("frame_id", 0))
                x = float(row.get("face_bbox_x", 0))
                y = float(row.get("face_bbox_y", 0))
                w = float(row.get("face_bbox_w", 0))
                h = float(row.get("face_bbox_h", 0))
                recognized = str(row.get("recognized", "0")).strip().lower() in {"1", "true", "yes"}
                attack_records.append(AttackRecord(
                    frame_id=fid,
                    face_bbox=_Roi(x, y, w, h),
                    predicted_identity=row.get("predicted_identity", ""),
                    identity_confidence=float(row.get("identity_confidence", 0)),
                    recognized=recognized,
                ))

            pre_anon_detection_count = sum(len(v) for v in detected_bboxes.values())
            non_gallery_probe_ids = all_gt_person_ids - enrolled_ids
            frasr = face_reidentification_attack_success_rate(
                gt_attack_face_dict, non_gallery_probe_ids, attack_records, iou_thr)
            m["frasr"] = frasr
            m["fda"] = face_detection_after_anonymization(attack_records, pre_anon_detection_count)
        else:
            m["frasr"] = None
            m["fda"] = None

        frame_ids = sorted(processed_fids)
        m["b_ssim"] = None
        original_frames_dir = REPO_ROOT / sequence["frames_dir"] if "frames_dir" in sequence else None
        if original_frames_dir and original_frames_dir.exists() and frame_out_dir.exists():
            m["b_ssim"] = background_ssim(
                frame_out_dir,
                mask_dir,
                frame_ids,
                original_frames_dir,
                gt_face_rois_frame=gt_face_rois_frame,
            )
            m["b_lpips"] = background_lpips(
                frame_out_dir,
                mask_dir,
                frame_ids,
                original_frames_dir,
                gt_face_rois_frame=gt_face_rois_frame,
            )
        else:
            m["b_lpips"] = None

        rt = compute_all_runtime(runtime_records, processing_time_seconds)
        m.update(rt)

        result.metrics = m

    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()

    return result


def run_eval_app(
    evaluator_binary: str,
    config_path: str,
    sequence: dict,
    output_dir: str,
    gallery_db: str,
    attack_gallery_db: str,
    fps: float,
    source_fps: float,
    deadline_ms: float,
) -> float:
    start = time.time()
    cmd = [
        evaluator_binary,
        "--config", config_path,
        "--frames-dir", str(REPO_ROOT / sequence["frames_dir"]),
        "--sequence-id", sequence["id"],
        "--output-dir", output_dir,
        "--gallery-db", gallery_db,
        "--attack-gallery-db", attack_gallery_db,
        "--fps", str(fps),
        "--source-fps", str(source_fps),
        "--deadline-ms", str(deadline_ms),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
        print(f"  WARNING: eval app returned {result.returncode}")
    else:
        print(f"  Done in {elapsed:.1f}s")
    print(f"  {result.stdout.strip()}")
    return elapsed


def run_gallery_creation(
    enrollment_binary: str,
    config: dict,
    gallery_db: str,
    enroll_ratio: float | None = None,
) -> None:
    from scripts.chokepoint.create_gallery import create_gallery

    enrollment = config["enrollment"]
    sequences = config["sequences"]

    groundtruth_xmls = [REPO_ROOT / s["groundtruth_xml"] for s in sequences]
    faces_dirs = [REPO_ROOT / "assets/chokepoint/faces" / s["id"] for s in sequences]

    create_gallery(
        enrollment_binary=Path(enrollment_binary),
        config_path=REPO_ROOT / config["paths"]["base_config"],
        groundtruth_xmls=groundtruth_xmls,
        faces_dirs=faces_dirs,
        output_db=Path(gallery_db),
        enroll_ratio=enrollment["ratio"] if enroll_ratio is None else enroll_ratio,
        angles_step=enrollment.get("face_angles_step", 5),
        min_embeddings_per_id=enrollment.get("min_embeddings_per_id", 2),
    )


def build_cpp_targets() -> bool:
    for target in ["veilsight_eval_chokepoint", "enroll_faces"]:
        print(f"Building {target}...")
        result = subprocess.run(
            ["cmake", "--build", str(REPO_ROOT / "build"), "--target", target, "-j2"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"Build failed for {target}:")
            print(result.stderr.strip()[-2000:])
            return False
        print(f"  {target} built successfully")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ChokePoint evaluation orchestrator")
    parser.add_argument("--config", type=Path, required=True, help="Evaluation config YAML")
    parser.add_argument("--skip-gallery", action="store_true", help="Skip gallery creation")
    parser.add_argument("--skip-build", action="store_true", help="Skip C++ build step")
    parser.add_argument("--sequences", nargs="*", default=None, help="Specific sequences to evaluate")
    args = parser.parse_args()

    config_path = args.config.resolve() if not args.config.is_absolute() else args.config
    if not config_path.exists():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_root = REPO_ROOT / config["paths"]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    gallery_db = output_root / "gallery.sqlite3"
    attack_gallery_db = output_root / "attack_gallery.sqlite3"
    gallery_db_str = str(gallery_db)
    attack_gallery_db_str = str(attack_gallery_db)

    if not args.skip_build:
        print("=== Building C++ targets ===")
        if not build_cpp_targets():
            print("Build failed. Fix issues and re-run, or use --skip-build to skip.")
            sys.exit(1)

    if not args.skip_gallery:
        print("=== Creating gallery database ===")
        run_gallery_creation(
            enrollment_binary=config["runtime"]["enrollment_binary"],
            config=config,
            gallery_db=gallery_db_str,
        )
        run_gallery_creation(
            enrollment_binary=config["runtime"]["enrollment_binary"],
            config=config,
            gallery_db=attack_gallery_db_str,
            enroll_ratio=1.0,
        )

    enrolled_ids: set[str] = set()
    import sqlite3
    if Path(gallery_db_str).exists():
        conn = sqlite3.connect(gallery_db_str)
        rows = conn.execute("SELECT identity_key FROM identities WHERE active = 1").fetchall()
        enrolled_ids = {r[0] for r in rows}
        conn.close()
        print(f"Gallery contains {len(enrolled_ids)} enrolled identities: {sorted(enrolled_ids)}")

    sequences = config["sequences"]
    if args.sequences:
        sequence_ids = set(args.sequences)
        sequences = [s for s in sequences if s["id"] in sequence_ids]

    fps = config["fps"]["target"]
    source_fps = config["fps"]["source"]
    thresholds = config["thresholds"]
    eval_binary = config["runtime"]["evaluator_binary"]

    print(f"\n=== Evaluating {len(sequences)} sequences at {fps} fps ===")
    results: dict[str, SequenceResult] = {}

    for i, seq in enumerate(sequences):
        seq_id = seq["id"]
        print(f"\n[{i+1}/{len(sequences)}] {seq_id}")
        seq_output_dir = output_root / seq_id
        seq_output_dir.mkdir(parents=True, exist_ok=True)

        elapsed = run_eval_app(
            evaluator_binary=eval_binary,
            config_path=str(REPO_ROOT / config["paths"]["base_config"]),
            sequence=seq,
            output_dir=str(seq_output_dir),
            gallery_db=gallery_db_str,
            attack_gallery_db=attack_gallery_db_str if attack_gallery_db.exists() else gallery_db_str,
            fps=fps,
            source_fps=source_fps,
            deadline_ms=thresholds.get("deadline_ms", 40.0),
        )

        result = compute_metrics_for_sequence(
            sequence=seq,
            output_dir=seq_output_dir,
            enrolled_ids=enrolled_ids,
            fps=fps,
            source_fps=source_fps,
            thresholds=thresholds,
            processing_time_seconds=elapsed,
        )
        results[seq_id] = result

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Metrics computed: {len(result.metrics)} values")

    per_video = {sid: r.metrics for sid, r in results.items() if r.metrics}
    with open(output_root / "metrics_per_video.json", "w") as f:
        json.dump(per_video, f, indent=2, default=str)
    print(f"\nPer-video metrics written to {output_root / 'metrics_per_video.json'}")

    combined: dict[str, Any] = {}
    metric_keys = [
        "face_detection_recall", "non_gallery_face_anonymization_recall",
        "face_false_allow_rate",
        "regar", "ttfa_frames", "ttfa_ms", "allow_stability",
        "face_rcr", "effective_face_rcr", "face_per",
        "frasr", "fda",
        "b_ssim", "b_lpips",
        "fps", "mean_latency_ms", "p50_ms", "p95_ms", "p99_ms",
    ]
    for key in metric_keys:
        values = []
        for r in results.values():
            v = r.metrics.get(key)
            if v is not None:
                values.append(v)
        if values:
            combined[key] = sum(values) / len(values)
        else:
            combined[key] = None

    with open(output_root / "metrics_combined.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Combined metrics written to {output_root / 'metrics_combined.json'}")

    with open(output_root / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence"] + metric_keys)
        for sid, r in results.items():
            row = [sid]
            for key in metric_keys:
                v = r.metrics.get(key)
                row.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            writer.writerow(row)
        row = ["combined"]
        for key in metric_keys:
            v = combined.get(key)
            row.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        writer.writerow(row)
    print(f"Summary CSV written to {output_root / 'metrics_summary.csv'}")

    failed = sum(1 for r in results.values() if r.error)
    if failed:
        print(f"\n{failed}/{len(results)} sequences had errors.")
        sys.exit(1)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
