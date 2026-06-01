#!/usr/bin/env python3
"""Compare MOT20-style extreme-condition coverage for body and face systems.

This script treats public MOT20 detections as a practical reference signal, not
as true ground truth. Veilsight body boxes are treated as anonymization ROIs; the
friend system is face-only, so face ROIs are checked for intersection with the
same public body boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Box:
    frame: int
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    source_index: int = -1

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def contains(outer: Box, inner: Box) -> bool:
    return inner.x >= outer.x and inner.y >= outer.y and inner.x2 <= outer.x2 and inner.y2 <= outer.y2


def overlaps(a: Box, b: Box) -> bool:
    return min(a.x2, b.x2) > max(a.x, b.x) and min(a.y2, b.y2) > max(a.y, b.y)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def load_mot_boxes(path: Path) -> tuple[dict[int, list[Box]], dict[str, int]]:
    boxes_by_frame: dict[int, list[Box]] = {}
    stats = {"rows": 0, "valid_rows": 0, "invalid_rows": 0}

    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader, start=1):
            stats["rows"] += 1
            try:
                if len(row) < 6:
                    raise ValueError("not enough columns")
                frame = int(float(row[0]))
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                score = float(row[6]) if len(row) > 6 and row[6] != "" else 1.0
                if w <= 0.0 or h <= 0.0:
                    raise ValueError("non-positive box")
            except (TypeError, ValueError):
                stats["invalid_rows"] += 1
                continue

            stats["valid_rows"] += 1
            boxes_by_frame.setdefault(frame, []).append(Box(frame, x, y, w, h, score, row_index))

    return boxes_by_frame, stats


def load_face_boxes(path: Path) -> tuple[list[Box], dict[str, int]]:
    boxes: list[Box] = []
    stats = {"rows": 0, "valid_rows": 0, "invalid_rows": 0}
    required = ["frame_id", "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h"]

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

        for row_index, row in enumerate(reader, start=2):
            stats["rows"] += 1
            try:
                frame = int(float(row["frame_id"]))
                x = float(row["face_bbox_x"])
                y = float(row["face_bbox_y"])
                w = float(row["face_bbox_w"])
                h = float(row["face_bbox_h"])
                score_raw = row.get("face_confidence", "")
                score = float(score_raw) if score_raw not in ("", None) else 1.0
                if w <= 0.0 or h <= 0.0:
                    raise ValueError("non-positive box")
            except (TypeError, ValueError):
                stats["invalid_rows"] += 1
                continue

            stats["valid_rows"] += 1
            boxes.append(Box(frame, x, y, w, h, score, row_index))

    return boxes, stats


def frame_range(boxes_by_frame: dict[int, list[Box]]) -> tuple[int | None, int | None]:
    if not boxes_by_frame:
        return None, None
    frames = boxes_by_frame.keys()
    return min(frames), max(frames)


def count_boxes(boxes_by_frame: dict[int, list[Box]]) -> int:
    return sum(len(boxes) for boxes in boxes_by_frame.values())


def greedy_match_iou(preds: list[Box], refs: list[Box], threshold: float) -> tuple[list[tuple[int, int, float]], int, int]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(preds):
        for ref_index, ref in enumerate(refs):
            score = iou(pred, ref)
            if score >= threshold:
                candidates.append((score, pred_index, ref_index))

    candidates.sort(reverse=True)
    matched_preds: set[int] = set()
    matched_refs: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, pred_index, ref_index in candidates:
        if pred_index in matched_preds or ref_index in matched_refs:
            continue
        matched_preds.add(pred_index)
        matched_refs.add(ref_index)
        matches.append((pred_index, ref_index, score))

    return matches, len(matched_preds), len(matched_refs)


def evaluate_person_detections(
    preds_by_frame: dict[int, list[Box]],
    refs_by_frame: dict[int, list[Box]],
    threshold: float = 0.5,
) -> tuple[dict[str, float | int], dict[int, dict[str, float | int]]]:
    ref_min, ref_max = frame_range(refs_by_frame)
    total_detections = count_boxes(preds_by_frame)
    out_of_reference = 0
    tp = 0
    fp = 0
    fn = 0
    ious: list[float] = []
    per_frame: dict[int, dict[str, float | int]] = {}

    frames = sorted(set(refs_by_frame) | set(preds_by_frame))
    for frame in frames:
        refs = refs_by_frame.get(frame, [])
        preds = preds_by_frame.get(frame, [])
        outside_range = ref_min is None or ref_max is None or frame < ref_min or frame > ref_max
        if outside_range:
            out_of_reference += len(preds)
            per_frame[frame] = {
                "frame": frame,
                "reference_boxes": len(refs),
                "veilsight_detections": len(preds),
                "veilsight_tp": 0,
                "veilsight_fp": 0,
                "veilsight_fn": 0,
            }
            continue

        matches, matched_preds, matched_refs = greedy_match_iou(preds, refs, threshold)
        frame_tp = len(matches)
        frame_fp = len(preds) - matched_preds
        frame_fn = len(refs) - matched_refs
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        ious.extend(match[2] for match in matches)

        per_frame[frame] = {
            "frame": frame,
            "reference_boxes": len(refs),
            "veilsight_detections": len(preds),
            "veilsight_tp": frame_tp,
            "veilsight_fp": frame_fp,
            "veilsight_fn": frame_fn,
        }

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    evaluated_detections = total_detections - out_of_reference

    return (
        {
            "total_detections": total_detections,
            "evaluated_detections": evaluated_detections,
            "out_of_reference_detections": out_of_reference,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": safe_div(sum(ious), len(ious)),
        },
        per_frame,
    )


def evaluate_body_roi_coverage(
    rois_by_frame: dict[int, list[Box]],
    refs_by_frame: dict[int, list[Box]],
) -> tuple[dict[str, float | int], dict[int, dict[str, float | int]]]:
    ref_min, ref_max = frame_range(refs_by_frame)
    if ref_min is None or ref_max is None:
        raise ValueError("reference boxes are empty")

    total_rois = count_boxes(rois_by_frame)
    evaluated_rois = 0
    out_of_reference = 0
    intersecting_rois = 0
    non_intersecting_rois = 0
    covered_public_boxes: set[tuple[int, int]] = set()
    per_frame: dict[int, dict[str, float | int]] = {}

    for frame in sorted(set(refs_by_frame) | set(rois_by_frame)):
        refs = refs_by_frame.get(frame, [])
        rois = rois_by_frame.get(frame, [])
        row = per_frame.setdefault(
            frame,
            {
                "frame": frame,
                "reference_boxes": len(refs),
                "body_rois": len(rois),
                "body_rois_intersecting_public_box": 0,
                "body_rois_not_intersecting_public_box": 0,
                "covered_public_boxes": 0,
            },
        )

        if frame < ref_min or frame > ref_max:
            out_of_reference += len(rois)
            continue

        evaluated_rois += len(rois)
        for roi in rois:
            hit_refs = [index for index, ref in enumerate(refs) if overlaps(roi, ref)]
            if hit_refs:
                intersecting_rois += 1
                row["body_rois_intersecting_public_box"] = int(row["body_rois_intersecting_public_box"]) + 1
                for ref_index in hit_refs:
                    covered_public_boxes.add((frame, ref_index))
            else:
                non_intersecting_rois += 1
                row["body_rois_not_intersecting_public_box"] = int(row["body_rois_not_intersecting_public_box"]) + 1

    for frame, row in per_frame.items():
        row["covered_public_boxes"] = sum(1 for covered_frame, _ in covered_public_boxes if covered_frame == frame)

    total_reference_boxes = count_boxes(refs_by_frame)
    return (
        {
            "total_rois": total_rois,
            "evaluated_rois": evaluated_rois,
            "out_of_reference_rois": out_of_reference,
            "rois_intersecting_public_box": intersecting_rois,
            "rois_not_intersecting_public_box": non_intersecting_rois,
            "roi_hit_rate": safe_div(intersecting_rois, evaluated_rois),
            "covered_public_boxes": len(covered_public_boxes),
            "reference_boxes": total_reference_boxes,
            "public_box_coverage": safe_div(len(covered_public_boxes), total_reference_boxes),
        },
        per_frame,
    )


def make_stretch_mapper(max_friend_frame: int, ref_min: int, ref_max: int) -> Callable[[int], int]:
    span = ref_max - ref_min
    if max_friend_frame <= 0 or span <= 0:
        return lambda _frame: ref_min

    def mapper(friend_frame: int) -> int:
        return math.floor(friend_frame / max_friend_frame * span) + ref_min

    return mapper


def map_fps_frame(friend_frame: int, source_fps: float, target_fps: float, ref_min: int = 1) -> int:
    return math.floor(friend_frame * target_fps / source_fps) + ref_min


def evaluate_face_detections(
    faces: list[Box],
    refs_by_frame: dict[int, list[Box]],
    mapper: Callable[[int], int],
    ref_min: int,
    ref_max: int,
) -> tuple[dict[str, float | int], dict[int, dict[str, float | int]]]:
    total_faces = len(faces)
    evaluated_faces = 0
    out_of_reference = 0
    inside = 0
    outside = 0
    covered_public_boxes: set[tuple[int, int]] = set()
    per_frame: dict[int, dict[str, float | int]] = {}

    for face in faces:
        mapped_frame = mapper(face.frame)
        if mapped_frame < ref_min or mapped_frame > ref_max:
            out_of_reference += 1
            continue

        evaluated_faces += 1
        refs = refs_by_frame.get(mapped_frame, [])
        intersecting = [index for index, ref in enumerate(refs) if overlaps(ref, face)]
        row = per_frame.setdefault(
            mapped_frame,
            {
                "frame": mapped_frame,
                "evaluated_faces": 0,
                "faces_intersecting_public_box": 0,
                "faces_not_intersecting_public_box": 0,
                "covered_public_boxes": 0,
            },
        )
        row["evaluated_faces"] = int(row["evaluated_faces"]) + 1

        if intersecting:
            inside += 1
            row["faces_intersecting_public_box"] = int(row["faces_intersecting_public_box"]) + 1
            for ref_index in intersecting:
                covered_public_boxes.add((mapped_frame, ref_index))
        else:
            outside += 1
            row["faces_not_intersecting_public_box"] = int(row["faces_not_intersecting_public_box"]) + 1

    for frame, row in per_frame.items():
        row["covered_public_boxes"] = sum(1 for covered_frame, _ in covered_public_boxes if covered_frame == frame)

    total_reference_boxes = count_boxes(refs_by_frame)
    return (
        {
            "total_faces": total_faces,
            "evaluated_faces": evaluated_faces,
            "out_of_reference_faces": out_of_reference,
            "faces_intersecting_public_box": inside,
            "faces_not_intersecting_public_box": outside,
            "face_hit_rate": safe_div(inside, evaluated_faces),
            "person_coverage": safe_div(len(covered_public_boxes), total_reference_boxes),
            "covered_public_boxes": len(covered_public_boxes),
            "reference_boxes": total_reference_boxes,
            "hit_rule": "face ROI intersects public body box",
        },
        per_frame,
    )


def load_runtime_summary(runtime_log: Path | None, report_csv: Path | None, configured_fps: float | None = None) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {
        "configured_fps": configured_fps,
        "measured_fps": None,
        "processed_frames": None,
        "time_sec": None,
        "mean_latency_ms": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "latency_source": None,
    }

    if report_csv and report_csv.exists():
        with report_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            for src, dst, cast in [
                ("avg_fps", "measured_fps", float),
                ("processed_frames", "processed_frames", int),
                ("time_sec", "time_sec", float),
            ]:
                raw = row.get(src)
                if raw not in (None, "", "N/A"):
                    try:
                        out[dst] = cast(float(raw)) if cast is int else cast(raw)
                    except ValueError:
                        pass

    latencies: list[float] = []
    if runtime_log and runtime_log.exists():
        with runtime_log.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("latency_ms")
                if raw in (None, "", "N/A"):
                    continue
                try:
                    latencies.append(float(raw))
                except ValueError:
                    continue
        out["latency_source"] = str(runtime_log)

    if latencies:
        out["mean_latency_ms"] = sum(latencies) / len(latencies)
        out["p50_latency_ms"] = percentile(latencies, 0.50)
        out["p95_latency_ms"] = percentile(latencies, 0.95)
        out["p99_latency_ms"] = percentile(latencies, 0.99)

    if out["measured_fps"] is None and out["processed_frames"] and out["time_sec"]:
        out["measured_fps"] = safe_div(float(out["processed_frames"]), float(out["time_sec"]))

    return out


def write_per_frame_csv(
    path: Path,
    refs_by_frame: dict[int, list[Box]],
    person_rows: dict[int, dict[str, float | int]],
    face_rows_by_mapping: dict[str, dict[int, dict[str, float | int]]],
) -> None:
    fieldnames = [
        "mapping",
        "frame",
        "reference_boxes",
        "veilsight_body_rois",
        "veilsight_body_rois_intersecting_public_box",
        "veilsight_body_rois_not_intersecting_public_box",
        "veilsight_covered_public_boxes",
        "evaluated_faces",
        "faces_intersecting_public_box",
        "faces_not_intersecting_public_box",
        "covered_public_boxes",
    ]
    frames = sorted(set(refs_by_frame) | set(person_rows) | {f for rows in face_rows_by_mapping.values() for f in rows})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            person = person_rows.get(frame, {})
            for mapping, face_rows in face_rows_by_mapping.items():
                face = face_rows.get(frame, {})
                writer.writerow(
                    {
                        "mapping": mapping,
                        "frame": frame,
                        "reference_boxes": len(refs_by_frame.get(frame, [])),
                        "veilsight_body_rois": person.get("body_rois", 0),
                        "veilsight_body_rois_intersecting_public_box": person.get(
                            "body_rois_intersecting_public_box", 0
                        ),
                        "veilsight_body_rois_not_intersecting_public_box": person.get(
                            "body_rois_not_intersecting_public_box", 0
                        ),
                        "veilsight_covered_public_boxes": person.get("covered_public_boxes", 0),
                        "evaluated_faces": face.get("evaluated_faces", 0),
                        "faces_intersecting_public_box": face.get("faces_intersecting_public_box", 0),
                        "faces_not_intersecting_public_box": face.get("faces_not_intersecting_public_box", 0),
                        "covered_public_boxes": face.get("covered_public_boxes", 0),
                    }
                )


def print_table(summary: dict) -> None:
    reference = summary["reference"]
    print(
        f"Reference public boxes: {reference['boxes']} "
        f"frames={reference['frames_min']}..{reference['frames_max']}"
    )
    print()
    print(
        f"{'system/mapping':<28} {'roi_total':>9} {'eval':>9} {'hit':>9} "
        f"{'miss':>9} {'roi_hit':>9} {'body_cov':>9} {'fps':>9} {'p95_ms':>10}"
    )
    person = summary["veilsight_body_overlap"]
    person_rt = summary["veilsight_runtime"]
    print(
        f"{'veilsight_body_overlap':<28} {person['total_rois']:>9} {person['evaluated_rois']:>9} "
        f"{person['rois_intersecting_public_box']:>9} {person['rois_not_intersecting_public_box']:>9} "
        f"{person['roi_hit_rate']:>9.4f} {person['public_box_coverage']:>9.4f} "
        f"{format_optional(person_rt.get('measured_fps') or person_rt.get('configured_fps')):>9} "
        f"{format_optional(person_rt.get('p95_latency_ms')):>10}"
    )
    for key in ["friend_stretch_2782_to_429", "friend_fps_25_to_5", "friend_fps_30_to_5"]:
        face = summary[key]
        face_rt = summary["friend_runtime"]
        print(
            f"{key:<28} {face['total_faces']:>9} {face['evaluated_faces']:>9} "
            f"{face['faces_intersecting_public_box']:>9} {face['faces_not_intersecting_public_box']:>9} "
            f"{face['face_hit_rate']:>9.4f} {face['person_coverage']:>9.4f} "
            f"{format_optional(face_rt.get('measured_fps') or face_rt.get('configured_fps')):>9} "
            f"{format_optional(face_rt.get('p95_latency_ms')):>10}"
        )


def format_optional(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_summary(args: argparse.Namespace) -> tuple[dict, dict[int, dict[str, float | int]], dict[str, dict[int, dict[str, float | int]]]]:
    refs_by_frame, ref_load_stats = load_mot_boxes(args.public_det)
    ours_by_frame, ours_load_stats = load_mot_boxes(args.ours)
    faces, face_load_stats = load_face_boxes(args.friend_face_log)

    ref_min, ref_max = frame_range(refs_by_frame)
    if ref_min is None or ref_max is None:
        raise ValueError(f"no valid reference boxes found in {args.public_det}")

    person_detection_summary, _ = evaluate_person_detections(ours_by_frame, refs_by_frame, args.iou_threshold)
    body_overlap_summary, body_overlap_rows = evaluate_body_roi_coverage(ours_by_frame, refs_by_frame)

    max_friend_frame = max((face.frame for face in faces), default=0)
    mappings: dict[str, Callable[[int], int]] = {
        "friend_stretch_2782_to_429": make_stretch_mapper(max_friend_frame, ref_min, ref_max),
        "friend_fps_25_to_5": lambda frame: map_fps_frame(frame, 25.0, 5.0, ref_min),
        "friend_fps_30_to_5": lambda frame: map_fps_frame(frame, 30.0, 5.0, ref_min),
    }

    face_summaries: dict[str, dict[str, float | int]] = {}
    face_rows_by_mapping: dict[str, dict[int, dict[str, float | int]]] = {}
    for name, mapper in mappings.items():
        face_summary, face_rows = evaluate_face_detections(faces, refs_by_frame, mapper, ref_min, ref_max)
        face_summary["mapping_note"] = (
            "headline normalized-time comparison"
            if name == "friend_stretch_2782_to_429"
            else "fps sensitivity check"
        )
        face_summaries[name] = face_summary
        face_rows_by_mapping[name] = face_rows

    summary = {
        "reference": {
            "path": str(args.public_det),
            "frames_min": ref_min,
            "frames_max": ref_max,
            "boxes": count_boxes(refs_by_frame),
            "load_stats": ref_load_stats,
            "note": "Public MOT20 detections used as pseudo-reference, not true MOT ground truth.",
        },
        "veilsight_body_overlap": {
            "path": str(args.ours),
            "definition": "Veilsight body/anonymization ROI counts as covered if it intersects any public MOT20 body box.",
            "load_stats": ours_load_stats,
            **body_overlap_summary,
        },
        "veilsight_person_iou_diagnostic": {
            "path": str(args.ours),
            "iou_threshold": args.iou_threshold,
            "load_stats": ours_load_stats,
            **person_detection_summary,
        },
        "veilsight_runtime": load_runtime_summary(args.ours_runtime_log, None, args.ours_configured_fps),
        "friend_runtime": load_runtime_summary(args.friend_runtime_log, args.friend_report, args.friend_configured_fps),
        "combined_metric_note": {
            "goal": "Extreme-condition comparison: throughput/latency plus body-box privacy coverage.",
            "veilsight_rule": "body ROI intersects public body box",
            "friend_rule": "face ROI intersects public body box",
            "fps_alignment": "friend_stretch_2782_to_429 is the headline mapping; 25->5 and 30->5 are sensitivity checks.",
        },
        "friend_face_input": {
            "path": str(args.friend_face_log),
            "max_friend_frame": max_friend_frame,
            "load_stats": face_load_stats,
            "note": "Face-only detections evaluated by ROI intersection with public person boxes.",
        },
        **face_summaries,
    }
    return summary, body_overlap_rows, face_rows_by_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-det", type=Path, required=True, help="MOT20 public detections file, e.g. det.txt")
    parser.add_argument("--ours", type=Path, required=True, help="Veilsight MOT-format detection result")
    parser.add_argument("--friend-face-log", type=Path, required=True, help="Friend system face_log.csv")
    parser.add_argument("--friend-runtime-log", type=Path, default=None, help="Friend frame_runtime_log.csv")
    parser.add_argument("--friend-report", type=Path, default=None, help="Friend report.csv with avg_fps/time_sec")
    parser.add_argument("--ours-runtime-log", type=Path, default=None, help="Optional Veilsight runtime log for MOT20 run")
    parser.add_argument("--ours-configured-fps", type=float, default=5.0, help="Configured Veilsight FPS if no runtime log exists")
    parser.add_argument("--friend-configured-fps", type=float, default=None, help="Configured friend FPS fallback")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for summary.json and per_frame.csv")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for person boxes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, person_rows, face_rows_by_mapping = build_summary(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    per_frame_path = args.out_dir / "per_frame.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    refs_by_frame, _ = load_mot_boxes(args.public_det)
    write_per_frame_csv(per_frame_path, refs_by_frame, person_rows, face_rows_by_mapping)

    print_table(summary)
    print()
    print(f"Summary written to: {summary_path}")
    print(f"Per-frame CSV written to: {per_frame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
