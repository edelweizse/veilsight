"""ChokePoint ROI coverage and attack resistance metrics.

Implements formulas from the methodology chapter:
  Face RCR, Effective Face RCR, Face PER, FRASR, FDA.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Roi:
    x: float
    y: float
    w: float
    h: float


@dataclass
class AttackRecord:
    frame_id: int
    face_bbox: _Roi
    predicted_identity: str
    identity_confidence: float
    recognized: bool


def _coverage_union(rois: list[_Roi], gt: _Roi) -> float:
    clipped: list[tuple[float, float, float, float]] = []
    gt_x2 = gt.x + gt.w
    gt_y2 = gt.y + gt.h
    for roi in rois:
        x1 = max(roi.x, gt.x)
        y1 = max(roi.y, gt.y)
        x2 = min(roi.x + roi.w, gt_x2)
        y2 = min(roi.y + roi.h, gt_y2)
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
    area_gt = max(0.0, gt.w) * max(0.0, gt.h)
    if area_gt <= 0.0 or not clipped:
        return 0.0

    xs = sorted({gt.x, gt_x2, *(x for rect in clipped for x in (rect[0], rect[2]))})
    ys = sorted({gt.y, gt_y2, *(y for rect in clipped for y in (rect[1], rect[3]))})
    covered_area = 0.0
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            x1, x2 = xs[xi], xs[xi + 1]
            y1, y2 = ys[yi], ys[yi + 1]
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            if any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for rx1, ry1, rx2, ry2 in clipped):
                covered_area += (x2 - x1) * (y2 - y1)
    return min(1.0, covered_area / area_gt)


def face_region_coverage_ratio(
    gt_faces_frame: dict[int, list[_Roi]],
    face_anon_regions_frame: dict[int, list[_Roi]],
) -> float:
    total_coverage = 0.0
    total_faces = 0
    for fid, gt_faces in gt_faces_frame.items():
        face_regions = face_anon_regions_frame.get(fid, [])
        for gt in gt_faces:
            total_coverage += _coverage_union(face_regions, gt)
            total_faces += 1
    if total_faces == 0:
        return 0.0
    return total_coverage / total_faces


def effective_face_region_coverage_ratio(
    gt_faces_frame: dict[int, list[_Roi]],
    face_anon_regions_frame: dict[int, list[_Roi]],
    body_anon_regions_frame: dict[int, list[_Roi]],
    frame_anonymized_frames: set[int],
) -> float:
    total_coverage = 0.0
    total_faces = 0
    for fid, gt_faces in gt_faces_frame.items():
        all_regions = list(face_anon_regions_frame.get(fid, []))
        all_regions.extend(body_anon_regions_frame.get(fid, []))
        if fid in frame_anonymized_frames:
            for gt in gt_faces:
                total_coverage += 1.0
                total_faces += 1
            continue
        for gt in gt_faces:
            total_coverage += _coverage_union(all_regions, gt)
            total_faces += 1
    if total_faces == 0:
        return 0.0
    return total_coverage / total_faces


def face_protection_error_rate(
    gt_faces_frame: dict[int, list[_Roi]],
    face_anon_regions_frame: dict[int, list[_Roi]],
    body_anon_regions_frame: dict[int, list[_Roi]],
    tau: float = 0.8,
    frame_anonymized_frames: set[int] | None = None,
) -> float:
    full_frame_ids = frame_anonymized_frames or set()
    total_faces = 0
    exposed_faces = 0
    for fid, gt_faces in gt_faces_frame.items():
        all_regions = list(face_anon_regions_frame.get(fid, []))
        all_regions.extend(body_anon_regions_frame.get(fid, []))
        for gt in gt_faces:
            total_faces += 1
            effective_coverage = 1.0 if fid in full_frame_ids else _coverage_union(all_regions, gt)
            if effective_coverage < tau:
                exposed_faces += 1
    if total_faces == 0:
        return 0.0
    return exposed_faces / total_faces


def _iou(a: _Roi, b: _Roi) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h
    xx1 = max(a.x, b.x)
    yy1 = max(a.y, b.y)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    area_a = max(0.0, a.w) * max(0.0, a.h)
    area_b = max(0.0, b.w) * max(0.0, b.h)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def face_reidentification_attack_success_rate(
    gt_faces_frame: dict[int, list[dict]],
    identity_probe_ids: set[str],
    attack_records: list[AttackRecord],
    iou_threshold: float = 0.5,
) -> float:
    total_probes = 0
    successful = 0
    for rec in attack_records:
        gt_faces = gt_faces_frame.get(rec.frame_id, [])
        for gt in gt_faces:
            pid = gt.get("person_id", "")
            if pid not in identity_probe_ids:
                continue
            gt_box = gt.get("bbox")
            if gt_box is None:
                continue
            gt_roi = _Roi(gt_box.x, gt_box.y, gt_box.width, gt_box.height)
            if _iou(rec.face_bbox, gt_roi) < iou_threshold:
                continue
            total_probes += 1
            if rec.recognized and rec.predicted_identity == pid:
                successful += 1
            break
    if total_probes == 0:
        return 0.0
    return successful / total_probes


def face_detection_after_anonymization(
    attack_records: list[AttackRecord],
    pre_anon_detection_count: int,
) -> float:
    if pre_anon_detection_count == 0:
        return 0.0
    return len(attack_records) / pre_anon_detection_count
