"""ChokePoint detection and protection metrics.

Implements formulas from the methodology chapter:
  Face Detection Recall, Non-Gallery Face Anonymization Recall,
  Face False Allow Rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Bbox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class FaceDecision:
    frame_id: int
    bbox: _Bbox | None
    recognition_state: str
    privacy_action: str


def _iou(a: _Bbox, b: _Bbox) -> float:
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


def _coverage(roi: _Bbox, gt: _Bbox) -> float:
    ax2 = roi.x + roi.w
    ay2 = roi.y + roi.h
    bx2 = gt.x + gt.w
    by2 = gt.y + gt.h
    xx1 = max(roi.x, gt.x)
    yy1 = max(roi.y, gt.y)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    area_gt = max(0.0, gt.w) * max(0.0, gt.h)
    if area_gt <= 0.0:
        return 0.0
    return inter / area_gt


def face_detection_recall(
    gt_bboxes_frame: dict[int, list[_Bbox]],
    detected_bboxes_frame: dict[int, list[_Bbox]],
    iou_threshold: float = 0.5,
) -> float:
    total_gt = 0
    matched = 0
    for fid, gt_boxes in gt_bboxes_frame.items():
        det_boxes = detected_bboxes_frame.get(fid, [])
        total_gt += len(gt_boxes)
        for gt_box in gt_boxes:
            for det_box in det_boxes:
                if _iou(gt_box, det_box) >= iou_threshold:
                    matched += 1
                    break
    if total_gt == 0:
        return 0.0
    return matched / total_gt


def _center_distance(a: _Bbox, b: _Bbox) -> float:
    cx_a = a.x + a.w * 0.5
    cy_a = a.y + a.h * 0.5
    cx_b = b.x + b.w * 0.5
    cy_b = b.y + b.h * 0.5
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def face_detection_recall_center(
    gt_bboxes_frame: dict[int, list[_Bbox]],
    detected_bboxes_frame: dict[int, list[_Bbox]],
    max_center_offset: float = 0.5,
) -> float:
    total_gt = 0
    matched = 0
    for fid, gt_boxes in gt_bboxes_frame.items():
        det_boxes = detected_bboxes_frame.get(fid, [])
        total_gt += len(gt_boxes)
        for gt_box in gt_boxes:
            gt_diag = (gt_box.w ** 2 + gt_box.h ** 2) ** 0.5
            if gt_diag <= 0:
                continue
            for det_box in det_boxes:
                dist = _center_distance(gt_box, det_box)
                if dist / gt_diag <= max_center_offset:
                    matched += 1
                    break
    if total_gt == 0:
        return 0.0
    return matched / total_gt


def non_gallery_face_anonymization_recall(
    non_gallery_faces_frame: dict[int, list[_Bbox]],
    anon_regions_frame: dict[int, list[_Bbox]],
    tau: float = 0.8,
) -> float:
    total_non_gallery = 0
    effectively_anon = 0
    for fid, faces in non_gallery_faces_frame.items():
        regions = anon_regions_frame.get(fid, [])
        total_non_gallery += len(faces)
        for face in faces:
            covered = any(_coverage(r, face) >= tau for r in regions)
            if covered:
                effectively_anon += 1
    if total_non_gallery == 0:
        return 0.0
    return effectively_anon / total_non_gallery


def face_false_allow_rate(
    non_gallery_faces_frame: dict[int, list[_Bbox]],
    face_decisions_frame: dict[int, list[FaceDecision]],
    anon_regions_frame: dict[int, list[_Bbox]],
    tau: float = 0.8,
    iou_threshold: float = 0.5,
) -> float:
    total_non_gallery = 0
    incorrectly_allowed = 0
    for fid, faces in non_gallery_faces_frame.items():
        decisions = face_decisions_frame.get(fid, [])
        regions = anon_regions_frame.get(fid, [])
        total_non_gallery += len(faces)
        for face in faces:
            raw_visible = not any(_coverage(r, face) >= tau for r in regions)
            if not raw_visible:
                continue
            allowed = any(
                decision.bbox is not None
                and decision.privacy_action == "allow"
                and _iou(decision.bbox, face) >= iou_threshold
                for decision in decisions
            )
            if allowed:
                incorrectly_allowed += 1
    if total_non_gallery == 0:
        return 0.0
    return incorrectly_allowed / total_non_gallery
