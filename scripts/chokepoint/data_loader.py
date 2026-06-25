# ChokePoint data loader — parses groundtruth XML and labels JSON
"""Data loading for ChokePoint evaluation.

Parses groundtruth XML (person IDs per frame) and labels JSON (face boxes per frame),
correlating them by frame number.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class FaceBox:
    """A face bounding box from labels JSON, converted to pixel coordinates."""

    x: float  # left edge in pixels
    y: float  # top edge in pixels
    width: float  # width in pixels
    height: float  # height in pixels
    face_visible: bool = False
    face_quality: str = ""
    recognition_eligible: bool = False
    reason: str = ""


@dataclass
class GtFace:
    """A ground-truth face observation with person ID and bounding box."""

    person_id: str
    bbox: FaceBox | None = None  # None if no face box in labels for this frame
    face_visible: bool = False
    quality: str = ""
    recognition_eligible: bool = False
    reason: str = ""


def load_groundtruth(xml_path: Path) -> dict[int, list[str]]:
    """Load groundtruth person IDs from ChokePoint XML.

    Returns dict mapping frame_id (int) -> list of person_id strings.
    Frame IDs are parsed from the ``number`` attribute of ``<frame>`` elements
    as zero-padded 8-digit strings and converted to int.
    """
    gt: dict[int, list[str]] = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for frame_elem in root.findall("frame"):
        frame_id = int(frame_elem.get("number", "0"))
        person_ids: list[str] = []
        for person_elem in frame_elem.findall("person"):
            pid = person_elem.get("id")
            if pid:
                person_ids.append(pid)
        gt[frame_id] = person_ids
    return gt


LabelJoinMode = Literal["frame-id", "task-index"]


def load_labels(json_path: Path, frame_id_map: list[int] | None = None) -> dict[int, list[FaceBox]]:
    """Load face bounding boxes from LabelStudio labels JSON.

    Returns dict mapping frame_id (int) -> list of FaceBox objects.
    Coordinates are converted from percentages (0-100) to pixels using
    original_width/original_height from the annotation metadata.

    If ``frame_id_map`` is provided, task index ``i`` is keyed by
    ``frame_id_map[i]``. The current ChokePoint Label Studio exports use dense
    task frame IDs, while local JPEG/XML files use sparse original frame IDs.
    """
    labels: dict[int, list[FaceBox]] = {}
    with open(json_path) as f:
        data = json.load(f)
    for task_index, item in enumerate(data):
        if frame_id_map is not None:
            if task_index >= len(frame_id_map):
                continue
            frame_id = frame_id_map[task_index]
        else:
            frame_id = item["data"]["frame_id"]
        boxes: list[FaceBox] = []
        for ann in item.get("annotations", []):
            face_box_data: dict | None = None
            _person_id_val: str = ""
            face_visible_val: bool = False
            face_quality_val: str = ""
            rec_eligible_val: bool = False
            reason_val: str = ""
            orig_w: int = 800
            orig_h: int = 600

            for r in ann.get("result", []):
                name = r.get("from_name", "")
                value = r.get("value", {})
                if name == "face_box":
                    face_box_data = value
                    orig_w = r.get("original_width", 800)
                    orig_h = r.get("original_height", 600)
                elif name == "person_id":
                    choices = value.get("choices", [])
                    _person_id_val = choices[0] if choices else ""
                elif name == "face_visible":
                    choices = value.get("choices", [])
                    face_visible_val = choices[0] == "1" if choices else False
                elif name == "face_quality":
                    choices = value.get("choices", [])
                    face_quality_val = choices[0] if choices else ""
                elif name == "recognition_eligible":
                    choices = value.get("choices", [])
                    rec_eligible_val = choices[0] == "1" if choices else False
                elif name == "reason":
                    choices = value.get("choices", [])
                    reason_val = choices[0] if choices else ""

            if face_box_data is not None:
                boxes.append(
                    FaceBox(
                        x=face_box_data["x"] / 100.0 * orig_w,
                        y=face_box_data["y"] / 100.0 * orig_h,
                        width=face_box_data["width"] / 100.0 * orig_w,
                        height=face_box_data["height"] / 100.0 * orig_h,
                        face_visible=face_visible_val,
                        face_quality=face_quality_val,
                        recognition_eligible=rec_eligible_val,
                        reason=reason_val,
                    )
                )
        labels[frame_id] = boxes
    return labels


def get_groundtruth_faces(
    gt_xml_path: Path,
    labels_json_path: Path,
    label_join: LabelJoinMode = "task-index",
) -> dict[int, list[GtFace]]:
    """Correlate groundtruth person IDs with labels face boxes.

    Returns dict mapping frame_id (int) -> list of GtFace objects.
    Person IDs come from groundtruth XML. Face boxes come from labels JSON.
    Correlated by ordered task index by default because the current
    Label Studio exports use dense task IDs, while ChokePoint XML/JPEG frame
    IDs are sparse original frame IDs. Use ``label_join="frame-id"`` only for
    label exports whose ``data.frame_id`` already equals the XML/JPEG frame ID.

    Edge cases:
    - Frame has groundtruth person but no labels face boxes → GtFace with bbox=None
    - Frame has multiple face boxes but 1 groundtruth person → all boxes get same person ID
    - Frame has no groundtruth annotations → empty list
    """
    gt = load_groundtruth(gt_xml_path)
    frame_id_map = list(gt.keys()) if label_join == "task-index" else None
    labels = load_labels(labels_json_path, frame_id_map=frame_id_map)
    result: dict[int, list[GtFace]] = {}

    for frame_id, person_ids in gt.items():
        frame_boxes = labels.get(frame_id, [])
        faces: list[GtFace] = []
        if not person_ids:
            continue
        pid = person_ids[0]  # ChokePoint has 1 person per annotated frame
        if not frame_boxes:
            # Person present but no face box in labels → no_face case
            faces.append(
                GtFace(
                    person_id=pid,
                    bbox=None,
                )
            )
        else:
            for box in frame_boxes:
                faces.append(
                    GtFace(
                        person_id=pid,
                        bbox=box,
                        face_visible=box.face_visible,
                        quality=box.face_quality,
                        recognition_eligible=box.recognition_eligible,
                        reason=box.reason,
                    )
                )
        result[frame_id] = faces

    return result


def get_all_person_ids(xml_path: Path) -> list[str]:
    """Get all unique person IDs from a groundtruth XML file, sorted."""
    gt = load_groundtruth(xml_path)
    ids: set[str] = set()
    for person_list in gt.values():
        for pid in person_list:
            ids.add(pid)
    return sorted(ids)
