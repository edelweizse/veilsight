#!/usr/bin/env python3
"""Validate ChokePoint Label Studio annotation exports.

The validator checks every Label Studio JSON file under
``assets/chokepoint/labels`` by default and compares it with the local frame
files and ground-truth XML for the same sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "assets" / "chokepoint"
LABEL_FILE_RE = re.compile(
    r"^labelstudio_tasks_annotations_(?P<sequence>P\d[EL]_S\d_C\d)_chokepoint_format\.json$"
)
IMAGE_NAME_RE = re.compile(r"(?P<sequence>P\d[EL]_S\d_C\d)_(?P<frame>\d{8})\.jpg$")
IMAGE_DIR_RE = re.compile(r"(?P<sequence>P\d[EL]_S\d_C\d)/(?P<frame>\d{8})\.jpg$")
PERSON_ID_RE = re.compile(r"^P\d{3}$")

EXPECTED_CONTROLS = {
    "face_box",
    "person_id",
    "face_visible",
    "face_quality",
    "recognition_eligible",
    "reason",
}
CHOICE_CONTROLS = EXPECTED_CONTROLS - {"face_box"}
BOOLEAN_CHOICES = {"0", "1"}
FACE_QUALITY_CHOICES = {
    "usable",
    "blurred",
    "occluded",
    "too_small",
    "profile",
    "not_visible",
    "unknown",
}
REASON_CHOICES = {
    "usable_face",
    "blurred_face",
    "occluded_face",
    "too_small",
    "profile_face",
    "no_face",
    "not_visible",
    "ambiguous",
    "unknown",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    path: Path
    message: str
    frame_id: int | None = None
    result_id: str | None = None

    def format(self, repo_root: Path) -> str:
        try:
            path = self.path.relative_to(repo_root)
        except ValueError:
            path = self.path
        parts = [self.severity.upper(), str(path)]
        if self.frame_id is not None:
            parts.append(f"frame={self.frame_id}")
        if self.result_id is not None:
            parts.append(f"result={self.result_id}")
        return ": ".join(parts) + f": {self.message}"


@dataclass
class SequenceStats:
    sequence_id: str
    tasks: int = 0
    annotated_frames: int = 0
    face_boxes: int = 0
    empty_frames: int = 0
    person_ids: Counter[str] = field(default_factory=Counter)
    face_quality: Counter[str] = field(default_factory=Counter)
    recognition_eligible: Counter[str] = field(default_factory=Counter)


class Validator:
    def __init__(
        self,
        dataset_root: Path,
        labels_glob: str,
        check_image_size: bool,
        max_issue_samples: int,
        join_by: str,
    ) -> None:
        self.dataset_root = dataset_root
        self.labels_glob = labels_glob
        self.check_image_size = check_image_size
        self.max_issue_samples = max_issue_samples
        self.join_by = join_by
        self.issues: list[Issue] = []
        self.issue_counts: Counter[str] = Counter()
        self.stats: dict[str, SequenceStats] = {}
        self._image_size_cache: dict[Path, tuple[int, int]] = {}

    def error(
        self,
        path: Path,
        message: str,
        frame_id: int | None = None,
        result_id: str | None = None,
    ) -> None:
        self._add_issue("error", path, message, frame_id, result_id)

    def warning(
        self,
        path: Path,
        message: str,
        frame_id: int | None = None,
        result_id: str | None = None,
    ) -> None:
        self._add_issue("warning", path, message, frame_id, result_id)

    def _add_issue(
        self,
        severity: str,
        path: Path,
        message: str,
        frame_id: int | None,
        result_id: str | None,
    ) -> None:
        self.issue_counts[severity] += 1
        if len(self.issues) < self.max_issue_samples:
            self.issues.append(Issue(severity, path, message, frame_id, result_id))

    def validate(self) -> None:
        if not self.dataset_root.exists():
            self.error(self.dataset_root, "dataset root does not exist")
            return

        labels_dir = self.dataset_root / "labels"
        if not labels_dir.exists():
            self.error(labels_dir, "labels directory does not exist")
            return

        label_paths = sorted(labels_dir.glob(self.labels_glob))
        if not label_paths:
            self.error(labels_dir, f"no label files matched {self.labels_glob!r}")
            return

        seen_sequences: set[str] = set()
        for json_path in label_paths:
            sequence_id = self._sequence_from_label_path(json_path)
            if sequence_id is None:
                self.error(json_path, "filename does not match expected ChokePoint Label Studio export pattern")
                continue
            if sequence_id in seen_sequences:
                self.error(json_path, f"duplicate label file for sequence {sequence_id}")
                continue
            seen_sequences.add(sequence_id)
            self._validate_label_file(json_path, sequence_id)

        for frames_dir in sorted(self.dataset_root.glob("P*/P*_C*")):
            if not frames_dir.is_dir():
                continue
            sequence_id = frames_dir.name
            if sequence_id not in seen_sequences:
                self.error(frames_dir, "sequence has frame files but no matching label JSON")

    def _sequence_from_label_path(self, path: Path) -> str | None:
        match = LABEL_FILE_RE.match(path.name)
        return match.group("sequence") if match else None

    def _validate_label_file(self, json_path: Path, sequence_id: str) -> None:
        stats = SequenceStats(sequence_id)
        self.stats[sequence_id] = stats

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            self.error(json_path, f"invalid JSON: {exc}")
            return

        if not isinstance(data, list):
            self.error(json_path, "top-level Label Studio export must be a list of tasks")
            return

        frames_dir = self._frames_dir(sequence_id)
        gt_xml = self.dataset_root / "groundtruth" / f"{sequence_id}.xml"
        image_frames = self._load_image_frames(frames_dir, json_path)
        gt_frames = self._load_gt_frames(gt_xml, json_path)
        gt_frame_order = self._load_gt_frame_order(gt_xml, json_path)
        task_frames: set[int] = set()

        for index, task in enumerate(data):
            stats.tasks += 1
            if not isinstance(task, dict):
                self.error(json_path, f"task at index {index} is not an object")
                continue

            frame_id = self._validate_task_data(json_path, task, sequence_id)
            if frame_id is None:
                continue

            source_frame_id = frame_id
            if self.join_by == "task-index":
                if gt_frame_order is None or index >= len(gt_frame_order):
                    self.error(json_path, "task index has no matching ground-truth XML frame", frame_id)
                    continue
                source_frame_id = gt_frame_order[index]

            if source_frame_id in task_frames:
                self.error(json_path, "duplicate task for source frame", source_frame_id)
            task_frames.add(source_frame_id)

            if image_frames is not None and source_frame_id not in image_frames:
                self.error(json_path, "task source frame has no matching JPEG frame file", source_frame_id)
            if gt_frames is not None and source_frame_id not in gt_frames:
                self.error(json_path, "task source frame has no matching ground-truth XML frame", source_frame_id)

            image_path = frames_dir / f"{source_frame_id:08d}.jpg"
            boxes = self._validate_annotations(json_path, task, image_path, source_frame_id, stats)
            if boxes == 0:
                stats.empty_frames += 1
            else:
                stats.annotated_frames += 1
                stats.face_boxes += boxes

        self._compare_frame_sets(json_path, "JPEG frame files", image_frames, task_frames)
        self._compare_frame_sets(json_path, "ground-truth XML frames", gt_frames, task_frames)

    def _frames_dir(self, sequence_id: str) -> Path:
        session = "_".join(sequence_id.split("_")[:2])
        return self.dataset_root / session / sequence_id

    def _load_image_frames(self, frames_dir: Path, report_path: Path) -> set[int] | None:
        if not frames_dir.exists():
            self.error(report_path, f"frames directory does not exist: {frames_dir}")
            return None
        frames: set[int] = set()
        for image_path in frames_dir.glob("*.jpg"):
            try:
                frames.add(int(image_path.stem))
            except ValueError:
                self.warning(image_path, "JPEG filename stem is not an integer frame id")
        return frames

    def _load_gt_frames(self, gt_xml: Path, report_path: Path) -> set[int] | None:
        if not gt_xml.exists():
            self.error(report_path, f"ground-truth XML does not exist: {gt_xml}")
            return None
        try:
            root = ET.parse(gt_xml).getroot()
        except ET.ParseError as exc:
            self.error(gt_xml, f"invalid XML: {exc}")
            return None

        frames: set[int] = set()
        for frame_elem in root.findall("frame"):
            number = frame_elem.get("number")
            if number is None:
                self.error(gt_xml, "frame element missing number attribute")
                continue
            try:
                frames.add(int(number))
            except ValueError:
                self.error(gt_xml, f"frame number is not an integer: {number!r}")
        return frames

    def _load_gt_frame_order(self, gt_xml: Path, report_path: Path) -> list[int] | None:
        if not gt_xml.exists():
            return None
        try:
            root = ET.parse(gt_xml).getroot()
        except ET.ParseError:
            return None

        frames: list[int] = []
        for frame_elem in root.findall("frame"):
            number = frame_elem.get("number")
            if number is None:
                continue
            try:
                frames.append(int(number))
            except ValueError:
                self.error(gt_xml, f"frame number is not an integer: {number!r}")
        return frames

    def _validate_task_data(self, json_path: Path, task: dict[str, Any], sequence_id: str) -> int | None:
        data = task.get("data")
        if not isinstance(data, dict):
            self.error(json_path, "task missing data object")
            return None

        dataset = data.get("dataset")
        if dataset != "ChokePoint":
            self.error(json_path, f"expected data.dataset='ChokePoint', got {dataset!r}")

        actual_sequence = data.get("sequence_id")
        if actual_sequence != sequence_id:
            self.error(json_path, f"data.sequence_id {actual_sequence!r} does not match filename sequence {sequence_id!r}")

        frame_id = data.get("frame_id")
        if not isinstance(frame_id, int) or frame_id < 0:
            self.error(json_path, f"data.frame_id must be a non-negative integer, got {frame_id!r}")
            return None

        image = data.get("image")
        if not isinstance(image, str) or not image:
            self.error(json_path, "data.image must be a non-empty string", frame_id)
            return frame_id

        image_match = IMAGE_NAME_RE.search(image) or IMAGE_DIR_RE.search(image)
        if image_match is None:
            self.error(
                json_path,
                f"data.image must reference {sequence_id}_########.jpg or {sequence_id}/########.jpg",
                frame_id,
            )
            return frame_id

        image_sequence = image_match.group("sequence")
        image_frame = int(image_match.group("frame"))
        if image_sequence != sequence_id:
            self.error(json_path, f"data.image sequence {image_sequence!r} does not match {sequence_id!r}", frame_id)
        if self.join_by == "frame-id" and image_frame != frame_id:
            self.error(json_path, f"data.image frame {image_frame} does not match data.frame_id {frame_id}", frame_id)

        return frame_id

    def _validate_annotations(
        self,
        json_path: Path,
        task: dict[str, Any],
        image_path: Path,
        frame_id: int,
        stats: SequenceStats,
    ) -> int:
        annotations = task.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            self.error(json_path, "task.annotations must be a non-empty list", frame_id)
            return 0
        if len(annotations) != 1:
            self.warning(json_path, f"expected one annotation object, got {len(annotations)}", frame_id)

        face_box_count = 0
        for annotation in annotations:
            if not isinstance(annotation, dict):
                self.error(json_path, "annotation entry is not an object", frame_id)
                continue
            if annotation.get("was_cancelled") is True:
                self.warning(json_path, "annotation was marked cancelled", frame_id)
            if annotation.get("ground_truth") is not True:
                self.warning(json_path, "annotation is not marked ground_truth=true", frame_id)

            results = annotation.get("result")
            if not isinstance(results, list):
                self.error(json_path, "annotation.result must be a list", frame_id)
                continue
            if not results:
                continue

            grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
            for result in results:
                if not isinstance(result, dict):
                    self.error(json_path, "result entry is not an object", frame_id)
                    continue
                result_id = result.get("id")
                from_name = result.get("from_name")
                if not isinstance(result_id, str) or not result_id:
                    self.error(json_path, "result entry missing non-empty id", frame_id)
                    continue
                if from_name not in EXPECTED_CONTROLS:
                    self.error(json_path, f"unexpected from_name {from_name!r}", frame_id, result_id)
                    continue
                grouped[result_id][from_name].append(result)

            for result_id, controls in grouped.items():
                face_box_count += self._validate_result_group(
                    json_path,
                    image_path,
                    frame_id,
                    result_id,
                    controls,
                    stats,
                )

        return face_box_count

    def _validate_result_group(
        self,
        json_path: Path,
        image_path: Path,
        frame_id: int,
        result_id: str,
        controls: dict[str, list[dict[str, Any]]],
        stats: SequenceStats,
    ) -> int:
        present_controls = set(controls.keys())
        missing = EXPECTED_CONTROLS - present_controls
        extra = present_controls - EXPECTED_CONTROLS
        if missing:
            self.error(json_path, f"annotation group missing controls: {sorted(missing)}", frame_id, result_id)
        if extra:
            self.error(json_path, f"annotation group has unexpected controls: {sorted(extra)}", frame_id, result_id)

        for control_name, entries in controls.items():
            if len(entries) != 1:
                self.error(json_path, f"control {control_name!r} appears {len(entries)} times in one group", frame_id, result_id)

        face_entries = controls.get("face_box", [])
        if not face_entries:
            return 0

        face_entry = face_entries[0]
        if self._validate_face_box(json_path, image_path, frame_id, result_id, face_entry):
            self._validate_choices(json_path, frame_id, result_id, controls, stats)
            return 1
        return 0

    def _validate_face_box(
        self,
        json_path: Path,
        image_path: Path,
        frame_id: int,
        result_id: str,
        result: dict[str, Any],
    ) -> bool:
        if result.get("type") != "rectanglelabels":
            self.error(json_path, f"face_box type must be 'rectanglelabels', got {result.get('type')!r}", frame_id, result_id)

        value = result.get("value")
        if not isinstance(value, dict):
            self.error(json_path, "face_box value must be an object", frame_id, result_id)
            return False

        labels = value.get("rectanglelabels")
        if labels != ["Face"]:
            self.error(json_path, f"face_box rectanglelabels must be ['Face'], got {labels!r}", frame_id, result_id)

        ok = True
        for key in ("x", "y", "width", "height"):
            if not isinstance(value.get(key), (int, float)):
                self.error(json_path, f"face_box value.{key} must be numeric", frame_id, result_id)
                ok = False

        if not ok:
            return False

        x = float(value["x"])
        y = float(value["y"])
        width = float(value["width"])
        height = float(value["height"])

        if width <= 0 or height <= 0:
            self.error(json_path, "face_box width and height must be positive", frame_id, result_id)
            ok = False
        if x < 0 or y < 0 or x > 100 or y > 100:
            self.error(json_path, "face_box x/y must be in percentage range [0, 100]", frame_id, result_id)
            ok = False
        if x + width > 100.0 + 1e-6 or y + height > 100.0 + 1e-6:
            self.error(json_path, "face_box extends beyond image percentage bounds", frame_id, result_id)
            ok = False

        original_width = result.get("original_width")
        original_height = result.get("original_height")
        if not isinstance(original_width, int) or original_width <= 0:
            self.error(json_path, "face_box original_width must be a positive integer", frame_id, result_id)
            ok = False
        if not isinstance(original_height, int) or original_height <= 0:
            self.error(json_path, "face_box original_height must be a positive integer", frame_id, result_id)
            ok = False

        if self.check_image_size and isinstance(original_width, int) and isinstance(original_height, int) and image_path.exists():
            actual_size = self._read_jpeg_size(image_path)
            if actual_size is not None and actual_size != (original_width, original_height):
                self.error(
                    json_path,
                    f"face_box original size {(original_width, original_height)} does not match JPEG size {actual_size}",
                    frame_id,
                    result_id,
                )
                ok = False

        return ok

    def _validate_choices(
        self,
        json_path: Path,
        frame_id: int,
        result_id: str,
        controls: dict[str, list[dict[str, Any]]],
        stats: SequenceStats,
    ) -> None:
        values: dict[str, str] = {}
        for control_name in CHOICE_CONTROLS:
            entries = controls.get(control_name, [])
            if not entries:
                continue
            result = entries[0]
            if result.get("type") != "choices":
                self.error(json_path, f"{control_name} type must be 'choices', got {result.get('type')!r}", frame_id, result_id)
            value = result.get("value")
            if not isinstance(value, dict):
                self.error(json_path, f"{control_name} value must be an object", frame_id, result_id)
                continue
            choices = value.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], str):
                self.error(json_path, f"{control_name} must contain exactly one string choice", frame_id, result_id)
                continue
            values[control_name] = choices[0]

        person_id = values.get("person_id")
        if person_id:
            stats.person_ids[person_id] += 1
            if not PERSON_ID_RE.match(person_id):
                self.error(json_path, f"person_id choice must match P###, got {person_id!r}", frame_id, result_id)

        face_visible = values.get("face_visible")
        if face_visible and face_visible not in BOOLEAN_CHOICES:
            self.error(json_path, f"face_visible must be one of {sorted(BOOLEAN_CHOICES)}, got {face_visible!r}", frame_id, result_id)

        recognition_eligible = values.get("recognition_eligible")
        if recognition_eligible:
            stats.recognition_eligible[recognition_eligible] += 1
            if recognition_eligible not in BOOLEAN_CHOICES:
                self.error(
                    json_path,
                    f"recognition_eligible must be one of {sorted(BOOLEAN_CHOICES)}, got {recognition_eligible!r}",
                    frame_id,
                    result_id,
                )

        face_quality = values.get("face_quality")
        if face_quality:
            stats.face_quality[face_quality] += 1
            if face_quality not in FACE_QUALITY_CHOICES:
                self.warning(json_path, f"unknown face_quality choice {face_quality!r}", frame_id, result_id)

        reason = values.get("reason")
        if reason and reason not in REASON_CHOICES:
            self.warning(json_path, f"unknown reason choice {reason!r}", frame_id, result_id)

        if values.get("face_visible") == "0" and values.get("recognition_eligible") == "1":
            self.error(json_path, "recognition_eligible=1 requires face_visible=1", frame_id, result_id)

    def _compare_frame_sets(
        self,
        json_path: Path,
        source_name: str,
        expected: set[int] | None,
        actual: set[int],
    ) -> None:
        if expected is None:
            return
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            self.error(json_path, f"missing {len(missing)} task frames present in {source_name}; first: {missing[:10]}")
        if extra:
            self.error(json_path, f"{len(extra)} task frames are absent from {source_name}; first: {extra[:10]}")

    def _read_jpeg_size(self, path: Path) -> tuple[int, int] | None:
        if path in self._image_size_cache:
            return self._image_size_cache[path]

        try:
            with path.open("rb") as f:
                if f.read(2) != b"\xff\xd8":
                    self.error(path, "file is not a JPEG")
                    return None
                while True:
                    marker_prefix = f.read(1)
                    if not marker_prefix:
                        break
                    if marker_prefix != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3"}:
                        _length = struct.unpack(">H", f.read(2))[0]
                        _precision = f.read(1)
                        height, width = struct.unpack(">HH", f.read(4))
                        size = (width, height)
                        self._image_size_cache[path] = size
                        return size
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    length_bytes = f.read(2)
                    if len(length_bytes) != 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]
                    if length < 2:
                        break
                    f.seek(length - 2, 1)
        except OSError as exc:
            self.error(path, f"failed to read JPEG size: {exc}")
            return None

        self.error(path, "could not find JPEG size marker")
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "errors": self.issue_counts["error"],
            "warnings": self.issue_counts["warning"],
            "sequences": {
                sequence_id: {
                    "tasks": stats.tasks,
                    "annotated_frames": stats.annotated_frames,
                    "empty_frames": stats.empty_frames,
                    "face_boxes": stats.face_boxes,
                    "person_ids": dict(stats.person_ids),
                    "face_quality": dict(stats.face_quality),
                    "recognition_eligible": dict(stats.recognition_eligible),
                }
                for sequence_id, stats in sorted(self.stats.items())
            },
        }


def load_groundtruth_people(xml_path: Path) -> list[tuple[int, list[str]]]:
    """Return XML frames in file order with their ground-truth person ids."""
    root = ET.parse(xml_path).getroot()
    frames: list[tuple[int, list[str]]] = []
    for frame_elem in root.findall("frame"):
        number = frame_elem.get("number")
        if number is None:
            continue
        person_ids = [
            person_id
            for person in frame_elem.findall("person")
            if (person_id := person.get("id"))
        ]
        frames.append((int(number), person_ids))
    return frames


def image_frame_from_task(task: dict[str, Any]) -> int | None:
    data = task.get("data")
    if not isinstance(data, dict):
        return None
    image = data.get("image")
    if not isinstance(image, str):
        return None
    image_match = IMAGE_NAME_RE.search(image) or IMAGE_DIR_RE.search(image)
    if image_match is None:
        return None
    return int(image_match.group("frame"))


def grouped_results(task: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    annotations = task.get("annotations", [])
    if not isinstance(annotations, list):
        return grouped
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        results = annotation.get("result", [])
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            result_id = result.get("id")
            from_name = result.get("from_name")
            if isinstance(result_id, str) and isinstance(from_name, str):
                grouped[result_id][from_name].append(result)
    return grouped


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def iter_bbox_id_rows(dataset_root: Path, labels_glob: str, join_by: str) -> list[dict[str, Any]]:
    """Build rows with bbox fields from Label Studio and ids from groundtruth XML.

    ``task-index`` join is the practical default for the current ChokePoint
    exports because the label task ids are dense, while XML/JPEG frame ids are
    sparse original frame ids.
    """
    labels_dir = dataset_root / "labels"
    rows: list[dict[str, Any]] = []

    for json_path in sorted(labels_dir.glob(labels_glob)):
        match = LABEL_FILE_RE.match(json_path.name)
        if match is None:
            continue
        sequence_id = match.group("sequence")
        gt_path = dataset_root / "groundtruth" / f"{sequence_id}.xml"
        gt_frames = load_groundtruth_people(gt_path)
        gt_by_frame = {frame_id: person_ids for frame_id, person_ids in gt_frames}

        with json_path.open("r", encoding="utf-8") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            continue

        for task_index, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            data = task.get("data", {})
            if not isinstance(data, dict):
                data = {}
            label_frame_id = data.get("frame_id")
            image = data.get("image", "")
            image_frame_id = image_frame_from_task(task)

            if join_by == "frame-id" and isinstance(label_frame_id, int):
                gt_frame_id = label_frame_id
                person_ids = gt_by_frame.get(label_frame_id, [])
            else:
                gt_frame_id, person_ids = gt_frames[task_index] if task_index < len(gt_frames) else (None, [])

            person_id = person_ids[0] if len(person_ids) == 1 else "|".join(person_ids)

            for result_id, controls in grouped_results(task).items():
                face_entries = controls.get("face_box", [])
                if not face_entries:
                    continue
                face_entry = face_entries[0]
                value = face_entry.get("value", {})
                if not isinstance(value, dict):
                    continue
                if not all(isinstance(value.get(key), (int, float)) for key in ("x", "y", "width", "height")):
                    continue

                original_width = face_entry.get("original_width", 0)
                original_height = face_entry.get("original_height", 0)
                if not isinstance(original_width, int) or not isinstance(original_height, int):
                    original_width = 0
                    original_height = 0

                x_pct = float(value["x"])
                y_pct = float(value["y"])
                width_pct = float(value["width"])
                height_pct = float(value["height"])

                source_frame_id = gt_frame_id if isinstance(gt_frame_id, int) else image_frame_id
                source_image_path = ""
                if isinstance(source_frame_id, int):
                    session = "_".join(sequence_id.split("_")[:2])
                    source_image_path = display_path(dataset_root / session / sequence_id / f"{source_frame_id:08d}.jpg")

                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "label_file": display_path(json_path),
                        "task_index": task_index,
                        "label_frame_id": label_frame_id,
                        "image_frame_id": image_frame_id,
                        "gt_frame_id": gt_frame_id,
                        "source_frame_id": source_frame_id,
                        "source_image_path": source_image_path,
                        "person_id": person_id,
                        "person_ids": "|".join(person_ids),
                        "result_id": result_id,
                        "bbox_x": x_pct / 100.0 * original_width,
                        "bbox_y": y_pct / 100.0 * original_height,
                        "bbox_width": width_pct / 100.0 * original_width,
                        "bbox_height": height_pct / 100.0 * original_height,
                        "bbox_x_pct": x_pct,
                        "bbox_y_pct": y_pct,
                        "bbox_width_pct": width_pct,
                        "bbox_height_pct": height_pct,
                        "original_width": original_width,
                        "original_height": original_height,
                        "image": image,
                    }
                )

    return rows


def write_rows(rows: list[dict[str, Any]], output_format: str, output_path: Path | None) -> None:
    if output_format == "jsonl":
        lines = [json.dumps(row, sort_keys=True) for row in rows]
        payload = "\n".join(lines) + ("\n" if lines else "")
        if output_path is None:
            print(payload, end="")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        return

    fieldnames = [
        "sequence_id",
        "task_index",
        "label_frame_id",
        "image_frame_id",
        "gt_frame_id",
        "source_frame_id",
        "person_id",
        "person_ids",
        "result_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "bbox_x_pct",
        "bbox_y_pct",
        "bbox_width_pct",
        "bbox_height_pct",
        "original_width",
        "original_height",
        "source_image_path",
        "label_file",
        "image",
    ]
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def render_labeled_images(
    dataset_root: Path,
    labels_glob: str,
    sequence_id: str,
    join_by: str,
    output_dir: Path,
    limit: int | None,
) -> int:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Rendering labeled images requires Pillow: python -m pip install Pillow") from exc

    rows = [
        row
        for row in iter_bbox_id_rows(dataset_root, labels_glob, join_by)
        if row["sequence_id"] == sequence_id
    ]
    if not rows:
        raise SystemExit(f"No bbox rows found for sequence {sequence_id!r}")

    by_image: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_path = row.get("source_image_path")
        if not source_path:
            continue
        by_image[REPO_ROOT / source_path].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    rendered = 0
    for image_path, image_rows in sorted(by_image.items(), key=lambda item: item[0].name):
        if limit is not None and rendered >= limit:
            break
        if not image_path.exists():
            continue

        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for row in image_rows:
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_width"])
                h = float(row["bbox_height"])
                person_id = str(row["person_id"] or "no_id")
                label = f"id={person_id}"
                color = (0, 220, 90) if row["person_id"] else (255, 170, 0)

                draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                label_y = max(0, y - text_h - 6)
                draw.rectangle((x, label_y, x + text_w + 8, label_y + text_h + 6), fill=color)
                draw.text((x + 4, label_y + 3), label, fill=(0, 0, 0), font=font)

            out_path = output_dir / f"{sequence_id}_{image_path.stem}_labeled.jpg"
            image.save(out_path, quality=95)
            rendered += 1

    if rendered == 0:
        sample_dir = dataset_root / "_".join(sequence_id.split("_")[:2]) / sequence_id
        raise SystemExit(f"No images rendered. Checked source frames under {sample_dir}")

    metadata_path = output_dir / f"{sequence_id}_labels.csv"
    sequence_rows = [row for row in rows if row.get("source_image_path")]
    write_rows(sequence_rows, "csv", metadata_path)
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all ChokePoint Label Studio label JSON files against local frames and ground-truth XML.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"ChokePoint dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--labels-glob",
        default="labelstudio_tasks_annotations_*_chokepoint_format.json",
        help="Glob to match label files inside DATASET_ROOT/labels.",
    )
    parser.add_argument(
        "--skip-image-size",
        action="store_true",
        help="Skip comparing Label Studio original_width/original_height with JPEG dimensions.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=50,
        help="Maximum issue samples to print (counts still include all issues).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary JSON.",
    )
    parser.add_argument(
        "--export-bboxes",
        action="store_true",
        help="Output bbox rows from Label Studio labels with person_id from ground-truth XML instead of validation summary.",
    )
    parser.add_argument(
        "--render-labeled-images",
        action="store_true",
        help="Render JPEG copies with Label Studio bboxes and ground-truth person ids drawn on the image.",
    )
    parser.add_argument(
        "--sequence",
        help="Sequence id to render, for example P1E_S2_C1. Required with --render-labeled-images.",
    )
    parser.add_argument(
        "--join-by",
        choices=("task-index", "frame-id"),
        default="task-index",
        help="How to join labels to ground-truth XML for export mode (default: task-index).",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Export format for --export-bboxes (default: csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for --export-bboxes, or output directory for --render-labeled-images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of labeled images to render.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_labeled_images:
        if not args.sequence:
            raise SystemExit("--sequence is required with --render-labeled-images")
        output_dir = args.output or (REPO_ROOT / "results" / "chokepoint" / "labeled_images" / args.sequence)
        rendered = render_labeled_images(
            dataset_root=args.dataset_root.resolve(),
            labels_glob=args.labels_glob,
            sequence_id=args.sequence,
            join_by=args.join_by,
            output_dir=output_dir,
            limit=args.limit,
        )
        print(f"Rendered {rendered} labeled image(s) to {output_dir}")
        return 0

    if args.export_bboxes:
        rows = iter_bbox_id_rows(
            dataset_root=args.dataset_root.resolve(),
            labels_glob=args.labels_glob,
            join_by=args.join_by,
        )
        write_rows(rows, args.format, args.output)
        return 0

    validator = Validator(
        dataset_root=args.dataset_root.resolve(),
        labels_glob=args.labels_glob,
        check_image_size=not args.skip_image_size,
        max_issue_samples=max(args.max_issues, 0),
        join_by=args.join_by,
    )
    validator.validate()

    summary = validator.summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Validated dataset root: {summary['dataset_root']}")
        print(f"Errors: {summary['errors']}")
        print(f"Warnings: {summary['warnings']}")
        for sequence_id, stats in summary["sequences"].items():
            print(
                f"{sequence_id}: tasks={stats['tasks']} annotated_frames={stats['annotated_frames']} "
                f"empty_frames={stats['empty_frames']} face_boxes={stats['face_boxes']}"
            )
        if validator.issues:
            print()
            print(f"First {len(validator.issues)} issue(s):")
            for issue in validator.issues:
                print(issue.format(REPO_ROOT))

    return 1 if validator.issue_counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
