#!/usr/bin/env python3
"""Run the AyuGram Streamlit anonymizer on ChokePoint and compute Veilsight metrics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.chokepoint.data_loader import get_all_person_ids
from scripts.run_chokepoint_eval import compute_metrics_for_sequence


class _Progress:
    def progress(self, _value: float) -> None:
        return None


class _Status:
    def text(self, value: str) -> None:
        print(f"    {value}", flush=True)


class _BlurSwapper:
    def __init__(self, module) -> None:
        self._module = module

    def get(self, frame, face, _donor_face, paste_back: bool = True):
        return self._module.blur_face_region(frame, face.bbox)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ayugram_chokepoint_app", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import AyuGram app from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_enrolled_people(person_ids: list[str], ratio: float) -> list[str]:
    return person_ids[:max(1, int(len(person_ids) * ratio))]


def _load_base_runtime_params(config: dict[str, Any]) -> dict[str, Any]:
    base_config_path = REPO_ROOT / config.get("paths", {}).get("base_config", "configs/full_reference.yaml")
    with open(base_config_path) as f:
        base = yaml.safe_load(f) or {}

    modules = base.get("modules", {})
    face_detector = modules.get("face_detector", {})
    recognizer = modules.get("recognizer", {})
    detector_type = face_detector.get("type", "scrfd")
    detector_cfg = face_detector.get(detector_type, {})

    return {
        "base_config": str(base_config_path),
        "det_size": (
            int(detector_cfg.get("input_w", 640)),
            int(detector_cfg.get("input_h", 640)),
        ),
        "det_thresh": float(detector_cfg.get("score_threshold", 0.35)),
        "min_face_size": int(max(0.0, float(recognizer.get("min_face_size_px", 0.0)))),
        "recognizer_min_face_score": float(recognizer.get("min_face_score", 0.0)),
        "min_inter_eye_px": float(recognizer.get("min_inter_eye_px", 0.0)),
        "max_roll_deg": float(recognizer.get("max_roll_deg", 180.0)),
        "max_yaw_offset_ratio": float(recognizer.get("max_yaw_offset_ratio", 1.0)),
    }


def _frame_paths(frames_dir: Path, source_fps: float, target_fps: float) -> list[Path]:
    frames = sorted(p for p in frames_dir.glob("*.jpg") if p.stem.isdigit())
    stride = max(1, round(source_fps / target_fps))
    return frames[::stride]


def _write_sampled_video(frame_paths: list[Path], output_path: Path, fps: float) -> dict[int, int]:
    if not frame_paths:
        raise ValueError("No frames to write")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {frame_paths[0]}")
    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")
    dense_to_original: dict[int, int] = {}
    try:
        for dense_id, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Cannot read {frame_path}")
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            dense_to_original[dense_id] = int(frame_path.stem)
    finally:
        writer.release()
    return dense_to_original


def _read_images(paths: list[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def _collect_identity_images(faces_dir: Path, person_id: str) -> list[Path]:
    person_dir = faces_dir / person_id
    return sorted(p for p in person_dir.glob("*") if p.suffix.lower() in {".pgm", ".jpg", ".jpeg", ".png"})


def _build_identity_embeddings(module, face_app, faces_dir: Path, person_ids: list[str]) -> dict[str, list[np.ndarray]]:
    by_id: dict[str, list[np.ndarray]] = {}
    recognition_model = face_app.models.get("recognition")
    if recognition_model is None:
        raise RuntimeError("AyuGram FaceAnalysis did not load a recognition model")
    for person_id in person_ids:
        images = _read_images(_collect_identity_images(faces_dir, person_id))
        embeddings: list[np.ndarray] = []
        if images:
            # ChokePoint PGM chips are already cropped face images. Running full
            # detector+landmark alignment for every gallery chip is prohibitively
            # slow on CPU, so use the app's ArcFace model directly on every chip.
            features = recognition_model.get_feat(images)
            embeddings = [module.normalize_embedding(feat.copy()) for feat in features]
        by_id[person_id] = embeddings
        print(f"  {person_id}: {len(embeddings)} AyuGram embeddings", flush=True)
    return by_id


def _flatten_embeddings(by_id: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for embeddings in by_id.values():
        out.extend(embeddings)
    return out


def _choose_donor_image(faces_dir: Path, enrolled_ids: set[str]) -> np.ndarray:
    for person_dir in sorted(p for p in faces_dir.iterdir() if p.is_dir() and p.name not in enrolled_ids):
        images = _read_images(_collect_identity_images(faces_dir, person_dir.name)[:1])
        if images:
            return images[0]
    for person_dir in sorted(p for p in faces_dir.iterdir() if p.is_dir()):
        images = _read_images(_collect_identity_images(faces_dir, person_dir.name)[:1])
        if images:
            return images[0]
    raise RuntimeError(f"No donor image found under {faces_dir}")


def _remap_csv_frame_ids(path: Path, dense_to_original: dict[int, int], output_frame_dir: Path | None = None) -> None:
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        dense = int(float(row.get("frame_id", 0)))
        original = dense_to_original.get(dense, dense)
        row["frame_id"] = str(original)
        if output_frame_dir and "output_frame_path" in row:
            row["output_frame_path"] = str(output_frame_dir / f"{original:08d}.jpg")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extract_output_frames(video_path: Path, dense_to_original: dict[int, int], frame_out_dir: Path) -> None:
    if frame_out_dir.exists():
        shutil.rmtree(frame_out_dir)
    frame_out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open AyuGram output video {video_path}")
    dense = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if dense in dense_to_original:
                original = dense_to_original[dense]
                cv2.imwrite(str(frame_out_dir / f"{original:08d}.jpg"), frame)
            dense += 1
    finally:
        cap.release()


def _write_masks_from_anon_log(anon_log: Path, mask_dir: Path) -> None:
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if not anon_log.exists():
        return
    with open(anon_log, newline="") as f:
        for row in csv.DictReader(f):
            fid = int(float(row.get("frame_id", 0)))
            x = int(round(float(row.get("roi_x", 0))))
            y = int(round(float(row.get("roi_y", 0))))
            w = int(round(float(row.get("roi_w", 0))))
            h = int(round(float(row.get("roi_h", 0))))
            if w <= 0 or h <= 0:
                continue
            frame_path = mask_dir.parent / "output_frames" / f"{fid:08d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1 = max(0, min(frame.shape[1], x))
            y1 = max(0, min(frame.shape[0], y))
            x2 = max(0, min(frame.shape[1], x + w))
            y2 = max(0, min(frame.shape[0], y + h))
            mask[y1:y2, x1:x2] = 255
            region_id = row.get("region_id", "R000000")
            cv2.imwrite(str(mask_dir / f"r_{fid}_{region_id}.png"), mask)


def _best_identity(module, embedding: np.ndarray, identity_embeddings: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    best_id = ""
    best_score = 0.0
    normalized = module.normalize_embedding(embedding)
    for person_id, embeddings in identity_embeddings.items():
        for enrolled in embeddings:
            score = module.cosine_similarity(normalized, enrolled)
            if score > best_score:
                best_id = person_id
                best_score = float(score)
    return best_id, best_score


def _write_attack_log(
    module,
    face_app,
    output_frame_dir: Path,
    attack_log_path: Path,
    identity_embeddings: dict[str, list[np.ndarray]],
    threshold: float,
    system_id: str,
    dataset: str,
    sequence_id: str,
) -> None:
    with open(attack_log_path, "w", newline="") as f:
        fieldnames = [
            "system_id", "dataset", "sequence_id", "frame_id", "post_face_id",
            "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h",
            "face_confidence", "predicted_identity", "identity_confidence", "recognized",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_path in sorted(output_frame_dir.glob("*.jpg")):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            faces = module.get_faces_robust(frame, face_app)
            for idx, face in enumerate(faces):
                predicted, confidence = _best_identity(module, face.embedding, identity_embeddings)
                recognized = bool(predicted and confidence >= threshold)
                x1, y1, x2, y2 = [float(v) for v in face.bbox]
                writer.writerow({
                    "system_id": system_id,
                    "dataset": dataset,
                    "sequence_id": sequence_id,
                    "frame_id": int(frame_path.stem),
                    "post_face_id": idx,
                    "face_bbox_x": round(x1, 3),
                    "face_bbox_y": round(y1, 3),
                    "face_bbox_w": round(max(0.0, x2 - x1), 3),
                    "face_bbox_h": round(max(0.0, y2 - y1), 3),
                    "face_confidence": round(float(getattr(face, "det_score", 0.0) or 0.0), 6),
                    "predicted_identity": predicted if recognized else "",
                    "identity_confidence": round(confidence, 6),
                    "recognized": int(recognized),
                })


def _run_one_sequence(
    module,
    face_app,
    swapper,
    config: dict[str, Any],
    sequence: dict[str, Any],
    output_root: Path,
    donor_img: np.ndarray,
    identity_embeddings: dict[str, list[np.ndarray]],
    attack_identity_embeddings: dict[str, list[np.ndarray]],
    threshold: float,
    runtime_params: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    sid = sequence["id"]
    seq_dir = output_root / sid
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)

    target_fps = float(config["fps"]["target"])
    source_fps = float(config["fps"]["source"])
    frame_paths = _frame_paths(REPO_ROOT / sequence["frames_dir"], source_fps, target_fps)
    sampled_video = seq_dir / f"{sid}_sampled_input.mp4"
    dense_to_original = _write_sampled_video(frame_paths, sampled_video, target_fps)

    print(f"[{sid}] running AyuGram on {len(frame_paths)} sampled frames", flush=True)
    start = time.time()
    output_video, error, stats = module.process_video(
        face_app=face_app,
        swapper=swapper,
        input_video_path=str(sampled_video),
        donor_img=donor_img,
        protected_embeddings=_flatten_embeddings(identity_embeddings),
        similarity_threshold=threshold,
        frame_stride=1,
        det_thresh=float(runtime_params["det_thresh"]),
        calc_metrics=False,
        draw_boxes=False,
        min_face_size=int(runtime_params["min_face_size"]),
        iou_dedup_thr=module.DEFAULT_IOU_DEDUP,
        exhaustive_search=False,
        small_face_boost=False,
        track_iou_thr=module.DEFAULT_TRACK_IOU_THRESHOLD,
        track_max_missed=module.DEFAULT_TRACK_MAX_MISSED,
        protected_history=module.DEFAULT_PROTECTED_HISTORY,
        protected_votes_required=module.DEFAULT_PROTECTED_VOTES_REQUIRED,
        experiment_name="compatible_eval",
        progress_bar=_Progress(),
        status_placeholder=_Status(),
        status_prefix=f"{sid}: ",
        system_id="ayugram",
        dataset=config.get("dataset", "ChokePoint"),
        sequence_id=sid,
        input_video_name=sampled_video.name,
        latency_budget_ms=float(config.get("thresholds", {}).get("deadline_ms", 800.0)),
        save_document_logs=True,
    )
    elapsed = time.time() - start
    if error:
        raise RuntimeError(f"AyuGram failed for {sid}: {error}")
    if not output_video:
        raise RuntimeError(f"AyuGram did not return an output video for {sid}")
    stats["compatibility_runtime_params"] = runtime_params

    run_log_dir = Path(stats["run_log_dir"])
    for name in ["face_log.csv", "anonymization_log.csv", "frame_runtime_log.csv", "body_log.csv", "face_body_link_log.csv", "config.json"]:
        src = run_log_dir / name
        if src.exists():
            shutil.copy2(src, seq_dir / name)

    frame_out_dir = seq_dir / "output_frames"
    _extract_output_frames(Path(output_video), dense_to_original, frame_out_dir)
    for name in ["face_log.csv", "anonymization_log.csv", "frame_runtime_log.csv", "body_log.csv", "face_body_link_log.csv"]:
        _remap_csv_frame_ids(seq_dir / name, dense_to_original, frame_out_dir)

    _write_masks_from_anon_log(seq_dir / "anonymization_log.csv", seq_dir / "masks")
    shutil.copy2(output_video, seq_dir / "anonymized.mp4")
    _write_attack_log(
        module=module,
        face_app=face_app,
        output_frame_dir=frame_out_dir,
        attack_log_path=seq_dir / "attack_log.csv",
        identity_embeddings=attack_identity_embeddings,
        threshold=threshold,
        system_id="ayugram",
        dataset=config.get("dataset", "ChokePoint"),
        sequence_id=sid,
    )

    return elapsed, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chokepoint_eval.yaml"))
    parser.add_argument("--ayugram-app", type=Path, default=Path.home() / "Downloads/AyuGram Desktop/weights/app_improved_tracking_ablation.py")
    parser.add_argument("--output-root", type=Path, default=Path("results/chokepoint_ayugram"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--exact-inswapper", action="store_true", help="Use AyuGram's real InSwapper model; very slow on CPU")
    args = parser.parse_args()

    with open(args.config if args.config.is_absolute() else REPO_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    runtime_params = _load_base_runtime_params(config)
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    module = _load_module(args.ayugram_app)
    if args.exact_inswapper:
        face_app, swapper = module.load_models(runtime_params["det_size"])
    else:
        face_app = module.load_face_analysis_only(runtime_params["det_size"])
        swapper = _BlurSwapper(module)

    ref_sequence_id = config["enrollment"]["ref_sequence"]
    ref_sequence = next(s for s in config["sequences"] if s["id"] == ref_sequence_id)
    all_people = get_all_person_ids(REPO_ROOT / ref_sequence["groundtruth_xml"])
    enrolled = _select_enrolled_people(all_people, float(config["enrollment"]["ratio"]))
    faces_dir = REPO_ROOT / "assets/chokepoint/faces" / ref_sequence_id

    print(f"Building AyuGram allowlist embeddings for {len(enrolled)} IDs: {enrolled}", flush=True)
    identity_embeddings = _build_identity_embeddings(module, face_app, faces_dir, enrolled)
    print(f"Building AyuGram attack-gallery embeddings for {len(all_people)} IDs", flush=True)
    attack_identity_embeddings = _build_identity_embeddings(module, face_app, faces_dir, all_people)
    donor_img = _choose_donor_image(faces_dir, set(enrolled))

    sequences = config["sequences"]
    if args.sequences:
        selected = set(args.sequences)
        sequences = [s for s in sequences if s["id"] in selected]

    elapsed_by_sequence: dict[str, float] = {}
    for sequence in sequences:
        elapsed, _stats = _run_one_sequence(
            module=module,
            face_app=face_app,
            swapper=swapper,
            config=config,
            sequence=sequence,
            output_root=output_root,
            donor_img=donor_img,
            identity_embeddings=identity_embeddings,
            attack_identity_embeddings=attack_identity_embeddings,
            threshold=float(args.threshold),
            runtime_params=runtime_params,
        )
        elapsed_by_sequence[sequence["id"]] = elapsed

    thresholds = config.get("thresholds", {})
    per_video: dict[str, dict[str, Any]] = {}
    for sequence in sequences:
        sid = sequence["id"]
        result = compute_metrics_for_sequence(
            sequence=sequence,
            output_dir=output_root / sid,
            enrolled_ids=set(enrolled),
            fps=float(config["fps"]["target"]),
            thresholds=thresholds,
            processing_time_seconds=elapsed_by_sequence[sid],
        )
        if result.error:
            raise RuntimeError(f"Metric computation failed for {sid}: {result.error}")
        per_video[sid] = result.metrics

    metric_keys = [
        "face_detection_recall", "non_gallery_face_anonymization_recall",
        "face_false_allow_rate",
        "regar", "ttfa_frames", "ttfa_ms", "allow_stability",
        "face_rcr", "effective_face_rcr", "face_per",
        "frasr", "fda",
        "b_ssim", "b_lpips",
        "fps", "mean_latency_ms", "p50_ms", "p95_ms", "p99_ms",
    ]
    combined = {}
    for key in metric_keys:
        values = [metrics.get(key) for metrics in per_video.values() if metrics.get(key) is not None]
        combined[key] = sum(values) / len(values) if values else None

    (output_root / "metrics_per_video.json").write_text(json.dumps(per_video, indent=2, default=str), encoding="utf-8")
    (output_root / "metrics_combined.json").write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    with open(output_root / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence"] + metric_keys)
        for sid, metrics in per_video.items():
            writer.writerow([sid] + [f"{metrics.get(k):.6f}" if isinstance(metrics.get(k), float) else str(metrics.get(k)) for k in metric_keys])
        writer.writerow(["combined"] + [f"{combined.get(k):.6f}" if isinstance(combined.get(k), float) else str(combined.get(k)) for k in metric_keys])
    (output_root / "run_meta.json").write_text(json.dumps({
        "system_id": "ayugram",
        "ayugram_app": str(args.ayugram_app),
        "enrolled_ids": enrolled,
        "threshold": args.threshold,
        "compatibility_runtime_params": runtime_params,
        "exact_inswapper": bool(args.exact_inswapper),
        "anonymizer_note": "exact InSwapper" if args.exact_inswapper else "fast compatible blur substitute for AyuGram swap ROIs",
        "elapsed_by_sequence": elapsed_by_sequence,
    }, indent=2), encoding="utf-8")
    print(f"Wrote compatible AyuGram metrics to {output_root}", flush=True)


if __name__ == "__main__":
    main()
