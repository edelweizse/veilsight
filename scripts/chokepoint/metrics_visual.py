"""ChokePoint visual utility metrics.

Computes B-SSIM and B-LPIPS on background regions. Prefer independent
ground-truth face masks over system anonymization masks so overmasking outside
the true face area is still counted as visual distortion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class _Roi:
    x: int
    y: int
    w: int
    h: int


def background_ssim(
    frame_dir: Path,
    mask_dir: Path,
    frame_ids: list[int],
    original_frames_dir: Path | None = None,
    img_ext: str = ".jpg",
    gt_face_rois_frame: dict[int, list[_Roi]] | None = None,
    ellipse_dilation: float = 0.20,
) -> float:
    values: list[float] = []
    for fid in frame_ids:
        anon_path = frame_dir / f"{fid:08d}{img_ext}"
        orig_path = (original_frames_dir / f"{fid:08d}{img_ext}") if original_frames_dir else None
        if orig_path and orig_path.exists():
            orig = cv2.imread(str(orig_path))
        else:
            continue
        if not anon_path.exists():
            continue
        anon = cv2.imread(str(anon_path))
        if orig is None or anon is None:
            continue

        mask = _load_background_exclusion_mask(
            mask_dir,
            fid,
            orig.shape[:2],
            gt_face_rois_frame,
            ellipse_dilation,
        )
        if mask is None:
            values.append(_compute_ssim(orig, anon, None))
            continue

        if mask.shape != orig.shape[:2]:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
        bg_mask = (mask == 0)
        ssim_val = _compute_ssim(orig, anon, bg_mask)
        values.append(ssim_val)

    if not values:
        return 0.0
    return float(np.mean(values))


def background_lpips(
    frame_dir: Path,
    mask_dir: Path,
    frame_ids: list[int],
    original_frames_dir: Path | None = None,
    img_ext: str = ".jpg",
    gt_face_rois_frame: dict[int, list[_Roi]] | None = None,
    ellipse_dilation: float = 0.20,
) -> float:
    try:
        import lpips
        import torch
    except ImportError:
        return float("nan")

    loss_fn = lpips.LPIPS(net="alex")
    values: list[float] = []
    for fid in frame_ids:
        anon_path = frame_dir / f"{fid:08d}{img_ext}"
        orig_path = (original_frames_dir / f"{fid:08d}{img_ext}") if original_frames_dir else None
        if not orig_path or not orig_path.exists() or not anon_path.exists():
            continue
        orig = cv2.imread(str(orig_path))
        anon = cv2.imread(str(anon_path))
        if orig is None or anon is None:
            continue

        mask = _load_background_exclusion_mask(
            mask_dir,
            fid,
            orig.shape[:2],
            gt_face_rois_frame,
            ellipse_dilation,
        )
        if mask is not None:
            if mask.shape != orig.shape[:2]:
                mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
            bg_mask = (mask == 0)
            orig = orig * bg_mask[:, :, None]
            anon = anon * bg_mask[:, :, None]

        orig_t = torch.from_numpy(orig).permute(2, 0, 1).float() / 255.0
        anon_t = torch.from_numpy(anon).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            val = loss_fn(orig_t, anon_t).item()
        values.append(val)

    if not values:
        return float("nan")
    return float(np.mean(values))


def _load_background_exclusion_mask(
    mask_dir: Path,
    frame_id: int,
    frame_shape: tuple[int, int],
    gt_face_rois_frame: dict[int, list[_Roi]] | None = None,
    ellipse_dilation: float = 0.20,
) -> np.ndarray | None:
    if gt_face_rois_frame is not None:
        gt_mask = _build_gt_face_ellipse_mask(
            frame_shape,
            gt_face_rois_frame.get(frame_id, []),
            ellipse_dilation,
        )
        if gt_mask is not None:
            return gt_mask
    return _load_composite_mask(mask_dir, frame_id)


def _build_gt_face_ellipse_mask(
    frame_shape: tuple[int, int],
    rois: list[_Roi],
    dilation: float = 0.20,
) -> np.ndarray | None:
    if not rois:
        return None
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for roi in rois:
        if roi.w <= 0 or roi.h <= 0:
            continue
        cx = int(round(roi.x + roi.w * 0.5))
        cy = int(round(roi.y + roi.h * 0.5))
        axis_x = max(1, int(round(roi.w * (0.5 + dilation))))
        axis_y = max(1, int(round(roi.h * (0.5 + dilation))))
        cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0, 0, 360, 255, -1)
    if not mask.any():
        return None
    return mask


def _load_composite_mask(mask_dir: Path, frame_id: int) -> np.ndarray | None:
    prefix = f"r_{frame_id}_"
    masks = sorted(mask_dir.glob(f"{prefix}*.png"))
    if not masks:
        return None
    composite = None
    for mp in masks:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if composite is None:
            composite = np.zeros_like(m, dtype=np.uint8)
        composite = np.maximum(composite, m)
    return composite


def _compute_ssim(orig: np.ndarray, anon: np.ndarray, bg_mask: np.ndarray | None) -> float:
    gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_anon = cv2.cvtColor(anon, cv2.COLOR_BGR2GRAY)
    if bg_mask is not None:
        bg_float = bg_mask.astype(np.float32)
        total_px = bg_float.sum()
        if total_px < 100:
            return 1.0
        ssim_val, _ = structural_similarity(
            gray_orig, gray_anon, full=True,
            data_range=255, gradient=False, sigma=1.5,
        )
        ssim_map = _
        masked_ssim = (ssim_map * bg_float).sum() / total_px
        return float(masked_ssim)
    else:
        ssim_val, _ = structural_similarity(
            gray_orig, gray_anon, full=True,
            data_range=255,
        )
        return float(ssim_val)
