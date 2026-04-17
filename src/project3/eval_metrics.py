from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .io_utils import list_images, write_json


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for evaluation image loading. Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _load_binary_masks(directory: str | Path) -> List[np.ndarray]:
    cv2 = _require_cv2()
    masks = []
    for path in list_images(directory):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        masks.append((image > 127).astype(np.uint8))
    return masks


def evaluate_mask_dir(pred_dir: str | Path, gt_dir: str | Path, output: str | Path | None = None) -> Dict:
    pred_masks = _load_binary_masks(pred_dir)
    gt_masks = _load_binary_masks(gt_dir)
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Prediction and GT mask counts do not match.")

    ious = []
    recalls = []
    precisions = []
    per_frame = []
    for idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
        intersection = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        gt_sum = int(gt.sum())
        pred_sum = int(pred.sum())

        iou = _safe_div(intersection, union)
        recall = _safe_div(intersection, gt_sum)
        precision = _safe_div(intersection, pred_sum)
        ious.append(iou)
        recalls.append(recall)
        precisions.append(precision)
        per_frame.append(
            {
                "frame": idx,
                "iou": iou,
                "recall": recall,
                "precision": precision,
            }
        )

    metrics = {
        "jm": float(np.mean(ious)),
        "jr": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "num_frames": len(pred_masks),
        "per_frame": per_frame,
    }
    if output is not None:
        write_json(metrics, output)
    return metrics


def _load_rgb_frames(directory: str | Path) -> List[np.ndarray]:
    cv2 = _require_cv2()
    frames = []
    for path in list_images(directory):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        frames.append(image)
    return frames


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError:
        pred_gray = pred.mean(axis=2)
        gt_gray = gt.mean(axis=2)
        c1 = 6.5025
        c2 = 58.5225
        mu_x = pred_gray.mean()
        mu_y = gt_gray.mean()
        sigma_x = pred_gray.var()
        sigma_y = gt_gray.var()
        sigma_xy = ((pred_gray - mu_x) * (gt_gray - mu_y)).mean()
        return float(
            ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2))
            / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        )

    return float(structural_similarity(pred, gt, channel_axis=2))


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))
    if mse == 0:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def evaluate_frame_dir(pred_dir: str | Path, gt_dir: str | Path, output: str | Path | None = None) -> Dict:
    pred_frames = _load_rgb_frames(pred_dir)
    gt_frames = _load_rgb_frames(gt_dir)
    if len(pred_frames) != len(gt_frames):
        raise ValueError("Prediction and GT frame counts do not match.")

    psnrs = []
    ssims = []
    per_frame = []
    for idx, (pred, gt) in enumerate(zip(pred_frames, gt_frames)):
        psnr = _psnr(pred, gt)
        ssim = _ssim(pred, gt)
        psnrs.append(psnr)
        ssims.append(ssim)
        per_frame.append({"frame": idx, "psnr": psnr, "ssim": ssim})

    metrics = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "num_frames": len(pred_frames),
        "per_frame": per_frame,
    }
    if output is not None:
        write_json(metrics, output)
    return metrics


def summarize_mask_dir(directory: str | Path, output: str | Path | None = None) -> Dict:
    masks = _load_binary_masks(directory)
    if not masks:
        raise ValueError("No masks found.")

    area_ratios = [float(mask.mean()) for mask in masks]
    non_empty = [ratio for ratio in area_ratios if ratio > 0.0]
    non_empty_frames = [idx for idx, ratio in enumerate(area_ratios) if ratio > 0.0]

    summary = {
        "num_frames": len(masks),
        "non_empty_frames": len(non_empty_frames),
        "temporal_coverage": _safe_div(len(non_empty_frames), len(masks)),
        "mean_area_ratio_all": float(np.mean(area_ratios)),
        "mean_area_ratio_non_empty": float(np.mean(non_empty)) if non_empty else 0.0,
        "max_area_ratio": float(np.max(area_ratios)),
        "first_non_empty_frame": int(non_empty_frames[0]) if non_empty_frames else None,
        "last_non_empty_frame": int(non_empty_frames[-1]) if non_empty_frames else None,
    }
    if output is not None:
        write_json(summary, output)
    return summary
