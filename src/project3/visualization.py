from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .io_utils import ensure_dir, list_images


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for figure generation. Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def _sample_indices(length: int, samples: int) -> List[int]:
    if length <= samples:
        return list(range(length))
    return np.linspace(0, length - 1, num=samples, dtype=int).tolist()


def generate_comparison_grid(experiment_dir: str | Path, output_path: str | Path, samples: int = 6) -> Path:
    cv2 = _require_cv2()
    experiment_dir = Path(experiment_dir)

    input_dir = experiment_dir / "input_frames"
    restored_dir = experiment_dir / "restored_frames"
    masks_dir = experiment_dir / "masks"

    input_paths = list_images(input_dir) if input_dir.exists() else []
    restored_paths = list_images(restored_dir)
    mask_paths = list_images(masks_dir)
    if not restored_paths or len(restored_paths) != len(mask_paths):
        raise RuntimeError("Expected matching restored frame and mask directories.")
    if input_paths and len(input_paths) != len(restored_paths):
        raise RuntimeError("Expected input frames to match restored frame count.")

    indices = _sample_indices(len(restored_paths), samples)
    rows = []
    for idx in indices:
        original = (
            cv2.imread(str(input_paths[idx]), cv2.IMREAD_COLOR) if input_paths else None
        )
        restored = cv2.imread(str(restored_paths[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_COLOR)
        if restored is None or mask is None:
            raise RuntimeError("Failed to read visualization inputs.")
        if original is None:
            original = np.zeros_like(restored)

        label = np.full((40, restored.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            label,
            f"Frame {idx:03d}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        column_titles = np.full((40, restored.shape[1] * 3, 3), 255, dtype=np.uint8)
        for column_idx, title in enumerate(["Original", "Mask", "Restored"]):
            cv2.putText(
                column_titles,
                title,
                (12 + column_idx * restored.shape[1], 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

        comparison_row = np.concatenate([original, mask, restored], axis=1)
        row = np.concatenate([label, column_titles, comparison_row], axis=0)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    output = Path(output_path)
    ensure_dir(output.parent)
    cv2.imwrite(str(output), grid)
    return output
