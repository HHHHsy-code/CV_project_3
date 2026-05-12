from __future__ import annotations

from pathlib import Path
import re
from typing import List

import numpy as np

from .io_utils import ensure_dir, list_images
from .video_utils import read_video


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

        total_width = restored.shape[1] * 3
        label = np.full((40, total_width, 3), 255, dtype=np.uint8)
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
        column_titles = np.full((40, total_width, 3), 255, dtype=np.uint8)
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


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    ref_height, ref_width = reference.shape[:2]
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (ref_width, ref_height), interpolation=cv2.INTER_AREA)


def _text_row(text: str, width: int, height: int = 40) -> np.ndarray:
    cv2 = _require_cv2()
    row = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        row,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return row


def _frame_index_from_path(path: Path) -> int | None:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def _indexed_images(directory: str | Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in list_images(directory):
        frame_index = _frame_index_from_path(path)
        if frame_index is None:
            continue
        mapping[frame_index] = path
    return mapping


def generate_method_comparison_grid(
    part1_dir: str | Path,
    part2_video: str | Path,
    output_path: str | Path,
    samples: int = 6,
    mask_dir: str | Path | None = None,
) -> Path:
    """Create a report-ready Original/Mask/Part1/Part2 comparison grid."""

    cv2 = _require_cv2()
    part1_dir = Path(part1_dir)
    input_paths = list_images(part1_dir / "input_frames")
    mask_paths = list_images(mask_dir) if mask_dir is not None else list_images(part1_dir / "masks")
    part1_paths = list_images(part1_dir / "restored_frames")
    part2_frames = read_video(part2_video).frames

    frame_count = min(len(input_paths), len(mask_paths), len(part1_paths), len(part2_frames))
    if frame_count == 0:
        raise RuntimeError("Expected non-empty Part 1 frames, masks, and Part 2 video frames.")

    indices = _sample_indices(frame_count, samples)
    rows = []
    for idx in indices:
        original = cv2.imread(str(input_paths[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_COLOR)
        part1 = cv2.imread(str(part1_paths[idx]), cv2.IMREAD_COLOR)
        part2 = part2_frames[idx]
        if original is None or mask is None or part1 is None:
            raise RuntimeError(f"Failed to read comparison inputs for frame {idx}.")

        mask = _resize_like(mask, original)
        part1 = _resize_like(part1, original)
        part2 = _resize_like(part2, original)

        column_width = original.shape[1]
        total_width = column_width * 4
        label = _text_row(f"Frame {idx:03d}", total_width)
        titles = np.full((40, total_width, 3), 255, dtype=np.uint8)
        for column_idx, title in enumerate(["Original", "Mask", "Part 1", "Part 2"]):
            cv2.putText(
                titles,
                title,
                (12 + column_idx * column_width, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

        comparison_row = np.concatenate([original, mask, part1, part2], axis=1)
        rows.append(np.concatenate([label, titles, comparison_row], axis=0))

    grid = np.concatenate(rows, axis=0)
    output = Path(output_path)
    ensure_dir(output.parent)
    cv2.imwrite(str(output), grid)
    return output


def generate_keyframe_comparison_grid(
    original_dir: str | Path,
    mask_dir: str | Path,
    reference_dir: str | Path,
    edited_dir: str | Path,
    output_path: str | Path,
    keyframes: List[int],
    reference_title: str = "ProPainter",
    edited_title: str = "AttentiveEraser",
) -> Path:
    cv2 = _require_cv2()
    original_lookup = _indexed_images(original_dir)
    mask_lookup = _indexed_images(mask_dir)
    reference_lookup = _indexed_images(reference_dir)
    edited_lookup = _indexed_images(edited_dir)

    rows = []
    for idx in keyframes:
        try:
            original_path = original_lookup[idx]
            mask_path = mask_lookup[idx]
            reference_path = reference_lookup[idx]
            edited_path = edited_lookup[idx]
        except KeyError as exc:
            raise RuntimeError(f"Missing keyframe asset for frame {idx:05d}.") from exc

        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        edited = cv2.imread(str(edited_path), cv2.IMREAD_COLOR)
        if original is None or mask is None or reference is None or edited is None:
            raise RuntimeError(f"Failed to read a comparison image for frame {idx:05d}.")

        mask = _resize_like(mask, original)
        reference = _resize_like(reference, original)
        edited = _resize_like(edited, original)

        column_width = original.shape[1]
        total_width = column_width * 4
        label = _text_row(f"Frame {idx:03d}", total_width)
        titles = np.full((40, total_width, 3), 255, dtype=np.uint8)
        for column_idx, title in enumerate(["Original", "Mask", reference_title, edited_title]):
            cv2.putText(
                titles,
                title,
                (12 + column_idx * column_width, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

        row = np.concatenate([original, mask, reference, edited], axis=1)
        rows.append(np.concatenate([label, titles, row], axis=0))

    grid = np.concatenate(rows, axis=0)
    output = Path(output_path)
    ensure_dir(output.parent)
    cv2.imwrite(str(output), grid)
    return output
