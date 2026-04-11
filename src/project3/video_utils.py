from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class VideoData:
    frames: List[np.ndarray]
    fps: float
    size: tuple[int, int]


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for video processing. Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def read_video(video_path: str | Path) -> VideoData:
    cv2 = _require_cv2()
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    height, width = frames[0].shape[:2]
    return VideoData(frames=frames, fps=fps, size=(width, height))


def write_video(video_path: str | Path, frames: List[np.ndarray], fps: float) -> None:
    cv2 = _require_cv2()
    if not frames:
        raise ValueError("No frames provided for video writing.")

    output_path = Path(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for: {output_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()


def save_frames(frames: List[np.ndarray], directory: str | Path, stem: str = "frame") -> None:
    cv2 = _require_cv2()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(target / f"{stem}_{idx:05d}.png"), frame)


def load_grayscale_images(directory: str | Path) -> List[np.ndarray]:
    cv2 = _require_cv2()
    from .io_utils import list_images

    images = []
    for path in list_images(directory):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        images.append(image)
    return images
