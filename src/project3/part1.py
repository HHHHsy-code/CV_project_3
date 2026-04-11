from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import ensure_dir, write_json
from .video_utils import read_frame_directory, read_video, save_frames, write_video


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for the baseline pipeline. Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8 segmentation. Install it with `pip install ultralytics`."
        ) from exc
    return YOLO


@dataclass
class Detection:
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox: np.ndarray


class BaselineVideoObjectRemoval:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["part1"]
        self.cv2 = _require_cv2()
        model_name = self.config["detector"]["model"]
        yolo = _require_ultralytics()
        self.detector = yolo(model_name)
        self.class_filter = set(self.config["detector"]["classes"])
        self.cv2_method = {
            "telea": self.cv2.INPAINT_TELEA,
            "ns": self.cv2.INPAINT_NS,
        }[self.config["inpainting"]["cv2_method"].lower()]

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        fps: float | None = None,
    ) -> Dict[str, Any]:
        video, input_type = self._load_input(input_path, fps=fps)
        output_root = ensure_dir(output_dir)
        input_frames_dir = ensure_dir(output_root / "input_frames")
        masks_dir = ensure_dir(output_root / "masks")
        restored_dir = ensure_dir(output_root / "restored_frames")
        figures_dir = ensure_dir(output_root / "figures")

        detections_per_frame = self._segment_frames(video.frames)
        masks = self._build_dynamic_masks(video.frames, detections_per_frame)
        restored_frames = self._restore_background(video.frames, masks)
        mask_rgb_frames = [self.cv2.cvtColor(mask, self.cv2.COLOR_GRAY2BGR) for mask in masks]

        save_frames(video.frames, input_frames_dir, stem="input")
        save_frames(mask_rgb_frames, masks_dir, stem="mask")
        save_frames(restored_frames, restored_dir, stem="restored")
        write_video(output_root / "mask_video.mp4", mask_rgb_frames, video.fps)
        write_video(output_root / "restored_video.mp4", restored_frames, video.fps)

        summary = {
            "input_path": str(input_path),
            "input_type": input_type,
            "num_frames": len(video.frames),
            "fps": video.fps,
            "frame_size": {"width": video.size[0], "height": video.size[1]},
            "detector_model": self.config["detector"]["model"],
            "detector_classes": sorted(self.class_filter),
            "mean_mask_ratio": float(np.mean([mask.mean() / 255.0 for mask in masks])),
            "outputs": {
                "input_frames_dir": str(input_frames_dir),
                "masks_dir": str(masks_dir),
                "restored_dir": str(restored_dir),
                "mask_video": str(output_root / "mask_video.mp4"),
                "restored_video": str(output_root / "restored_video.mp4"),
                "figures_dir": str(figures_dir),
            },
        }
        write_json(summary, output_root / "summary.json")
        return summary

    def _load_input(self, input_path: str | Path, fps: float | None = None):
        source = Path(input_path)
        if source.is_dir():
            video = read_frame_directory(source, fps=fps or 24.0)
            return video, "frames_dir"
        video = read_video(source)
        return video, "video"

    def _segment_frames(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        conf = self.config["detector"]["conf_threshold"]
        iou = self.config["detector"]["iou_threshold"]
        device = self.config["detector"].get("device")

        detections_per_frame: List[List[Detection]] = []
        for frame in frames:
            results = self.detector.predict(
                source=frame,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )
            detections_per_frame.append(self._parse_detections(frame, results[0]))
        return detections_per_frame

    def _parse_detections(self, frame: np.ndarray, result: Any) -> List[Detection]:
        detections: List[Detection] = []
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or boxes is None or masks.data is None:
            return detections

        height, width = frame.shape[:2]
        names = getattr(result, "names", {})
        for index in range(len(boxes.cls)):
            class_id = int(boxes.cls[index].item())
            class_name = names.get(class_id, str(class_id))
            if class_name not in self.class_filter:
                continue

            mask = masks.data[index].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask = self.cv2.resize(mask, (width, height), interpolation=self.cv2.INTER_NEAREST)
            bbox = boxes.xyxy[index].cpu().numpy()
            confidence = float(boxes.conf[index].item())
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    mask=mask,
                    bbox=bbox,
                )
            )
        return detections

    def _build_dynamic_masks(
        self, frames: List[np.ndarray], detections_per_frame: List[List[Detection]]
    ) -> List[np.ndarray]:
        masks: List[np.ndarray] = []
        prev_gray = None
        for frame, detections in zip(frames, detections_per_frame):
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            frame_mask = np.zeros_like(gray, dtype=np.uint8)
            if prev_gray is None:
                prev_gray = gray
                masks.append(frame_mask)
                continue

            for detection in detections:
                if self._is_dynamic(prev_gray, gray, detection.mask):
                    frame_mask = self.cv2.bitwise_or(frame_mask, detection.mask)

            frame_mask = self._refine_mask(frame_mask)
            masks.append(frame_mask)
            prev_gray = gray
        return masks

    def _is_dynamic(self, prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> bool:
        motion_cfg = self.config["motion"]
        feature_params = dict(
            maxCorners=motion_cfg["max_corners"],
            qualityLevel=motion_cfg["quality_level"],
            minDistance=motion_cfg["min_distance"],
            blockSize=motion_cfg["block_size"],
        )
        points = self.cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
        if points is None or len(points) == 0:
            return False

        lk_params = dict(
            winSize=(motion_cfg["lk_win_size"], motion_cfg["lk_win_size"]),
            maxLevel=motion_cfg["lk_max_level"],
            criteria=(
                self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        next_points, status, _ = self.cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points, None, **lk_params
        )
        if next_points is None or status is None:
            return False

        valid = status.reshape(-1) == 1
        if not np.any(valid):
            return False

        motion = np.linalg.norm(next_points[valid] - points[valid], axis=2).mean()
        return float(motion) >= float(motion_cfg["motion_threshold"])

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_cfg = self.config["mask"]
        kernel_size = int(mask_cfg["dilation_kernel"])
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        refined = self.cv2.dilate(mask, kernel, iterations=int(mask_cfg["dilation_iters"]))

        num_labels, labels, stats, _ = self.cv2.connectedComponentsWithStats(refined, connectivity=8)
        cleaned = np.zeros_like(refined)
        for label in range(1, num_labels):
            area = stats[label, self.cv2.CC_STAT_AREA]
            if area >= int(mask_cfg["min_area"]):
                cleaned[labels == label] = 255
        return cleaned

    def _restore_background(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        restored = []
        temporal_cfg = self.config["inpainting"]
        use_temporal = bool(temporal_cfg["use_temporal_fill"])
        use_inpaint = bool(temporal_cfg["use_cv2_inpaint"])
        radius = float(temporal_cfg["cv2_radius"])
        window = int(temporal_cfg["temporal_window"])

        frame_stack = np.stack(frames, axis=0)
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            repaired = frame.copy()
            if mask.max() == 0:
                restored.append(repaired)
                continue

            if use_temporal:
                temporal_fill = self._temporal_median_fill(frame_stack, masks, index, window)
                masked_pixels = mask > 0
                repaired[masked_pixels] = temporal_fill[masked_pixels]

            if use_inpaint:
                unresolved = mask.copy()
                repaired = self.cv2.inpaint(repaired, unresolved, radius, self.cv2_method)

            restored.append(repaired)
        return restored

    def _temporal_median_fill(
        self, frame_stack: np.ndarray, masks: List[np.ndarray], center_idx: int, window: int
    ) -> np.ndarray:
        start = max(0, center_idx - window)
        end = min(len(frame_stack), center_idx + window + 1)

        candidates = frame_stack[start:end].copy()
        candidate_masks = np.stack(masks[start:end], axis=0) > 0

        masked_candidates = candidates.astype(np.float32)
        masked_candidates[candidate_masks] = np.nan

        median = np.nanmedian(masked_candidates, axis=0)
        if np.isnan(median).any():
            fallback = frame_stack[center_idx].astype(np.float32)
            nan_mask = np.isnan(median)
            median[nan_mask] = fallback[nan_mask]
        return np.clip(median, 0, 255).astype(np.uint8)
