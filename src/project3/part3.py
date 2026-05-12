from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import ensure_dir, list_images, write_json
from .video_utils import read_video


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for Part 3 asset preparation. Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("The selected seed mask is empty; choose a frame with a visible target.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int, margin: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width - 1, x1 + margin)
    y1 = min(height - 1, y1 + margin)
    return [x0, y0, x1, y1]


def build_box_prompt_from_mask(
    mask_dir: str | Path,
    frame_index: int,
    output_path: str | Path | None = None,
    margin: int = 24,
) -> Dict[str, Any]:
    cv2 = _require_cv2()
    mask_paths = list_images(mask_dir)
    if frame_index < 0 or frame_index >= len(mask_paths):
        raise IndexError(f"Frame index {frame_index} is outside the mask range [0, {len(mask_paths) - 1}].")

    mask_path = mask_paths[frame_index]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    height, width = mask.shape[:2]
    prompt_box = _clamp_box(_mask_bbox(mask), width, height, margin)
    point_xy = [int((prompt_box[0] + prompt_box[2]) / 2), int((prompt_box[1] + prompt_box[3]) / 2)]
    prompt = {
        "frame_index": frame_index,
        "prompt_mask_path": str(mask_path),
        "prompt_box_xyxy": prompt_box,
        "prompt_point_xy": point_xy,
        "prompt_type": "box",
        "margin": margin,
    }
    if output_path is not None:
        write_json(prompt, output_path)
    return prompt


class Part3Adapters:
    """Store external-tool metadata and standardize experiment directories for Part 3."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("part3", {})

    def prepare_track_anything_experiment(
        self,
        experiment_dir: str | Path,
        prompt: Dict[str, Any],
    ) -> Dict[str, str]:
        root = ensure_dir(experiment_dir)
        paths = {
            "root": str(root),
            "masks": str(ensure_dir(root / "masks")),
            "prompts": str(ensure_dir(root / "prompts")),
            "comparisons": str(ensure_dir(root / "comparisons")),
            "propainter_output": str(ensure_dir(root / "propainter_output")),
        }
        metadata = {
            "track_anything_repo": self.config.get("track_anything", {}).get("repo_dir", ""),
            "track_anything_entrypoint": self.config.get("track_anything", {}).get("entrypoint", ""),
            "attentive_eraser_repo": self.config.get("attentive_eraser", {}).get("repo_dir", ""),
            "attentive_eraser_entrypoint": self.config.get("attentive_eraser", {}).get("entrypoint", ""),
            "prompt": prompt,
            "recommended_keyframes": self.config.get("keyframes", []),
        }
        write_json(metadata, root / "part3_metadata.json")
        return paths

    def recommended_track_anything_commands(
        self,
        video_path: str | Path,
        experiment_dir: str | Path,
        prompt_json: str | Path,
    ) -> Dict[str, str]:
        root = Path(experiment_dir)
        masks_dir = root / "masks"
        propainter_output = root / "propainter_output"
        track_repo = self.config.get("track_anything", {}).get("repo_dir", "external/Track-Anything")
        track_entry = self.config.get("track_anything", {}).get("entrypoint", "app.py")
        propainter_repo = self.config.get("propainter", {}).get("repo_dir", "external/ProPainter")

        return {
            "track_anything_template": (
                f"cd {track_repo} && "
                f"python {track_entry}  # initialize the object on the prompt frame stored in {prompt_json}; "
                f"export propagated masks into {masks_dir}"
            ),
            "propainter": (
                f"cd {propainter_repo} && "
                f"python inference_propainter.py --video {video_path} "
                f"--mask {masks_dir} --output {propainter_output}"
            ),
        }

    def recommended_attentive_eraser_commands(self, workspace_dir: str | Path) -> Dict[str, str]:
        root = Path(workspace_dir)
        attentive_repo = self.config.get("attentive_eraser", {}).get("repo_dir", "external/AttentiveEraser")
        return {
            "attentive_eraser_template": (
                "python scripts/run_attentive_eraser_keyframes.py "
                f"--workspace {root} "
                f"--repo-dir {attentive_repo} "
                "--model-id stabilityai/stable-diffusion-xl-base-1.0 "
                "--device cuda:0 "
                "# run image-level removal on frames from "
                f"{root / 'keyframes'} with masks from {root / 'masks'} and save into {root / 'edited'}"
            )
        }


def prepare_failure_case_workspace(
    experiment_dir: str | Path,
    failure_case_name: str,
    keyframes: List[int],
    notes: str,
) -> Dict[str, str]:
    root = ensure_dir(Path(experiment_dir) / failure_case_name)
    paths = {
        "root": str(root),
        "keyframes": str(ensure_dir(root / "keyframes")),
        "masks": str(ensure_dir(root / "masks")),
        "reference": str(ensure_dir(root / "reference")),
        "edited": str(ensure_dir(root / "edited")),
        "comparisons": str(ensure_dir(root / "comparisons")),
    }
    write_json(
        {
            "failure_case": failure_case_name,
            "keyframes": keyframes,
            "notes": notes,
            "goal": "Repair frames where the current video-level remover still leaves visible artifacts.",
        },
        root / "part3_plan.json",
    )
    return paths


def export_failure_case_assets(
    video_path: str | Path,
    mask_dir: str | Path,
    reference_video: str | Path,
    workspace_dir: str | Path,
    keyframes: List[int],
) -> Dict[str, Any]:
    cv2 = _require_cv2()
    workspace = Path(workspace_dir)
    keyframe_dir = ensure_dir(workspace / "keyframes")
    mask_output_dir = ensure_dir(workspace / "masks")
    reference_dir = ensure_dir(workspace / "reference")

    source_video = read_video(video_path)
    reference = read_video(reference_video)
    mask_paths = list_images(mask_dir)

    frame_count = min(len(source_video.frames), len(reference.frames), len(mask_paths))
    if frame_count == 0:
        raise RuntimeError("Expected non-empty video, reference video, and mask directory.")

    exported = []
    for frame_index in keyframes:
        if frame_index < 0 or frame_index >= frame_count:
            raise IndexError(f"Keyframe {frame_index} is outside the valid range [0, {frame_count - 1}].")

        original = source_video.frames[frame_index]
        reference_frame = reference.frames[frame_index]
        mask = cv2.imread(str(mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_paths[frame_index]}")
        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

        original_path = keyframe_dir / f"frame_{frame_index:05d}.png"
        mask_path = mask_output_dir / f"mask_{frame_index:05d}.png"
        reference_path = reference_dir / f"reference_{frame_index:05d}.png"

        cv2.imwrite(str(original_path), original)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(reference_path), reference_frame)
        exported.append(
            {
                "frame_index": frame_index,
                "original": str(original_path),
                "mask": str(mask_path),
                "reference": str(reference_path),
            }
        )

    manifest = {
        "video": str(video_path),
        "mask_dir": str(mask_dir),
        "reference_video": str(reference_video),
        "workspace": str(workspace),
        "keyframes": keyframes,
        "exported": exported,
    }
    write_json(manifest, workspace / "export_manifest.json")
    return manifest


def copy_edited_results(
    edited_dir: str | Path,
    destination_dir: str | Path,
    keyframes: List[int],
) -> Dict[str, Any]:
    source = Path(edited_dir)
    destination = ensure_dir(destination_dir)
    copied = []
    for frame_index in keyframes:
        matches = [item for item in list_images(source) if f"{frame_index:05d}" in item.stem]
        if not matches:
            raise FileNotFoundError(f"Could not find an edited result for keyframe {frame_index} in {source}.")
        target = destination / f"edited_{frame_index:05d}{matches[0].suffix.lower()}"
        shutil.copy2(matches[0], target)
        copied.append(str(target))
    result = {"edited_dir": str(source), "destination_dir": str(destination), "copied": copied}
    write_json(result, destination / "edited_manifest.json")
    return result


def convert_track_anything_npy_masks(
    source_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    cv2 = _require_cv2()
    source = Path(source_dir)
    output = ensure_dir(output_dir)

    npy_paths = sorted(source.glob("*.npy"))
    if not npy_paths:
        raise FileNotFoundError(f"No .npy masks found in {source}.")

    converted = []
    for npy_path in npy_paths:
        mask = np.load(npy_path)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        target = output / f"mask_{npy_path.stem}.png"
        cv2.imwrite(str(target), binary)
        converted.append(str(target))

    result = {
        "source_dir": str(source),
        "output_dir": str(output),
        "converted_count": len(converted),
        "converted": converted,
    }
    write_json(result, output / "conversion_manifest.json")
    return result


def merge_mask_dirs(
    mask_dirs: List[str | Path],
    output_dir: str | Path,
    mode: str = "union",
) -> Dict[str, Any]:
    cv2 = _require_cv2()
    if not mask_dirs:
        raise ValueError("Expected at least one mask directory.")

    image_lists = [list_images(mask_dir) for mask_dir in mask_dirs]
    counts = [len(items) for items in image_lists]
    if min(counts) == 0:
        raise ValueError("All mask directories must be non-empty.")
    if len(set(counts)) != 1:
        raise ValueError(f"Mask directories do not have the same frame count: {counts}")

    output = ensure_dir(output_dir)
    frame_count = counts[0]
    merged_paths: List[str] = []

    for frame_index in range(frame_count):
        masks = []
        for image_list in image_lists:
            image = cv2.imread(str(image_list[frame_index]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read mask: {image_list[frame_index]}")
            masks.append((image > 127).astype(np.uint8))

        stack = np.stack(masks, axis=0)
        if mode == "union":
            merged = (stack.max(axis=0) > 0).astype(np.uint8) * 255
        elif mode == "intersection":
            merged = (stack.min(axis=0) > 0).astype(np.uint8) * 255
        elif mode == "majority":
            threshold = len(masks) // 2 + 1
            merged = (stack.sum(axis=0) >= threshold).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unsupported merge mode: {mode}")

        target = output / f"mask_{frame_index:05d}.png"
        cv2.imwrite(str(target), merged)
        merged_paths.append(str(target))

    result = {
        "mask_dirs": [str(Path(mask_dir)) for mask_dir in mask_dirs],
        "output_dir": str(output),
        "mode": mode,
        "frame_count": frame_count,
        "merged_paths": merged_paths[:3] + (["..."] if len(merged_paths) > 3 else []),
    }
    write_json(result, output.parent / f"{output.name}_merge_manifest.json")
    return result
