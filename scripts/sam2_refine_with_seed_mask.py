from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project3.io_utils import ensure_dir, list_images  # noqa: E402


def _load_binary_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.array(image)
    return array > 127


def _find_prompt_frame(mask_paths: Iterable[Path], prompt_frame: int | None) -> tuple[int, np.ndarray, Path]:
    mask_paths = list(mask_paths)
    if not mask_paths:
        raise RuntimeError("No seed masks found.")

    if prompt_frame is not None:
        if prompt_frame < 0 or prompt_frame >= len(mask_paths):
            raise ValueError(f"prompt_frame {prompt_frame} is out of range for {len(mask_paths)} masks")
        chosen_path = mask_paths[prompt_frame]
        chosen_mask = _load_binary_mask(chosen_path)
        if not chosen_mask.any():
            raise RuntimeError(
                f"Seed mask on frame {prompt_frame} is empty. Pick another frame or use automatic prompt selection."
            )
        return prompt_frame, chosen_mask, chosen_path

    best_idx = -1
    best_area = -1
    best_mask = None
    best_path = None
    for idx, path in enumerate(mask_paths):
        mask = _load_binary_mask(path)
        area = int(mask.sum())
        if area > best_area:
            best_idx = idx
            best_area = area
            best_mask = mask
            best_path = path

    if best_area <= 0 or best_mask is None or best_path is None:
        raise RuntimeError("All seed masks are empty; cannot bootstrap SAM 2 from the current Part 1 result.")

    return best_idx, best_mask, best_path


def _mask_to_box(mask: np.ndarray, margin: int) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Cannot compute a box from an empty mask.")

    min_x = max(0, int(xs.min()) - margin)
    min_y = max(0, int(ys.min()) - margin)
    max_x = int(xs.max()) + margin
    max_y = int(ys.max()) + margin
    return [min_x, min_y, max_x, max_y]


def _save_mask(mask_logits, output_path: Path) -> None:
    mask = (mask_logits > 0.0).squeeze().detach().cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine video masks with SAM 2 using a seed mask-derived box prompt.")
    parser.add_argument("--video", required=True, help="Path to the input video (.mp4)")
    parser.add_argument("--seed-mask-dir", required=True, help="Existing Part 1 mask directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save refined mask PNGs")
    parser.add_argument("--checkpoint", required=True, help="Path to a SAM 2.1 checkpoint")
    parser.add_argument("--model-cfg", required=True, help="SAM 2 config path, e.g. configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--prompt-frame", type=int, default=None, help="Optional explicit prompt frame index")
    parser.add_argument("--margin", type=int, default=16, help="Padding added to the prompt box")
    parser.add_argument("--obj-id", type=int, default=1, help="Object id used by SAM 2")
    parser.add_argument("--device", default="cuda", help="Device passed to SAM 2")
    parser.add_argument("--offload-video-to-cpu", action="store_true", help="Reduce GPU memory usage during frame loading")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    output_dir = ensure_dir(args.output_dir)
    seed_mask_paths = list_images(args.seed_mask_dir)
    prompt_frame, prompt_mask, prompt_mask_path = _find_prompt_frame(seed_mask_paths, args.prompt_frame)
    box = _mask_to_box(prompt_mask, margin=args.margin)

    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=args.device)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )

    saved_frames: set[int] = set()
    with torch.inference_mode(), autocast_context:
        state = predictor.init_state(
            video_path=args.video,
            offload_video_to_cpu=args.offload_video_to_cpu,
        )
        _, obj_ids, masks = predictor.add_new_points_or_box(
            state,
            frame_idx=prompt_frame,
            obj_id=args.obj_id,
            box=box,
        )

        obj_index = obj_ids.index(args.obj_id)
        _save_mask(masks[obj_index], output_dir / f"mask_{prompt_frame:05d}.png")
        saved_frames.add(prompt_frame)

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            state,
            start_frame_idx=prompt_frame,
            reverse=False,
        ):
            obj_index = obj_ids.index(args.obj_id)
            _save_mask(masks[obj_index], output_dir / f"mask_{frame_idx:05d}.png")
            saved_frames.add(frame_idx)

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            state,
            start_frame_idx=prompt_frame,
            reverse=True,
        ):
            obj_index = obj_ids.index(args.obj_id)
            _save_mask(masks[obj_index], output_dir / f"mask_{frame_idx:05d}.png")
            saved_frames.add(frame_idx)

    summary = {
        "video": args.video,
        "seed_mask_dir": args.seed_mask_dir,
        "output_dir": str(output_dir),
        "checkpoint": args.checkpoint,
        "model_cfg": args.model_cfg,
        "prompt_frame": prompt_frame,
        "prompt_mask_path": str(prompt_mask_path),
        "prompt_box_xyxy": box,
        "saved_mask_count": len(saved_frames),
    }
    with (output_dir / "sam2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
