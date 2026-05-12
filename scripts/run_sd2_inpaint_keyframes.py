#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image


@dataclass
class RunItem:
    frame_index: int
    image_path: str
    mask_path: str
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight image-level diffusion inpainting baseline on Part 3 keyframes.")
    parser.add_argument("--workspace", required=True, help="Part 3 workspace containing keyframes/ and masks/.")
    parser.add_argument("--model-id", required=True, help="Inpainting checkpoint id or local path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--prompt", default="Remove the walking person and reconstruct a clean, realistic background.")
    parser.add_argument(
        "--negative-prompt",
        default="person, human, body, duplicate object, blur, distortion, warped architecture, broken pavement, artifacts",
    )
    parser.add_argument("--height", type=int, default=512, help="Inference height before resizing back to the source frame size.")
    parser.add_argument("--width", type=int, default=512, help="Inference width before resizing back to the source frame size.")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0, help="Kept for reporting consistency; not all inpainting pipelines expose it.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--keyframes", default=None, help="Comma-separated frame indices. Omit to use all frames in part3_plan.json.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _parse_keyframes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _read_workspace_keyframes(workspace: Path) -> list[int]:
    plan_path = workspace / "part3_plan.json"
    if plan_path.exists():
        payload = json.loads(plan_path.read_text())
        return [int(item) for item in payload["keyframes"]]

    frame_paths = sorted((workspace / "keyframes").glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No exported keyframes were found in {workspace / 'keyframes'}.")
    return [int(path.stem.split("_")[-1]) for path in frame_paths]


def _resolve_run_items(workspace: Path, keyframes: list[int], skip_existing: bool) -> list[RunItem]:
    items: list[RunItem] = []
    edited_dir = workspace / "edited_sd2"
    edited_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in keyframes:
        image_path = workspace / "keyframes" / f"frame_{frame_index:05d}.png"
        mask_path = workspace / "masks" / f"mask_{frame_index:05d}.png"
        output_path = edited_dir / f"edited_{frame_index:05d}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Missing keyframe image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing keyframe mask: {mask_path}")
        if skip_existing and output_path.exists():
            continue

        items.append(
            RunItem(
                frame_index=frame_index,
                image_path=str(image_path),
                mask_path=str(mask_path),
                output_path=str(output_path),
            )
        )
    return items


def _load_pipeline(model_id: str, device: torch.device) -> AutoPipelineForInpainting:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    tried: list[str] = []

    for variant_kwargs in (
        {"variant": "fp16", "use_safetensors": True},
        {"use_safetensors": True},
        {},
    ):
        try:
            pipeline = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs, **variant_kwargs)
            break
        except Exception as exc:  # pragma: no cover - fallback ladder depends on external checkpoint layout
            tried.append(f"{variant_kwargs}: {exc}")
    else:
        raise RuntimeError("Failed to load inpainting pipeline. Attempts:\n" + "\n".join(tried))

    pipeline.enable_attention_slicing()
    if device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    return pipeline


def _prepare_inputs(image_path: str, mask_path: str, width: int, height: int) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    original_size = image.size

    resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
    resized_mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return resized_image, resized_mask, original_size


def run_items(args: argparse.Namespace, items: Iterable[RunItem]) -> dict:
    workspace = Path(args.workspace).resolve()
    device = torch.device(args.device)
    pipeline = _load_pipeline(args.model_id, device)
    generated = []

    for item in items:
        image, mask, original_size = _prepare_inputs(item.image_path, item.mask_path, args.width, args.height)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        result = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        output_path = Path(item.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        generated.append(asdict(item))

    summary = {
        "workspace": str(workspace),
        "model_id": args.model_id,
        "device": str(device),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "seed": args.seed,
        "generated_count": len(generated),
        "generated": generated,
    }
    summary_path = workspace / "sd2_inpaint_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    keyframes = _parse_keyframes(args.keyframes) or _read_workspace_keyframes(workspace)
    items = _resolve_run_items(workspace, keyframes, skip_existing=args.skip_existing)
    if not items:
        result = {"workspace": str(workspace), "message": "No keyframes needed processing.", "generated_count": 0}
        print(json.dumps(result, indent=2))
        return

    result = run_items(args, items)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
