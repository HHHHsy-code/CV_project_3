#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
from torchvision.transforms.functional import gaussian_blur, to_tensor


@dataclass
class RunItem:
    frame_index: int
    image_path: str
    mask_path: str
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run AttentiveEraser on a Part 3 keyframe workspace.")
    parser.add_argument("--workspace", required=True, help="Part 3 workspace containing keyframes/ and masks/.")
    parser.add_argument("--repo-dir", required=True, help="Local AttentiveEraser repo clone.")
    parser.add_argument("--model-id", required=True, help="Base diffusion model id or local path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--prompt", default="", help="Prompt passed into the pipeline. Defaults to empty string.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--rm-guidance-scale", type=float, default=9.0)
    parser.add_argument("--ss-steps", type=int, default=9)
    parser.add_argument("--ss-scale", type=float, default=0.3)
    parser.add_argument("--aas-start-step", type=int, default=0)
    parser.add_argument("--aas-start-layer", type=int, default=34)
    parser.add_argument("--aas-end-layer", type=int, default=70)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--keyframes",
        default=None,
        help="Comma-separated frame indices to run. Omit to process every exported keyframe in the workspace.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist in workspace/edited.",
    )
    return parser.parse_args()


def _parse_keyframes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",")]
    return [int(item) for item in values if item]


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
    edited_dir = workspace / "edited"
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


def preprocess_image(image_path: str | Path, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    image = to_tensor(load_image(image_path))
    image = image.unsqueeze(0).float() * 2 - 1
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (height, width), mode="bilinear", align_corners=False)
    return image.to(dtype=dtype, device=device)


def preprocess_mask(mask_path: str | Path, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = to_tensor(load_image(mask_path, convert_method=lambda img: img.convert("L")))
    mask = mask.unsqueeze(0).float()
    mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    return mask.to(dtype=dtype, device=device)


def build_pipeline(repo_dir: Path, model_id: str, dtype: torch.dtype, device: torch.device) -> DiffusionPipeline:
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    custom_pipeline = repo_dir / "pipelines" / "pipeline_stable_diffusion_xl_attentive_eraser.py"
    if not custom_pipeline.exists():
        raise FileNotFoundError(f"AttentiveEraser custom pipeline was not found: {custom_pipeline}")

    load_kwargs = {
        "custom_pipeline": str(custom_pipeline),
        "scheduler": scheduler,
        "torch_dtype": dtype,
    }
    if dtype == torch.float16:
        load_kwargs["variant"] = "fp16"
        load_kwargs["use_safetensors"] = True

    pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    pipeline.enable_attention_slicing()
    if device.type == "cuda":
        pipeline = pipeline.to(device)
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    return pipeline


def run_items(args: argparse.Namespace, items: Iterable[RunItem]) -> dict:
    repo_dir = Path(args.repo_dir).resolve()
    workspace = Path(args.workspace).resolve()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipeline = build_pipeline(repo_dir, args.model_id, dtype, device)
    generated = []

    for item in items:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        source_image = preprocess_image(item.image_path, args.height, args.width, dtype, device)
        mask_image = preprocess_mask(item.mask_path, args.height, args.width, dtype, device)

        result = pipeline(
            prompt=args.prompt,
            image=source_image,
            mask_image=mask_image,
            height=args.height,
            width=args.width,
            AAS=True,
            strength=args.strength,
            rm_guidance_scale=args.rm_guidance_scale,
            ss_steps=args.ss_steps,
            ss_scale=args.ss_scale,
            AAS_start_step=args.aas_start_step,
            AAS_start_layer=args.aas_start_layer,
            AAS_end_layer=args.aas_end_layer,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            guidance_scale=args.guidance_scale,
        ).images[0]
        if not isinstance(result, Image.Image):
            raise RuntimeError(f"Unexpected pipeline output type for frame {item.frame_index}: {type(result)!r}")

        output_path = Path(item.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        generated.append(asdict(item))

    summary = {
        "workspace": str(workspace),
        "repo_dir": str(repo_dir),
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype),
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "strength": args.strength,
        "rm_guidance_scale": args.rm_guidance_scale,
        "ss_steps": args.ss_steps,
        "ss_scale": args.ss_scale,
        "aas_start_step": args.aas_start_step,
        "aas_start_layer": args.aas_start_layer,
        "aas_end_layer": args.aas_end_layer,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "generated_count": len(generated),
        "generated": generated,
    }
    summary_path = workspace / "attentive_eraser_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    keyframes = _parse_keyframes(args.keyframes) or _read_workspace_keyframes(workspace)
    items = _resolve_run_items(workspace, keyframes, skip_existing=args.skip_existing)
    if not items:
        result = {
            "workspace": str(workspace),
            "message": "No keyframes needed processing.",
            "generated_count": 0,
        }
        print(json.dumps(result, indent=2))
        return

    result = run_items(args, items)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
