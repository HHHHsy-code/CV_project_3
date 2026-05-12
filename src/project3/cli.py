from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .config import load_yaml_config
from .eval_metrics import evaluate_frame_dir, evaluate_mask_dir, summarize_mask_dir
from .io_utils import write_json
from .part1 import BaselineVideoObjectRemoval
from .part2 import Part2Adapters
from .part3 import (
    Part3Adapters,
    build_box_prompt_from_mask,
    copy_edited_results,
    convert_track_anything_npy_masks,
    export_failure_case_assets,
    merge_mask_dirs,
    prepare_failure_case_workspace,
)
from .visualization import (
    generate_comparison_grid,
    generate_keyframe_comparison_grid,
    generate_method_comparison_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project 3 video object removal toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    part1 = subparsers.add_parser("part1", help="Run the baseline Part 1 pipeline")
    part1.add_argument("--config", required=True)
    input_group = part1.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video")
    input_group.add_argument("--frames-dir")
    part1.add_argument("--experiment", required=True)
    part1.add_argument("--output-root", default="outputs/part1")
    part1.add_argument("--fps", type=float, default=24.0, help="Used only when --frames-dir is provided")

    eval_mask = subparsers.add_parser("eval-mask", help="Evaluate predicted masks against GT masks")
    eval_mask.add_argument("--pred", required=True)
    eval_mask.add_argument("--gt", required=True)
    eval_mask.add_argument("--output", required=True)

    eval_video = subparsers.add_parser("eval-video", help="Evaluate restored frames against GT frames")
    eval_video.add_argument("--pred", required=True)
    eval_video.add_argument("--gt", required=True)
    eval_video.add_argument("--output", required=True)

    summarize_mask = subparsers.add_parser("summarize-mask", help="Summarize temporal coverage and area of a mask dir")
    summarize_mask.add_argument("--pred", required=True)
    summarize_mask.add_argument("--output", required=True)

    figures = subparsers.add_parser("figures", help="Generate comparison figures")
    figures.add_argument("--input", required=True)
    figures.add_argument("--output", required=True)
    figures.add_argument("--samples", type=int, default=6)

    compare_methods = subparsers.add_parser(
        "compare-methods", help="Generate Original/Mask/Part1/Part2 comparison figures"
    )
    compare_methods.add_argument("--part1-dir", required=True)
    compare_methods.add_argument("--part2-video", required=True)
    compare_methods.add_argument("--output", required=True)
    compare_methods.add_argument("--samples", type=int, default=6)
    compare_methods.add_argument("--mask-dir", default=None, help="Optional mask directory override")

    part2 = subparsers.add_parser("part2-prepare", help="Prepare a Part 2 workspace and command templates")
    part2.add_argument("--config", required=True)
    part2.add_argument("--video", required=True)
    part2.add_argument("--experiment", required=True)
    part2.add_argument("--output-root", default="outputs/part2")

    part3 = subparsers.add_parser("part3-prepare", help="Prepare a failure-case workspace for keyframe editing")
    part3.add_argument("--experiment-dir", required=True)
    part3.add_argument("--name", required=True)
    part3.add_argument("--keyframes", required=True, help="Comma-separated frame indices")
    part3.add_argument("--notes", default="")

    part3_track = subparsers.add_parser(
        "part3-track-prepare", help="Prepare a Track-Anything + ProPainter experiment for a failure case"
    )
    part3_track.add_argument("--config", required=True)
    part3_track.add_argument("--video", required=True)
    part3_track.add_argument("--seed-mask-dir", required=True)
    part3_track.add_argument("--experiment", required=True)
    part3_track.add_argument("--frame-index", type=int, default=113)
    part3_track.add_argument("--margin", type=int, default=24)
    part3_track.add_argument("--output-root", default="outputs/part3")

    part3_prompt = subparsers.add_parser(
        "part3-bootstrap-prompt", help="Create a prompt box JSON from an existing mask directory"
    )
    part3_prompt.add_argument("--mask-dir", required=True)
    part3_prompt.add_argument("--frame-index", type=int, required=True)
    part3_prompt.add_argument("--output", required=True)
    part3_prompt.add_argument("--margin", type=int, default=24)

    part3_export = subparsers.add_parser(
        "part3-export-keyframes", help="Export original, mask, and reference frames for keyframe editing"
    )
    part3_export.add_argument("--video", required=True)
    part3_export.add_argument("--mask-dir", required=True)
    part3_export.add_argument("--reference-video", required=True)
    part3_export.add_argument("--workspace", required=True)
    part3_export.add_argument("--keyframes", default=None, help="Comma-separated frame indices; defaults to workspace plan")

    part3_collect = subparsers.add_parser(
        "part3-collect-edits", help="Copy edited keyframes into the standard Part 3 workspace naming"
    )
    part3_collect.add_argument("--edited-dir", required=True)
    part3_collect.add_argument("--destination-dir", required=True)
    part3_collect.add_argument("--keyframes", required=True, help="Comma-separated frame indices")

    part3_convert = subparsers.add_parser(
        "part3-convert-track-masks", help="Convert Track-Anything .npy masks into project-standard PNG masks"
    )
    part3_convert.add_argument("--source-dir", required=True)
    part3_convert.add_argument("--output-dir", required=True)

    merge_masks = subparsers.add_parser(
        "merge-mask-dirs", help="Merge multiple binary mask directories into one"
    )
    merge_masks.add_argument(
        "--mask-dir",
        dest="mask_dirs",
        action="append",
        required=True,
        help="Mask directory to merge; pass this flag multiple times",
    )
    merge_masks.add_argument("--output-dir", required=True)
    merge_masks.add_argument("--mode", choices=["union", "intersection", "majority"], default="union")

    compare_keyframes = subparsers.add_parser(
        "compare-keyframes", help="Generate Original/Mask/Reference/Edited grids for keyframe-level Part 3 results"
    )
    compare_keyframes.add_argument("--original-dir", required=True)
    compare_keyframes.add_argument("--mask-dir", required=True)
    compare_keyframes.add_argument("--reference-dir", required=True)
    compare_keyframes.add_argument("--edited-dir", required=True)
    compare_keyframes.add_argument("--output", required=True)
    compare_keyframes.add_argument("--keyframes", required=True, help="Comma-separated frame indices")
    compare_keyframes.add_argument("--reference-title", default="ProPainter")
    compare_keyframes.add_argument("--edited-title", default="AttentiveEraser")

    return parser


def _write_or_print(result: dict, output: Optional[str] = None) -> None:
    if output is not None:
        write_json(result, output)
    else:
        print(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "part1":
        config = load_yaml_config(args.config)
        pipeline = BaselineVideoObjectRemoval(config)
        output_dir = Path(args.output_root) / args.experiment
        input_path = args.video or args.frames_dir
        result = pipeline.run(input_path, output_dir, fps=args.fps)
        figures_path = output_dir / "figures" / "comparison_grid.png"
        generate_comparison_grid(output_dir, figures_path, samples=config["part1"]["visualization"]["representative_frames"])
        print(result)
        return

    if args.command == "eval-mask":
        result = evaluate_mask_dir(args.pred, args.gt, args.output)
        print(result)
        return

    if args.command == "eval-video":
        result = evaluate_frame_dir(args.pred, args.gt, args.output)
        print(result)
        return

    if args.command == "summarize-mask":
        result = summarize_mask_dir(args.pred, args.output)
        print(result)
        return

    if args.command == "figures":
        output = generate_comparison_grid(args.input, args.output, samples=args.samples)
        print({"figure": str(output)})
        return

    if args.command == "compare-methods":
        output = generate_method_comparison_grid(
            args.part1_dir,
            args.part2_video,
            args.output,
            samples=args.samples,
            mask_dir=args.mask_dir,
        )
        print({"figure": str(output)})
        return

    if args.command == "part2-prepare":
        config = load_yaml_config(args.config)
        adapters = Part2Adapters(config)
        experiment_dir = Path(args.output_root) / args.experiment
        adapters.prepare_experiment(experiment_dir)
        commands = adapters.recommended_commands(args.video, experiment_dir)
        write_json(commands, experiment_dir / "recommended_commands.json")
        print(commands)
        return

    if args.command == "part3-prepare":
        keyframes = [int(item.strip()) for item in args.keyframes.split(",") if item.strip()]
        result = prepare_failure_case_workspace(args.experiment_dir, args.name, keyframes, args.notes)
        print(result)
        return

    if args.command == "part3-track-prepare":
        config = load_yaml_config(args.config)
        adapters = Part3Adapters(config)
        experiment_dir = Path(args.output_root) / args.experiment
        prompt = build_box_prompt_from_mask(args.seed_mask_dir, args.frame_index, margin=args.margin)
        paths = adapters.prepare_track_anything_experiment(experiment_dir, prompt)
        prompt_path = Path(paths["prompts"]) / f"prompt_{args.frame_index:05d}.json"
        write_json(prompt, prompt_path)
        commands = adapters.recommended_track_anything_commands(args.video, experiment_dir, prompt_path)
        write_json(commands, experiment_dir / "recommended_commands.json")
        print({"paths": paths, "prompt": prompt, "commands": commands})
        return

    if args.command == "part3-bootstrap-prompt":
        result = build_box_prompt_from_mask(args.mask_dir, args.frame_index, args.output, margin=args.margin)
        print(result)
        return

    if args.command == "part3-export-keyframes":
        keyframes = None
        if args.keyframes:
            keyframes = [int(item.strip()) for item in args.keyframes.split(",") if item.strip()]
        else:
            plan_path = Path(args.workspace) / "part3_plan.json"
            if not plan_path.exists():
                raise FileNotFoundError("No --keyframes supplied and workspace/part3_plan.json was not found.")
            plan = load_yaml_config(plan_path)
            keyframes = [int(item) for item in plan["keyframes"]]

        result = export_failure_case_assets(
            args.video,
            args.mask_dir,
            args.reference_video,
            args.workspace,
            keyframes,
        )
        print(result)
        return

    if args.command == "part3-collect-edits":
        keyframes = [int(item.strip()) for item in args.keyframes.split(",") if item.strip()]
        result = copy_edited_results(args.edited_dir, args.destination_dir, keyframes)
        print(result)
        return

    if args.command == "part3-convert-track-masks":
        result = convert_track_anything_npy_masks(args.source_dir, args.output_dir)
        print(result)
        return

    if args.command == "merge-mask-dirs":
        result = merge_mask_dirs(args.mask_dirs, args.output_dir, mode=args.mode)
        print(result)
        return

    if args.command == "compare-keyframes":
        keyframes = [int(item.strip()) for item in args.keyframes.split(",") if item.strip()]
        output = generate_keyframe_comparison_grid(
            args.original_dir,
            args.mask_dir,
            args.reference_dir,
            args.edited_dir,
            args.output,
            keyframes,
            reference_title=args.reference_title,
            edited_title=args.edited_title,
        )
        print({"figure": str(output)})
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
