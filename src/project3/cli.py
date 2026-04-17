from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .config import load_yaml_config
from .eval_metrics import evaluate_frame_dir, evaluate_mask_dir, summarize_mask_dir
from .io_utils import write_json
from .part1 import BaselineVideoObjectRemoval
from .part2 import Part2Adapters
from .part3 import prepare_failure_case_workspace
from .visualization import generate_comparison_grid, generate_method_comparison_grid


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

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
