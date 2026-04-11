# Project 3: Video Object Removal & Inpainting

This repository implements a course-ready workflow for AIAA 3201 Project 3:

`dynamic object masking -> mask refinement -> video inpainting -> evaluation -> visualization -> report`

The current codebase is optimized for fast iteration:

- `Part 1`: a reproducible classical baseline built around `YOLOv8-seg + sparse optical flow + temporal background borrowing + cv2.inpaint`
- `Part 2`: adapter hooks for `SAM 2 + ProPainter`
- `Part 3`: a controlled extension path for keyframe-level generative inpainting on failure cases

## Repository layout

```text
configs/               Experiment configs
data/
  raw/                 Input videos and optional GT data
  frames/              Extracted frame sequences
docs/                  Execution checklist and presentation notes
metrics/               CSV / JSON evaluation outputs
outputs/
  part1/               Baseline masks, videos, figures
  part2/               SAM2 / ProPainter outputs
  part3/               Diffusion extension outputs
report/                Paper writing scaffold
scripts/               Thin entrypoint wrappers
src/project3/          Main implementation
```

## What is already implemented

- A configurable Part 1 pipeline that:
  - reads a video
  - runs instance segmentation with YOLOv8-seg
  - filters objects by motion magnitude using Lucas-Kanade sparse optical flow
  - refines masks with dilation and connected-component cleanup
  - restores the background with temporal median borrowing and `cv2.inpaint`
  - saves mask videos, restored videos, and representative figure grids
- Evaluation utilities for:
  - mask quality: `JM`, `JR`, precision, recall
  - restored frame quality: `PSNR`, `SSIM`
- Helper adapters and command templates for:
  - `SAM 2`
  - `ProPainter`
  - keyframe-level diffusion inpainting experiments
- Course execution docs for:
  - week-by-week project progress
  - final submission checklist
  - 8-minute presentation outline

## Environment setup

Create a virtual environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to run only the baseline first, the minimal dependencies are:

```bash
pip install numpy opencv-python pyyaml matplotlib
```

For the full target workflow you will eventually also need:

- `ultralytics` for YOLOv8-seg
- `scikit-image` for SSIM
- local clones or installed environments for `SAM 2` and `ProPainter`

## Dataset preparation

Put videos under:

```text
data/raw/wild/
data/raw/bmx/
data/raw/tennis/
data/raw/davis/
```

Recommended naming:

```text
data/raw/wild/corridor.mp4
data/raw/bmx/bmx-trees.mp4
data/raw/tennis/tennis.mp4
```

If you have mask ground truth or clean targets, organize them as:

```text
data/raw/davis/<sequence>/masks/
data/raw/davis/<sequence>/gt_frames/
```

## Baseline usage

Run Part 1 on a single video:

```bash
source .venv/bin/activate
python3 scripts/project3.py part1 \
  --config configs/project3.yaml \
  --video data/raw/wild/corridor.mp4 \
  --experiment wild_corridor
```

This writes outputs into:

```text
outputs/part1/wild_corridor/
```

including:

- `masks/`
- `restored_frames/`
- `mask_video.mp4`
- `restored_video.mp4`
- `summary.json`
- `figures/comparison_grid.png`

## Evaluation usage

Evaluate masks:

```bash
python3 scripts/project3.py eval-mask \
  --pred outputs/part1/davis_example/masks \
  --gt data/raw/davis/example/masks \
  --output metrics/davis_example_mask_metrics.json
```

Evaluate restored frames:

```bash
python3 scripts/project3.py eval-video \
  --pred outputs/part1/davis_example/restored_frames \
  --gt data/raw/davis/example/gt_frames \
  --output metrics/davis_example_video_metrics.json
```

Create a figure grid:

```bash
python3 scripts/project3.py figures \
  --input outputs/part1/wild_corridor \
  --output outputs/part1/wild_corridor/figures/comparison_grid.png
```

## Part 2 and Part 3 workflow

The repository does not vendor third-party SOTA code. Instead, it provides:

- a standard place to store prompts and outputs
- config-driven wrappers for external tools
- a unified experiment naming convention

Recommended process:

1. Use the Part 1 pipeline to bootstrap object boxes and rough masks.
2. Export prompt boxes for `SAM 2`.
3. Save refined masks under `outputs/part2/<experiment>/masks`.
4. Run `ProPainter` and save restored frames/video under `outputs/part2/<experiment>/`.
5. For failure cases, apply Part 3 keyframe repair and document both the improvement and the remaining artifacts.

The concrete adapter commands are documented in `docs/pipeline_notes.md`.

## Suggested experiments

- `wild corridor`: fixed camera, pedestrians crossing the scene
- `bmx-trees`: validate removal under repeated motion and thin structures
- `tennis`: validate removal on faster motion and more occlusion
- `DAVIS subset`: compute metrics and support claims for the report

## Deliverables checklist

- `Part 1` fully runs on all mandatory datasets
- `Part 2` runs on all mandatory datasets
- at least one `Part 3` failure-case extension
- processed videos for `wild`, `bmx-trees`, `tennis`
- 6-8 page CVPR-style report
- public GitHub repo
- arXiv upload

See `docs/submission_checklist.md` for the operational checklist.
