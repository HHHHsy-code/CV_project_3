# Project 3: Video Object Removal & Inpainting

This repository implements a course-ready workflow for AIAA 3201 Project 3:

`dynamic object masking -> mask refinement -> video inpainting -> evaluation -> visualization -> report`

The current codebase is optimized for fast iteration:

- `Part 1`: a reproducible classical baseline built around `YOLOv8-seg + sparse optical flow + temporal background borrowing + cv2.inpaint`
- `Part 2`: adapter hooks for `SAM 2 + ProPainter`
- `Part 3`: a failure-case extension built around `single-prompt SAM 2 refinement`, `multi-prompt SAM 2 ablation`, and a `lightweight SD2 keyframe baseline`

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
report/                Result figures and paper assets
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
- Helper adapters and command templates for:
  - `SAM 2`
  - `ProPainter`
  - keyframe-level diffusion inpainting experiments

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

Run Part 1 directly on a frame directory such as the course sample data:

```bash
source .venv/bin/activate
python3 scripts/project3.py part1 \
  --config configs/project3.yaml \
  --frames-dir Project3/bmx-trees \
  --fps 24 \
  --experiment bmx_trees_part1
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
  --pred outputs/part1/bmx_trees_part1/masks \
  --gt Project3/bmx-trees_mask \
  --output metrics/bmx_trees_mask_metrics.json
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

Create a Part 1 vs Part 2 figure grid after running ProPainter:

```bash
python3 scripts/project3.py compare-methods \
  --part1-dir outputs/part1/wild_corridor \
  --part2-video outputs/part2/wild_corridor_propainter/wild_corridor/inpaint_out.mp4 \
  --output outputs/comparisons/wild_corridor_part1_vs_part2.png
```

## Part 2 and Part 3 workflow

Recommended process:

1. Use the Part 1 pipeline to bootstrap object boxes and rough masks.
2. Export prompt boxes for `SAM 2`.
3. Save refined masks under `outputs/part2/<experiment>/masks`.
4. Run `ProPainter` and save restored frames/video under `outputs/part2/<experiment>/`.
5. For failure cases, apply Part 3 keyframe repair and document both the improvement and the remaining artifacts.

The concrete adapter commands are documented in `docs/pipeline_notes.md`.

## Part 3 advanced-model workflow

The final extension is intentionally focused on one failure case, `wild_video2`.
The validated Part 3 results in the current repo are:

1. **Single-prompt SAM 2 refinement** on `wild_video2`, followed by `ProPainter`
2. **Multi-prompt SAM 2 ablation** with union / majority fusion
3. **Lightweight SD2 keyframe inpainting baseline** as a supplemental image-level check

This means the current final story is:

- `SAM 2 + ProPainter` is the main improvement branch
- multi-prompt fusion is a negative / limited-gain ablation
- image-level diffusion is a supplemental experiment, not the main video pipeline

The repository also includes an optional `AttentiveEraser` helper branch, but it is not part of the final validated result set.

### Main branch: single-prompt `SAM 2 + ProPainter`

Run the mask-refinement script on a non-empty Part 1 seed frame:

```bash
python scripts/sam2_refine_with_seed_mask.py \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --seed-mask-dir outputs/part1/wild_video2_480p24_part1_gpu/masks \
  --output-dir outputs/part2/wild_video2_sam2_refined_masks \
  --checkpoint external/sam2/checkpoints/sam2.1_hiera_small.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --prompt-frame 75 \
  --margin 24 \
  --offload-video-to-cpu
```

Then summarize the refined masks:

```bash
python3 scripts/project3.py summarize-mask \
  --pred outputs/part2/wild_video2_sam2_refined_masks \
  --output outputs/part2/wild_video2_sam2_refined_masks_sam2_summary.json
```

And rerun `ProPainter` with the refined masks:

```bash
cd external/ProPainter
python inference_propainter.py \
  --video ../../data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask ../../outputs/part2/wild_video2_sam2_refined_masks \
  --output ../../outputs/part2/wild_video2_propainter_from_sam2 \
  --fp16 \
  --subvideo_length 80 \
  --save_fps 24 \
  --save_frames
```

Finally, build the comparison figure:

```bash
python3 scripts/project3.py compare-methods \
  --part1-dir outputs/part1/wild_video2_480p24_part1_gpu \
  --mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --part2-video outputs/part2/wild_video2_propainter_from_sam2/wild_video2_480p24/inpaint_out.mp4 \
  --output outputs/comparisons/wild_video2_part1_vs_sam2_part2.png
```

### Ablation branch: multi-prompt `SAM 2`

The current repo also supports a simple multi-seed ablation. In our final checked run, prompt frames `75`, `139`, and `152` were propagated independently and merged with:

- `union`
- `majority`

The helper CLI is:

```bash
python3 scripts/project3.py merge-mask-dirs \
  --mask-dir outputs/part3/wild_video2_sam2_seed075_masks \
  --mask-dir outputs/part3/wild_video2_sam2_seed139_masks \
  --mask-dir outputs/part3/wild_video2_sam2_seed152_masks \
  --output-dir outputs/part3/wild_video2_sam2_multi_majority_masks \
  --mode majority
```

In the checked final result, this branch did **not** materially improve over the single-prompt SAM 2 masks. Keep it as an ablation rather than the main result.

### Supplemental branch: lightweight SD2 keyframe inpainting

Create a failure-case workspace with a fixed set of keyframes:

```bash
python3 scripts/project3.py part3-prepare \
  --experiment-dir outputs/part3 \
  --name wild_video2_attentive_eraser \
  --keyframes 75,113,151,189 \
  --notes "Keyframe-level diffusion repair for wild_video2"
```

Export the original frame, the chosen mask, and the current best reference result:

```bash
python3 scripts/project3.py part3-export-keyframes \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --reference-video outputs/part2/wild_video2_propainter_from_sam2/wild_video2_480p24/inpaint_out.mp4 \
  --workspace outputs/part3/wild_video2_attentive_eraser
```

Run the lightweight image-level baseline:

```bash
python scripts/run_sd2_inpaint_keyframes.py \
  --workspace outputs/part3/wild_video2_attentive_eraser \
  --model-id stabilityai/stable-diffusion-2-inpainting \
  --device cuda:0 \
  --height 512 \
  --width 512 \
  --num-inference-steps 40 \
  --guidance-scale 7.5 \
  --seed 123
```

Generate the comparison grid:

```bash
python3 scripts/project3.py compare-keyframes \
  --original-dir outputs/part3/wild_video2_attentive_eraser/keyframes \
  --mask-dir outputs/part3/wild_video2_attentive_eraser/masks \
  --reference-dir outputs/part3/wild_video2_attentive_eraser/reference \
  --edited-dir outputs/part3/wild_video2_attentive_eraser/edited_sd2 \
  --keyframes 75,113,151,189 \
  --reference-title ProPainter \
  --edited-title SD2-Inpaint \
  --output outputs/part3/wild_video2_attentive_eraser/comparisons/wild_video2_sd2_inpaint_grid.png
```

In the current checked result, the SD2 baseline works, but it does **not** clearly outperform `ProPainter`. It is therefore best treated as a supplemental experiment rather than a main pipeline replacement.

### Optional heavy branch: `AttentiveEraser`

Only if you explicitly want to continue beyond the validated final result, the repo also includes an `AttentiveEraser` wrapper. This is an optional extension, not part of the current final conclusion.

Run `AttentiveEraser` through the local wrapper script. It mirrors the upstream `main.py` preprocessing and loops over the exported keyframes:

```bash
python scripts/run_attentive_eraser_keyframes.py \
  --workspace outputs/part3/wild_video2_attentive_eraser \
  --repo-dir external/AttentiveEraser \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --device cuda:0 \
  --height 1024 \
  --width 1024 \
  --strength 0.8 \
  --rm-guidance-scale 9 \
  --ss-steps 9 \
  --ss-scale 0.3 \
  --aas-start-step 0 \
  --aas-start-layer 34 \
  --aas-end-layer 70 \
  --num-inference-steps 50 \
  --guidance-scale 1 \
  --seed 123
```

For a smoke test, add `--keyframes 75` first and verify that `outputs/part3/wild_video2_attentive_eraser/edited/edited_00075.png` is produced before running the full batch.
