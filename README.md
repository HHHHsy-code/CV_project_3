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
data/raw/davis/<sequence>/gt_frames/   # only if you build a synthetic benchmark with clean targets
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

## Part 3 advanced-model workflow

The final extension is intentionally focused on one failure case, `wild_video2`.

We split Part 3 into two small, explainable branches instead of building a large experiment matrix:

1. **Mask branch**: `Track-Anything + ProPainter`
2. **Restoration branch**: `best current mask + AttentiveEraser` on a few hard keyframes

The repository now includes helper commands for:

- bootstrapping a prompt box from an existing mask
- preparing a standard `Track-Anything` workspace
- exporting failure-case keyframes for image-level diffusion editing
- batch-running `AttentiveEraser` on those exported keyframes
- generating comparison figures for the keyframe branch

### Prepare the Track-Anything branch

Generate a box prompt from an existing seed mask and create a standard workspace:

```bash
source .venv/bin/activate
python3 scripts/project3.py part3-track-prepare \
  --config configs/project3.yaml \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --seed-mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --experiment wild_video2_track_anything \
  --frame-index 113 \
  --margin 24
```

This creates:

```text
outputs/part3/wild_video2_track_anything/
  masks/
  prompts/
  propainter_output/
  recommended_commands.json
```

The prompt JSON stores:

- `frame_index`: zero-based frame index in the project code
- `prompt_box_xyxy`: a suggested region of interest
- `prompt_point_xy`: the center point of that region

The official `Track-Anything` Gradio app is click-based rather than box-based, so in practice:

- move the frame slider to `frame_index + 1`
- use `prompt_point_xy` as the first positive click
- add one or two more positive clicks inside `prompt_box_xyxy`
- add a negative click if the mask spills into the background

### Summarize the new masks

If you start the official Gradio app, launch it with mask saving enabled:

```bash
python app.py --device cuda:0 --mask_save True
```

The upstream app stores tracked masks as `.npy` files under `result/mask/<video_stem>/`.
Convert them into project-standard PNG masks with:

```bash
python3 scripts/project3.py part3-convert-track-masks \
  --source-dir external/Track-Anything/result/mask/wild_video2_480p24 \
  --output-dir outputs/part3/wild_video2_track_anything/masks
```

Then run:

```bash
python3 scripts/project3.py summarize-mask \
  --pred outputs/part3/wild_video2_track_anything/masks \
  --output metrics/wild_video2_track_anything_mask_summary.json
```

Compare that JSON against:

- `metrics/wild_video2_part1_mask_summary.json`
- `metrics/wild_video2_sam2_mask_summary.json`

### Prepare the keyframe-level diffusion branch

Create a failure-case workspace with a fixed set of keyframes:

```bash
python3 scripts/project3.py part3-prepare \
  --experiment-dir outputs/part3 \
  --name wild_video2_attentive_eraser \
  --keyframes 75,113,151,189 \
  --notes "Keyframe-level diffusion repair for wild_video2"
```

Export the original frame, the chosen mask, and the current best reference result (for example, `SAM 2 + ProPainter` or `Track-Anything + ProPainter`):

```bash
python3 scripts/project3.py part3-export-keyframes \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --reference-video outputs/part2/wild_video2_propainter_from_sam2/wild_video2_480p24/inpaint_out.mp4 \
  --workspace outputs/part3/wild_video2_attentive_eraser
```

This writes:

```text
outputs/part3/wild_video2_attentive_eraser/
  keyframes/
  masks/
  reference/
  edited/
  comparisons/
```

### Start with a lightweight image-level diffusion baseline

Before downloading a large SDXL checkpoint, run a smaller image-level inpainting baseline first. The local script below uses a standard Diffusers inpainting pipeline and writes results into `edited_sd2/`.

```bash
python scripts/run_sd2_inpaint_keyframes.py \
  --workspace outputs/part3/wild_video2_attentive_eraser \
  --model-id stabilityai/stable-diffusion-2-inpainting \
  --device cuda:0 \
  --keyframes 75 \
  --height 512 \
  --width 512 \
  --num-inference-steps 40 \
  --guidance-scale 7.5 \
  --seed 123
```

If the smoke test succeeds, run the full four-frame batch:

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

This writes:

```text
outputs/part3/wild_video2_attentive_eraser/edited_sd2/
  edited_00075.png
  edited_00113.png
  edited_00151.png
  edited_00189.png
```

After that, generate a four-column comparison grid:

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

Use this lightweight baseline to decide whether a heavier SDXL-based branch is worth the download and deployment cost.

### Escalate to `AttentiveEraser` only if needed

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

### Build the final keyframe comparison figure

After the diffusion outputs are ready, generate a four-column comparison grid:

```bash
python3 scripts/project3.py compare-keyframes \
  --original-dir outputs/part3/wild_video2_attentive_eraser/keyframes \
  --mask-dir outputs/part3/wild_video2_attentive_eraser/masks \
  --reference-dir outputs/part3/wild_video2_attentive_eraser/reference \
  --edited-dir outputs/part3/wild_video2_attentive_eraser/edited \
  --keyframes 75,113,151,189 \
  --reference-title ProPainter \
  --edited-title AttentiveEraser \
  --output outputs/part3/wild_video2_attentive_eraser/comparisons/wild_video2_diffusion_keyframes.png
```

## Official upstream references

The current Part 3 recommendations are based on the official public repositories:

- `Track-Anything`: [gaomingqi/Track-Anything](https://github.com/gaomingqi/Track-Anything)
- `AttentiveEraser`: [Alibaba-YuFeng/AttentiveEraser](https://github.com/Alibaba-YuFeng/AttentiveEraser)

These repos evolve independently. Treat the generated command templates as starting points and align them with the entrypoints in your local clones.

## Suggested experiments

- `wild corridor`: fixed camera, pedestrians crossing the scene
- `bmx-trees`: validate removal under repeated motion and thin structures
- `tennis`: validate removal on faster motion and more occlusion
- `DAVIS subset`: strengthen mask evaluation and support claims for the report

## Deliverables checklist

- `Part 1` fully runs on all mandatory datasets
- `Part 2` runs on all mandatory datasets
- at least one `Part 3` failure-case extension
- processed videos for `wild`, `bmx-trees`, `tennis`
- 6-8 page CVPR-style report
- public GitHub repo
- arXiv upload

See `docs/submission_checklist.md` for the operational checklist.
