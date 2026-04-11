# Repository Guide

This document explains what each file and directory is for, what is already usable, and what you should do next when running on a server with GPU access.

## 1. Top-level files

### `README.md`

The main entry document for the repository.

Use it for:

- overall project scope
- repository layout
- dependency setup
- baseline run commands
- evaluation commands
- expected outputs

If someone opens the repo for the first time, they should start here.

### `requirements.txt`

Python dependencies for the project.

Main packages:

- `opencv-python`: frame IO, optical flow, mask processing, inpainting
- `ultralytics`: YOLOv8-seg baseline detector/segmenter
- `PyYAML`: config loading
- `scikit-image`: SSIM
- `numpy`, `scipy`, `matplotlib`: core utilities

### `.gitignore`

Prevents large datasets, generated frames, experiment outputs, and cache files from being committed by accident.

This is important because videos, masks, and restored outputs will quickly become very large.

## 2. Configs

### `configs/project3.yaml`

The default experiment config.

This is the first file to edit when you want to change:

- YOLO model name
- target classes
- motion threshold
- mask dilation parameters
- temporal fill window
- Part 2 external repo locations

In practice, this is your experiment control panel.

## 3. Code entrypoint

### `scripts/project3.py`

Thin command-line entry script.

You will use this file to run:

- `part1`
- `eval-mask`
- `eval-video`
- `figures`
- `part2-prepare`
- `part3-prepare`

It exists so you do not need to manually manage `PYTHONPATH`.

## 4. Source code

Everything under `src/project3/` is the implementation.

### `src/project3/config.py`

Loads config files.

It supports normal YAML, and also has a lightweight fallback parser so the repository can still read `configs/project3.yaml` even if `PyYAML` is missing locally.

### `src/project3/io_utils.py`

Small shared file helpers.

Used for:

- making directories
- listing image files
- writing JSON summaries

### `src/project3/video_utils.py`

Low-level video helpers.

Used for:

- reading input videos into frames
- saving frames to folders
- writing processed videos back to disk

### `src/project3/part1.py`

The main Part 1 baseline implementation.

This is the most important code file right now.

It does:

1. load a video
2. run YOLOv8-seg on each frame
3. keep only dynamic objects using sparse optical flow
4. refine masks with dilation and connected-component cleanup
5. restore masked regions using temporal median borrowing
6. run `cv2.inpaint` as fallback
7. save masks, restored frames, restored video, and summary metadata

This file is your first runnable method for the course project.

### `src/project3/eval_metrics.py`

Evaluation code.

It computes:

- mask metrics: `JM`, `JR`, precision
- restoration metrics: `PSNR`, `SSIM`

Use this once you have GT masks or GT clean frames, especially on DAVIS or any GT-backed subset.

### `src/project3/visualization.py`

Creates paper/slide friendly comparison figures.

The generated grid shows:

- original frame
- predicted mask
- restored frame

This file is mainly for report figures and presentation slides.

### `src/project3/part2.py`

Part 2 adapter layer.

Important point:

This file does **not** implement SAM 2 or ProPainter itself.

Instead, it standardizes:

- output directories
- metadata files
- recommended command templates

This keeps your repo clean while still allowing you to plug in strong external methods.

### `src/project3/part3.py`

Part 3 helper for failure-case experiments.

It prepares a structured workspace for:

- selected failure cases
- chosen keyframes
- masks
- edited frames
- comparison outputs

Use this when ProPainter fails and you want to test keyframe-level diffusion repair.

### `src/project3/cli.py`

The actual command router behind `scripts/project3.py`.

It wires together all commands:

- run baseline
- evaluate masks
- evaluate restored frames
- generate figures
- prepare Part 2 workspace
- prepare Part 3 workspace

### `src/project3/__init__.py`

Package marker.

This just makes `project3` importable as a Python module.

## 5. Documentation

### `docs/submission_checklist.md`

Operational checklist for the final delivery.

Use it near the end of the project to make sure nothing important is missing.

### `docs/presentation_outline.md`

Suggested structure for the 8-minute progress presentation.

Use it when preparing Week 11/12 progress reporting.

### `docs/pipeline_notes.md`

Practical notes for how Part 1, Part 2, and Part 3 fit together.

This is more execution-oriented than the README.

### `docs/repo_guide.md`

This file.

Use it as the internal explanation of what the repository contains and how to operate it.

## 6. Report files

### `report/outline.md`

Plain-language report structure.

Use this to decide what content should go into each paper section.

### `report/main.tex`

LaTeX report scaffold.

Right now it is a lightweight starter template, not yet the official CVPR template.

Later, you should replace or merge it with the actual CVPR template required by the course.

### `report/references.bib`

Placeholder bibliography file.

You must replace it with the actual references from the project PDF, and make sure all required papers are cited in the final paper.

## 7. Data and output directories

### `data/raw/`

Put original videos and GT-backed data here.

Recommended usage:

- `data/raw/wild/`
- `data/raw/bmx/`
- `data/raw/tennis/`
- `data/raw/davis/`

### `data/frames/`

Reserved for extracted frame sequences if you want to preprocess videos manually.

### `outputs/part1/`

Stores baseline outputs.

Expected per experiment:

- input frames
- masks
- restored frames
- restored video
- summary JSON
- comparison figure

### `outputs/part2/`

Stores Part 2 experiment outputs such as:

- refined masks
- ProPainter input/output
- Part 2 metadata

### `outputs/part3/`

Stores failure-case extension experiments and keyframe editing outputs.

### `metrics/`

Stores exported metric JSON files.

## 8. What is already ready vs what still depends on the server

### Already ready now

- repository structure
- baseline pipeline code
- metric code
- visualization code
- Part 2 / Part 3 experiment preparation helpers
- report and presentation scaffolds

### Not runnable locally yet in your current environment

- actual YOLOv8-seg inference, because local dependencies are not installed
- SAM 2 / ProPainter, because they require server-side environment, weights, and GPU
- full experiments, because the actual datasets are not yet placed into the repo

## 9. What you should do next on the GPU server

Recommended order:

### Step 1: sync this repository to the server

Clone or upload the current repository to the server workspace.

### Step 2: create a Python environment

Recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the server has CUDA-ready PyTorch already, keep that environment consistent with `ultralytics`, SAM 2, and ProPainter requirements.

### Step 3: place the mandatory datasets

Put your data into:

```text
data/raw/wild/
data/raw/bmx/
data/raw/tennis/
```

Also add DAVIS if you want stronger quantitative results.

### Step 4: run Part 1 first

Example:

```bash
python3 scripts/project3.py part1 \
  --config configs/project3.yaml \
  --video data/raw/wild/corridor.mp4 \
  --experiment wild_corridor
```

Do this first because:

- it gives you an end-to-end baseline quickly
- it produces early visual results for the presentation
- it creates a comparison point for Part 2

### Step 5: inspect outputs manually

Check:

- whether masks cover only moving objects
- whether thin structures are missed
- whether the restored background is blurry
- whether temporal flicker is obvious

This manual inspection matters because the report needs qualitative analysis, not only metrics.

### Step 6: prepare Part 2 workspace

Example:

```bash
python3 scripts/project3.py part2-prepare \
  --config configs/project3.yaml \
  --video data/raw/wild/corridor.mp4 \
  --experiment wild_corridor
```

Then:

- clone/install `SAM 2`
- clone/install `ProPainter`
- edit `configs/project3.yaml` with the real repo paths and checkpoints
- run the generated command templates

### Step 7: run quantitative evaluation

Once you have GT-backed data, compute:

- `JM`, `JR` for masks
- `PSNR`, `SSIM` for restored frames

Use:

```bash
python3 scripts/project3.py eval-mask ...
python3 scripts/project3.py eval-video ...
```

### Step 8: start Part 3 only after Part 2 is stable

Do not start diffusion experiments too early.

Only do Part 3 after:

- Part 1 works
- Part 2 works
- you have identified real failure cases

That keeps the scope under control.

## 10. Recommended immediate task split for your team

### Person A

- set up server environment
- run baseline on wild / bmx / tennis
- inspect masks and restored outputs

### Person B

- collect datasets and organize folder structure
- prepare report references and CVPR template
- read SAM 2 / ProPainter setup instructions

After that, both of you can merge into the Part 2 stage.

## 11. Practical caution

- Do not spend too much time polishing Part 3 before Part 1 and Part 2 are solid.
- Do not wait until the end to collect qualitative figures.
- Do not postpone reference management; the project requires all listed papers to be cited.
- Do not assume server paths will match local paths; update `configs/project3.yaml` on the server.
