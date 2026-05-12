# Part 3 Advanced Models on `wild_video2`

This note turns the instructor feedback into two focused experiments on a single failure case:

1. `Track-Anything + ProPainter` for better temporal masks
2. `AttentiveEraser` on four failure keyframes for stronger image-level restoration

The scope stays narrow on purpose. The goal is not to benchmark many models; the goal is to show whether more advanced tracking-aware masking and diffusion-based removal improve the same difficult sequence.

## Why this Part 3 is structured this way

The current evidence already supports two conclusions:

- `ProPainter` is stronger than the classical Part 1 restoration backend when the masks are fixed.
- `wild_video2` fails mainly because the person is not covered consistently across frames.

That means the next useful questions are:

- Can a tracking-aware mask model improve temporal coverage beyond the current `SAM 2` branch?
- If the mask is already reasonably good, can a stronger diffusion remover improve the hardest keyframes?

## Branch A: `Track-Anything + ProPainter`

### 1. Prepare the experiment

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py part3-track-prepare \
  --config configs/project3.yaml \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --seed-mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --experiment wild_video2_track_anything \
  --frame-index 113 \
  --margin 24
```

This command:

- creates `outputs/part3/wild_video2_track_anything/`
- extracts a `prompt_box_xyxy` from the chosen seed mask
- stores that prompt in `prompts/prompt_00113.json`
- writes command templates into `recommended_commands.json`

### 2. Run `Track-Anything` externally

Follow the official repository:

- [Track-Anything GitHub](https://github.com/gaomingqi/Track-Anything)

The official repo ships an interactive Gradio app (`app.py`). The UI is click-based, not box-based, so use the generated prompt JSON as a guide:

- prompt frame: `113` in zero-based indexing, so move the UI slider to `114`
- first positive click: `prompt_point_xy`
- additional positive clicks: inside `prompt_box_xyxy`
- negative click: only if the mask bleeds into the background

Start the app with mask saving enabled:

```bash
python app.py --device cuda:0 --mask_save True
```

The upstream app stores tracked masks as `.npy` files under:

```text
external/Track-Anything/result/mask/<video_stem>/
```

Convert those masks into project-standard PNG files with:

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py part3-convert-track-masks \
  --source-dir external/Track-Anything/result/mask/wild_video2_480p24 \
  --output-dir outputs/part3/wild_video2_track_anything/masks
```

The converted masks will land in:

```text
outputs/part3/wild_video2_track_anything/masks/
```

### 3. Run `ProPainter`

```bash
cd /data2/zguo315/CV_project_3/CV_project_3/external/ProPainter
conda activate /data2/zguo315/conda_envs/propainter

CUDA_VISIBLE_DEVICES=0 python inference_propainter.py \
  --video ../../data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask ../../outputs/part3/wild_video2_track_anything/masks \
  --output ../../outputs/part3/wild_video2_track_anything/propainter_output \
  --fp16 \
  --subvideo_length 80 \
  --save_fps 24 \
  --save_frames
```

### 4. Evaluate the mask quality

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py summarize-mask \
  --pred outputs/part3/wild_video2_track_anything/masks \
  --output metrics/wild_video2_track_anything_mask_summary.json
```

Compare against:

- `metrics/wild_video2_part1_mask_summary.json`
- `metrics/wild_video2_sam2_mask_summary.json`

Default acceptance target:

- `temporal_coverage >= 0.642` or at least no worse than the `SAM 2` result
- keyframes `75`, `113`, `151`, `189` all cover the walking person
- `Track-Anything + ProPainter` is at least not worse than `SAM 2 + ProPainter`

### 5. Build the final comparison figure

You can reuse the existing Part 1 vs advanced-method figure builder:

```bash
python3 scripts/project3.py compare-methods \
  --part1-dir outputs/part1/wild_video2_480p24_part1_gpu \
  --mask-dir outputs/part3/wild_video2_track_anything/masks \
  --part2-video outputs/part3/wild_video2_track_anything/propainter_output/wild_video2_480p24/inpaint_out.mp4 \
  --output outputs/part3/wild_video2_track_anything/comparisons/wild_video2_part1_vs_track_anything_part2.png
```

## Branch B: `AttentiveEraser` on failure keyframes

### 1. Prepare the failure-case workspace

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py part3-prepare \
  --experiment-dir outputs/part3 \
  --name wild_video2_attentive_eraser \
  --keyframes 75,113,151,189 \
  --notes "Keyframe-level diffusion repair for wild_video2"
```

### 2. Export the keyframes

Use the best currently available mask branch as input. If `Track-Anything` is better than `SAM 2`, switch the `--mask-dir` and `--reference-video` below.

```bash
python3 scripts/project3.py part3-export-keyframes \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --reference-video outputs/part2/wild_video2_propainter_from_sam2/wild_video2_480p24/inpaint_out.mp4 \
  --workspace outputs/part3/wild_video2_attentive_eraser
```

This writes:

- original frames into `keyframes/`
- masks into `masks/`
- current best reference frames into `reference/`

### 3. Start with a lightweight diffusion baseline

Before downloading a large SDXL checkpoint, first test whether a smaller image-level inpainting model is already competitive on the four keyframes.

Smoke test on frame `75`:

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

Then run the full four-frame batch:

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

The script writes:

```text
outputs/part3/wild_video2_attentive_eraser/edited_sd2/
  edited_00075.png
  edited_00113.png
  edited_00151.png
  edited_00189.png
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

If this lightweight branch already fails clearly on most frames, do not continue to the larger SDXL-based branch.

### 4. Run `AttentiveEraser` only if the lightweight baseline looks promising

Follow the official repository:

- [AttentiveEraser GitHub](https://github.com/Alibaba-YuFeng/AttentiveEraser)

The official README positions `AttentiveEraser` as image-level object removal on top of pre-trained diffusion models and shows `main.py` as the entrypoint. In this repo, use the local wrapper so you can process the exported keyframes without patching the upstream example file.

Smoke test on the first frame:

```bash
python scripts/run_attentive_eraser_keyframes.py \
  --workspace outputs/part3/wild_video2_attentive_eraser \
  --repo-dir external/AttentiveEraser \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --device cuda:0 \
  --keyframes 75 \
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

Then run the full four-frame batch:

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

The wrapper writes:

```text
outputs/part3/wild_video2_attentive_eraser/edited/
  edited_00075.png
  edited_00113.png
  edited_00151.png
  edited_00189.png
```

### 5. Generate the 4-column keyframe grid

```bash
python3 scripts/project3.py compare-keyframes \
  --original-dir outputs/part3/wild_video2_attentive_eraser/keyframes \
  --mask-dir outputs/part3/wild_video2_attentive_eraser/masks \
  --reference-dir outputs/part3/wild_video2_attentive_eraser/reference \
  --edited-dir outputs/part3/wild_video2_attentive_eraser/edited \
  --keyframes 75,113,151,189 \
  --reference-title ProPainter \
  --edited-title AttentiveEraser \
  --output outputs/part3/wild_video2_attentive_eraser/comparisons/wild_video2_attentive_eraser_grid.png
```

Default acceptance target:

- at least 3 out of 4 keyframes are visibly better than the `ProPainter` reference
- no major structure drift around the building, pavement, or fountain boundaries

If the diffusion outputs are sharper but inconsistent or structurally unstable, report the result as:

> a keyframe-level upper bound on restoration quality, not as a full-video replacement

## How to write the final Part 3 story

Keep the narrative tight:

1. Part 2 already showed that `ProPainter` helps when masks are fixed.
2. Part 3A tests whether a stronger tracking-aware mask model improves the same failure case.
3. Part 3B tests whether a stronger diffusion model improves the hardest keyframes once the mask is available.

Do not expand beyond `wild_video2` unless time remains after these two branches are complete and verified.
