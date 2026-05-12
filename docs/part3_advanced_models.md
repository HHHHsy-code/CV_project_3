# Part 3 Final Status on `wild_video2`

This note records what was actually completed for the final `Part 3` branch, rather than listing every possible extension.

The checked final state is:

1. `SAM 2 + ProPainter` is the main improvement branch on `wild_video2`
2. multi-prompt `SAM 2` fusion was tested as an ablation and produced little extra gain
3. a lightweight image-level `SD2-Inpaint` baseline was run on four keyframes as a supplemental restoration check
4. the heavier `AttentiveEraser + SDXL` branch remains optional and was not needed for the final conclusion

## Why Part 3 ended up this way

The current evidence supports three stable conclusions:

- `ProPainter` is stronger than the classical Part 1 restoration backend when masks are fixed
- `wild_video2` fails mainly because the moving person is not covered consistently across frames
- once the mask quality is improved with `SAM 2`, the final restored video becomes visibly cleaner

That means the most useful final Part 3 questions are:

- does stronger mask propagation improve the same failure case?
- does prompt ensembling add more benefit after the first `SAM 2` gain?
- if `ProPainter` is already strong, does a lightweight image-level diffusion baseline clearly beat it on the hardest frames?

## Branch A: single-prompt `SAM 2 + ProPainter`

### 1. Run `SAM 2` refinement from a non-empty seed frame

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
conda activate /data2/zguo315/conda_envs/sam2

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

### 2. Summarize the refined masks

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py summarize-mask \
  --pred outputs/part2/wild_video2_sam2_refined_masks \
  --output outputs/part2/wild_video2_sam2_refined_masks_sam2_summary.json
```

### 3. Rerun `ProPainter`

```bash
cd /data2/zguo315/CV_project_3/CV_project_3/external/ProPainter
conda activate /data2/zguo315/conda_envs/propainter

CUDA_VISIBLE_DEVICES=0 python inference_propainter.py \
  --video ../../data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask ../../outputs/part2/wild_video2_sam2_refined_masks \
  --output ../../outputs/part2/wild_video2_propainter_from_sam2 \
  --fp16 \
  --subvideo_length 80 \
  --save_fps 24 \
  --save_frames
```

### 4. Build the comparison figure

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py compare-methods \
  --part1-dir outputs/part1/wild_video2_480p24_part1_gpu \
  --mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --part2-video outputs/part2/wild_video2_propainter_from_sam2/wild_video2_480p24/inpaint_out.mp4 \
  --output outputs/comparisons/wild_video2_part1_vs_sam2_part2.png
```

This branch is the main final `Part 3` result.

## Branch B: multi-prompt `SAM 2` ablation

We also tested a simple prompt-ensemble hypothesis: maybe the remaining failure is caused by relying on only one seed frame.

In the checked final run, three non-empty seeds were used:

- `75`
- `139`
- `152`

The propagated masks were merged with:

- `union`
- `majority`

Example merge command:

```bash
python3 scripts/project3.py merge-mask-dirs \
  --mask-dir outputs/part3/wild_video2_sam2_seed075_masks \
  --mask-dir outputs/part3/wild_video2_sam2_seed139_masks \
  --mask-dir outputs/part3/wild_video2_sam2_seed152_masks \
  --output-dir outputs/part3/wild_video2_sam2_multi_majority_masks \
  --mode majority
```

Then summarize:

```bash
python3 scripts/project3.py summarize-mask \
  --pred outputs/part3/wild_video2_sam2_multi_majority_masks \
  --output metrics/wild_video2_sam2_multi_majority_summary.json
```

Final checked conclusion:

- `union` and `majority` both kept `122` non-empty frames
- temporal coverage stayed at `0.642`
- mean mask area changed only marginally

So this branch is best reported as a **negative or limited-gain ablation**, not as a new main result.

## Branch C: lightweight `SD2-Inpaint` keyframe baseline

This branch is supplemental. It is not a video-level replacement pipeline.

### 1. Prepare the keyframe workspace

```bash
cd /data2/zguo315/CV_project_3/CV_project_3
source .venv-cu128/bin/activate

python3 scripts/project3.py part3-prepare \
  --experiment-dir outputs/part3 \
  --name wild_video2_attentive_eraser \
  --keyframes 75,113,151,189 \
  --notes "Keyframe-level diffusion repair for wild_video2"
```

### 2. Export frames, masks, and reference results

```bash
python3 scripts/project3.py part3-export-keyframes \
  --video data/raw/wild_processed/wild_video2_480p24.mp4 \
  --mask-dir outputs/part2/wild_video2_sam2_refined_masks \
  --reference-video outputs/part2/wild_video2_propainter_from_sam2/wild_video2_480p24/inpaint_out.mp4 \
  --workspace outputs/part3/wild_video2_attentive_eraser
```

### 3. Run the lightweight image-level baseline

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

### 4. Build the 4-column comparison grid

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

Final checked conclusion:

- `SD2-Inpaint` works on the four keyframes
- it removes the walking person cleanly
- but it does **not** clearly outperform `ProPainter`

So this branch should be reported as a **supplemental exploratory result**, not as a replacement for the main video pipeline.

## Optional heavy branch: `AttentiveEraser + SDXL`

The repo includes a wrapper script for a heavier image-level branch:

- `scripts/run_attentive_eraser_keyframes.py`

This branch is intentionally left optional.
It was not required for the final checked project state because the lighter `SD2` baseline already showed that diffusion was not clearly beating `ProPainter` on the selected keyframes.

## Final Part 3 story to keep consistent everywhere

Use this wording consistently in the report, README, and presentation:

1. `wild_video2` reveals that mask quality, not just restoration quality, is a key bottleneck
2. prompt-guided `SAM 2` refinement improves the final result when `ProPainter` is held fixed
3. multi-prompt `SAM 2` fusion does not materially improve over the single-prompt branch
4. lightweight diffusion inpainting is feasible on the hard keyframes, but it does not clearly outperform `ProPainter`

That is the checked final state of the current repository.
