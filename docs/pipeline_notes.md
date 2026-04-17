# Pipeline Notes

## Part 1

The baseline is intentionally conservative:

1. Detect likely dynamic categories with `YOLOv8-seg`
2. Use sparse optical flow inside each instance mask to reject static detections
3. Dilate and clean masks
4. Borrow clean pixels from nearby frames with a temporal median
5. Use `cv2.inpaint` as a fallback to close remaining holes

This gives you a full end-to-end classical baseline for the presentation and for Part 2 comparisons.

## Part 2

Recommended staged structure:

### Stage A: controlled restoration comparison

This is the current recommended first Part 2 experiment:

1. Reuse the masks produced by Part 1
2. Keep the removal region fixed
3. Replace only the restoration backend with `ProPainter`
4. Compare:
   - Part 1: temporal median + `cv2.inpaint`
   - Part 2: `ProPainter`

Why this first:

- lower engineering risk
- clear interpretation of gains
- faster path to presentation-ready evidence

### Stage B: targeted mask upgrade

After Stage A is stable, upgrade only the weak-mask failure cases:

1. Run Part 1 or a detector-only pass to generate rough boxes
2. Use those prompts to initialize `SAM 2`
3. Save refined masks under `outputs/part2/<experiment>/masks`
4. Run `ProPainter`
5. Compare the original failure with the improved mask + ProPainter output

Current recommended target:

- `wild_video2`
- Failure source: object not consistently masked
- Goal: improve mask completeness without changing the inpainting backend

The command templates are emitted by:

```bash
python3 scripts/project3.py part2-prepare \
  --config configs/project3.yaml \
  --video data/raw/wild/corridor.mp4 \
  --experiment wild_corridor
```

## Part 3

Keep the scope narrow:

1. Identify one or two clear failure cases
2. Determine whether the root cause is:
   - weak masks
   - inpainting failure despite good masks
3. If the root cause is missing background content, extract keyframes and masks
4. Apply diffusion inpainting only to those keyframes
5. Compare before/after and discuss why propagation-only methods fail

Prepare a workspace with:

```bash
python3 scripts/project3.py part3-prepare \
  --experiment-dir outputs/part3 \
  --name unseen_background_failure \
  --keyframes 18,24,35 \
  --notes "Background never appears after object leaves."
```
