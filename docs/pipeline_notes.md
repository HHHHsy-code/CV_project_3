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

Recommended structure:

1. Run Part 1 or a detector-only pass to generate rough boxes
2. Use those prompts to initialize `SAM 2`
3. Save refined masks under `outputs/part2/<experiment>/masks`
4. Run `ProPainter`
5. Save restored outputs under `outputs/part2/<experiment>/propainter_output`

The command templates are emitted by:

```bash
python3 scripts/project3.py part2-prepare \
  --config configs/project3.yaml \
  --video data/raw/wild/corridor.mp4 \
  --experiment wild_corridor
```

## Part 3

Keep the scope narrow:

1. Identify one or two clear `ProPainter` failures
2. Extract keyframes and masks
3. Apply diffusion inpainting only to those keyframes
4. Compare before/after and discuss why propagation-only methods fail

Prepare a workspace with:

```bash
python3 scripts/project3.py part3-prepare \
  --experiment-dir outputs/part3 \
  --name unseen_background_failure \
  --keyframes 18,24,35 \
  --notes "Background never appears after object leaves."
```
