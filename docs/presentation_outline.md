# 8-Minute Progress Presentation Outline

## Slide 1: Problem setup

- What is video object removal and why it matters
- The exact project scope: dynamic object mask + clean background restoration

## Slide 2: Dataset and task definition

- Wild video
- `bmx-trees`
- `tennis`
- Optional DAVIS plan

## Slide 3: Part 1 baseline

- YOLOv8-seg for object proposals
- Lucas-Kanade optical flow for dynamic judgment
- Temporal borrowing + `cv2.inpaint`

## Slide 4: Current results

- Show masks
- Show restored frames
- Explain where the classical method works and where it fails

## Slide 5: Part 2 plan

- SAM 2 for better video masks
- ProPainter for stronger temporal inpainting

## Slide 6: Part 3 plan

- Focus on failure cases
- Keyframe-level diffusion repair when the background never appears

## Slide 7: Risks and next steps

- GPU and runtime requirements
- Need for better masks on thin structures
- Need for cleaner quantitative evaluation on GT-backed data
