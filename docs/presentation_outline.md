# 8-Minute Progress Presentation Outline

This version is aligned with the current project status, not the original placeholder plan.

## Slide 1: Problem and project goal

- Task: remove moving foreground objects from a video and reconstruct a clean, temporally consistent background.
- Practical pipeline:
  - dynamic object mask generation
  - mask refinement
  - video inpainting
  - qualitative and quantitative evaluation
- Core project goal: compare a classical baseline with a stronger modern video restoration model.

## Slide 2: Datasets and current scope

- Mandatory or core sequences already processed:
  - `bmx-trees`
  - `tennis`
  - `wild_video1`
  - `wild_video2`
- Wild videos are self-recorded real scenes with a single person walking across a fixed-background scene.
- Optional next-step dataset:
  - DAVIS subset for stronger GT-backed evaluation

## Slide 3: Part 1 baseline pipeline

- Object proposal: `YOLOv8-seg`
- Dynamic filtering: Lucas-Kanade sparse optical flow
- Mask refinement: dilation + small connected-component cleanup
- Background restoration:
  - temporal median borrowing
  - `cv2.inpaint` fallback
- Key message: Part 1 is complete and fully reproducible.

## Slide 4: Part 1 quantitative status

- `bmx-trees`
  - `JM = 0.225`
  - `JR = 0.739`
  - precision is low, indicating overly large masks
- `tennis`
  - `JM = 0.517`
  - `JR = 0.591`
  - precision is much higher, giving a stronger baseline
- Interpretation:
  - mask quality is sequence-dependent
  - baseline is functional but not always precise

## Slide 5: Part 2 strategy and motivation

- We first keep the Part 1 masks fixed.
- We replace only the restoration backend:
  - Part 1: temporal median + `cv2.inpaint`
  - Part 2: `ProPainter`
- Why this controlled comparison matters:
  - isolates the gain from stronger video inpainting
  - avoids changing both masks and restoration at the same time
  - gives a clean story for the report and presentation

## Slide 6: Best qualitative results

- `bmx-trees`
  - Part 1 has strong blocky artifacts and background smearing
  - Part 2 restores graffiti and tree structures more naturally
- `tennis`
  - Part 1 leaves ghost-like residues on the court
  - Part 2 produces cleaner court texture and wall continuity
- Use these two as the main qualitative comparison figures.

## Slide 7: Real-scene success and failure analysis

- `wild_video1`
  - good real-scene success case
  - demonstrates generalization beyond provided sample clips
- `wild_video2`
  - important failure case
  - failure source is weak mask generation, not ProPainter itself
  - if the object is not masked, neither Part 1 nor Part 2 can remove it

## Slide 8: Next step and final delivery path

- Immediate next technical task:
  - improve `wild_video2` masks with `SAM 2` or prompt-guided propagation
  - keep `ProPainter` fixed
  - show improved removal completeness
- Final packaging path:
  - report with `bmx-trees` and `tennis` as main results
  - `wild_video1` as success case
  - `wild_video2` as failure case and improvement target
  - add one GT-backed quantitative experiment if time allows
