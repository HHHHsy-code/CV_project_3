# Report Outline

## Title

Video Object Removal and Background Inpainting with Classical Motion Filtering and Modern Video Priors

## Abstract

- Problem: remove dynamic objects from videos and restore a clean background
- Method summary: Part 1 baseline and a controlled Part 2 comparison using ProPainter with fixed masks
- Key contribution: strong comparison plus targeted failure-case analysis on weak masks
- GitHub link at the end of the abstract

## 1. Introduction

- Why dynamic object removal matters
- Why single-frame inpainting is insufficient for videos
- What this project studies

## 2. Related Work

- Object detection and segmentation: R-CNN, Mask R-CNN, YOLO, DETR
- Foundation segmentation and tracking: SAM, SAM 2, SAM 3, Track Anything, VGGT4D
- Video inpainting: FGVC, E2FGVI, ProPainter

## 3. Method

### 3.1 Part 1 baseline

- YOLOv8-seg proposals
- optical-flow-based motion filtering
- mask refinement
- temporal median borrowing
- `cv2.inpaint` fallback

### 3.2 Part 2 pipeline

- current controlled comparison:
  - reuse Part 1 masks
  - replace restoration with ProPainter
- next targeted upgrade:
  - prompt-assisted SAM 2 masks for failure cases such as `wild_video2`
  - ProPainter restoration with improved masks

### 3.3 Part 3 extension

- failure cases where missing content never appears
- keyframe-level generative repair

## 4. Experiments

- Datasets
- Implementation details
- Quantitative metrics
- Qualitative figures
- Failure analysis

### Current narrative for the paper

- `bmx-trees` and `tennis` are the main qualitative comparison sequences.
- `wild_video1` is the real-scene success case.
- `wild_video2` is the main failure case caused by weak mask generation.
- The central claim already supported by current experiments:
  - with the same masks, ProPainter produces better restoration quality than temporal median plus `cv2.inpaint`.

### Quantitative material already available

- `bmx-trees` Part 1 mask metrics:
  - `JM = 0.225`
  - `JR = 0.739`
  - precision `= 0.232`
- `tennis` Part 1 mask metrics:
  - `JM = 0.517`
  - `JR = 0.591`
  - precision `= 0.796`

### Quantitative material still worth adding

- one DAVIS subset experiment with `PSNR / SSIM`, if time allows
- mask-improvement evidence for `wild_video2`

## 5. Conclusion

- Summary
- Limitations
- Future work
