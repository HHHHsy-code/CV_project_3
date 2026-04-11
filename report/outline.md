# Report Outline

## Title

Video Object Removal and Background Inpainting with Classical Motion Filtering and Modern Video Priors

## Abstract

- Problem: remove dynamic objects from videos and restore a clean background
- Method summary: Part 1 baseline and Part 2 SOTA pipeline
- Key contribution: strong comparison plus targeted Part 3 failure-case analysis
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

- prompt-assisted SAM 2 masks
- ProPainter restoration

### 3.3 Part 3 extension

- failure cases where missing content never appears
- keyframe-level generative repair

## 4. Experiments

- Datasets
- Implementation details
- Quantitative metrics
- Qualitative figures
- Failure analysis

## 5. Conclusion

- Summary
- Limitations
- Future work
