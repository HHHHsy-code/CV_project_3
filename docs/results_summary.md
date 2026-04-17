# Current Results Summary

This document consolidates the current project status, the strongest qualitative findings, and the immediate next steps.

## Completed milestones

- Repository scaffold, README, report template, and execution notes are in place.
- Server-side environments are working for:
  - Part 1 baseline (`.venv-cu128`)
  - ProPainter (`/data2/zguo315/conda_envs/propainter`)
- Part 1 has been run on all current target videos:
  - `bmx-trees`
  - `tennis`
  - `wild_video1`
  - `wild_video2`
- Part 2 (ProPainter) has been run on the same four videos.

## Quantitative results available now

Only Part 1 mask metrics are currently available.

| Dataset | Method | JM | JR | Precision | Notes |
|---|---|---:|---:|---:|---|
| `bmx-trees` | Part 1 baseline | 0.225 | 0.739 | 0.232 | High recall, low precision, masks too large |
| `tennis` | Part 1 baseline | 0.517 | 0.591 | 0.796 | Much tighter masks, stronger qualitative baseline |

## Qualitative findings

### `bmx-trees`

- Strong Part 1 vs Part 2 comparison.
- Part 1 shows obvious blocky and smeared background artifacts after removal.
- Part 2 restores tree trunks, graffiti, and background texture much more naturally.
- Recommended as a primary qualitative figure in the report.

### `tennis`

- Strongest clean comparison among the current results.
- Masks are stable enough for a fair comparison.
- Part 1 leaves transparent or ghost-like residual artifacts on the court.
- Part 2 produces much cleaner court texture and wall continuity.
- Recommended as a primary qualitative figure in the report.

### `wild_video1`

- Good real-scene success case.
- Improvement from Part 1 to Part 2 is visible but smaller than on `bmx-trees` and `tennis`.
- Useful to demonstrate generalization beyond the provided sample sequences.

### `wild_video2`

- Best current failure-to-improvement case.
- Baseline failure:
  - the issue is not ProPainter itself
  - the main failure is upstream mask generation
  - the moving person is not consistently covered by the automatic Part 1 mask
  - because the object is not marked for removal, the original Part 2 output still leaves the target visible
- Improved version:
  - we used a prompt-guided `SAM 2` mask refinement step before ProPainter
  - the refined masks cover the walking person much more consistently in the key frames
  - the final restored video removes the target more completely than the original `wild_video2` Part 2 result
- Lightweight quantitative support:
  - `non_empty_frames` increases from `104` to `122`
  - `temporal_coverage` increases from `0.547` to `0.642`
  - `mean_area_ratio_all` decreases from `0.0166` to `0.0115`
  - this suggests the refined masks are both more temporally stable and spatially tighter
- This sequence should now be shown as:
  - original failure case: `wild_video2_part1_vs_part2.png`
  - improved case: `wild_video2_part1_vs_sam2_part2.png`

## Recommended report assets

- Main qualitative figure 1: `bmx_trees_part1_vs_part2.png`
- Main qualitative figure 2: `tennis_part1_vs_part2.png`
- Real-scene success case: `wild_video1_part1_vs_part2.png`
- Failure case: `wild_video2_part1_vs_part2.png`
- Improved failure case: `wild_video2_part1_vs_sam2_part2.png`

## Asset checklist

Make sure the following are copied to a local machine before report writing and presentation recording:

- Part 1 outputs:
  - `restored_video.mp4`
  - `figures/comparison_grid.png`
- Part 2 outputs:
  - `inpaint_out.mp4`
  - `masked_in.mp4`
- Quantitative files:
  - `bmx_trees_part1_gpu_mask_metrics.json`
  - `tennis_part1_gpu_mask_metrics.json`
- Part 1 vs Part 2 comparison grids:
  - `Original / Mask / Part 1 / Part 2`

## Immediate next steps

1. Keep `bmx-trees` and `tennis` as the primary report results.
2. Treat `wild_video1` as the real-video success case.
3. Present `wild_video2` as a failure-to-improvement case study.
4. Add one more quantitative experiment if time allows:
   - preferred: a small DAVIS subset for additional `JM / JR`
   - optional: a synthetic masked-frame benchmark for `PSNR / SSIM`

## Current project narrative

The current story is already clear enough for a progress presentation:

1. Build a fully reproducible classical baseline.
2. Replace the classical inpainting backend with `ProPainter` while keeping masks fixed.
3. Show that deep video inpainting clearly improves restoration quality.
4. Analyze a failure caused by weak automatic masks on `wild_video2`.
5. Show that prompt-guided `SAM 2` refinement improves mask completeness and leads to a better final restoration on the same sequence.
