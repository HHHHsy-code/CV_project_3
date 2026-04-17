# Report Assets

Put exported comparison images for the LaTeX report in this directory.

## Current expected filenames

- `bmx_trees_part1_vs_part2.png`
  - main qualitative figure for the controlled Part 1 vs Part 2 comparison
- `wild_video2_part1_vs_sam2_part2.png`
  - main failure-to-improvement figure for the SAM 2 refinement case

## Optional additional assets

- `tennis_part1_vs_part2.png`
  - backup or secondary qualitative figure
- `wild_video1_part1_vs_part2.png`
  - real-scene success case

The current `report/main.tex` is written to automatically include:

- `assets/bmx_trees_part1_vs_part2.png`
- `assets/wild_video2_part1_vs_sam2_part2.png`

If these files are missing, the report will still compile and show placeholders instead.
