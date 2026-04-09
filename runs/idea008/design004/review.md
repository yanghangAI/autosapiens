# Design Review — idea008/design004

**Design_ID:** idea008/design004
**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Summary

The hybrid two-resolution depth PE design is complete, mathematically explicit, and
feasible within the 20-epoch proxy budget. It cleanly extends the successful
row/column-plus-depth formulation without introducing heavy attention-side machinery.

---

## Detailed Checks

### 1. Architecture and feasibility

- The design keeps the successful row + column decomposition from `runs/idea005/design001`.
- The local depth term is a lightweight interpolated 16-anchor encoding on the existing
  `40 x 24` patch grid.
- The coarse depth term is a tiny 4-anchor image-level code broadcast across tokens.
- No pairwise token interactions, attention biases, or extra encoders are introduced.
- The parameter sizes and operations remain appropriate for a single 1080ti proxy run.

### 2. Mathematical completeness

- Fine local depth interpolation is specified explicitly with `floor`, `clamp`, and linear
  mixing between neighboring anchors.
- The coarse global depth code is specified explicitly via mean depth, interpolation over
  4 anchors, and broadcast to all tokens.
- The full injection rule is clear:
  `row_pe + col_pe + fine_local_depth_pe + coarse_global_depth_pe`.

### 3. Implementation mapping

- `code/model.py` is correctly identified as the place for the hybrid depth PE module and
  the backbone forward-path change.
- `code/train.py` is correctly limited to optional optimizer wiring only.
- `code/transforms.py` is correctly left unchanged.

### 4. Configuration completeness

- All required config values are explicit, including `lr_backbone`, `lr_head`,
  `lr_depth_pe`, `num_fine_depth_bins`, and `num_coarse_depth_bins`.
- The design does not leave any builder-facing ambiguity about hidden dimensions or
  optimizer grouping.

---

## Issues Found

None.

---

## Verdict

APPROVED
