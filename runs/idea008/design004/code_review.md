# Code Review — idea008/design004

**Design_ID:** idea008/design004
**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Summary

The hybrid two-resolution depth positional encoding is implemented correctly. `code/model.py`
adds both the fine local interpolated depth term and the coarse global broadcast depth term,
the config fields match the design, and the 2-epoch sanity check completed cleanly.

---

## Detailed Checks

### 1. `code/model.py`

- `HybridDepthPE` replaces the old hard-bucket lookup module and keeps the row/column
  embeddings unchanged.
- Fine local depth encoding uses 16 anchors with linear interpolation between `idx_lo`
  and `idx_hi`, matching the design.
- Coarse global depth encoding uses a single image-level pooled depth scalar, interpolates
  across 4 anchors, and broadcasts the resulting code to all patch tokens.
- The injected positional term is exactly `row + col + fine_local + coarse_global`.
- `fine_depth_emb` and `coarse_depth_emb` are zero-initialized, while `row_emb` and
  `col_emb` are initialized from the pretrained 2D positional embedding as specified.

### 2. `code/train.py`

- The training loop remains unchanged except for passing
  `num_fine_depth_bins` and `num_coarse_depth_bins` into `SapiensPose3D`.
- Optimizer grouping still uses `model.backbone.depth_bucket_pe.parameters()`, which now
  correctly includes both fine and coarse depth embedding tables in the high-LR depth-PE
  parameter group.

### 3. `code/config.py`

- `output_dir` is set to `runs/idea008/design004`.
- The explicit design fields match: `lr_backbone=1e-5`, `lr_head=1e-4`,
  `lr_depth_pe=1e-4`, `num_fine_depth_bins=16`, `num_coarse_depth_bins=4`,
  `epochs=20`, `drop_path=0.1`, `lambda_depth=0.1`, and `lambda_uv=0.2`.

### 4. Sanity Check

- `python scripts/cli.py submit-test runs/idea008/design004/` completed successfully.
- The test output includes `metrics.csv`, `iter_metrics.csv`,
  `slurm_test_55333738.out`, and `train_snapshot.py`.
- The SLURM log finished with `[test] Finished.` and reported a best validation weighted
  MPJPE of `734.8mm`.

---

## Issues Found

None.

---

## Verdict

APPROVED
