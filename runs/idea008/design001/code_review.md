# Code Review — idea008/design001

**Design_ID:** idea008/design001
**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Summary

The implementation matches the continuous interpolated depth PE design and is aligned with the actual starting-point layout. The new depth interpolation logic lives in `code/model.py`, the optimizer wiring still separates the depth-PE parameters into their own LR group, and the configuration fields match the spec exactly.

---

## Detailed Checks

### 1. `model.py`

- `DepthBucketPE` now performs continuous interpolation between neighboring depth anchors instead of hard bucket lookup.
- The interpolation math matches the design: normalized depth is pooled to the `40 x 24` patch grid, scaled to `[0, 15]`, split into `idx_lo` / `idx_hi`, and blended with `alpha`.
- `row_emb`, `col_emb`, and `depth_emb` have the required shapes and remain zero-initialized or pretrained-initialized as specified.
- The custom backbone forward path adds `row_pe + col_pe + depth_pe` at the patch-token stage, which is the correct implementation site.
- `vit.pos_embed` remains zeroed and frozen via a buffer, consistent with the design.

### 2. `train.py`

- Optimizer wiring still separates `depth_bucket_pe` parameters into their own `lr_depth_pe` group.
- The training loop, loss computation, dataloading, and schedule remain unchanged, which is what the design requested.
- No experiment-specific values are hardcoded in the loop; they come from `config.py`.

### 3. `config.py`

- `output_dir` points to `runs/idea008/design001`.
- All requested fields match the design: `arch`, image size, head dims, epochs, learning rates, `weight_decay`, `warmup_epochs`, `num_depth_bins`, `grad_clip`, and loss weights.

### 4. Sanity Check

- The design passed the required 2-epoch test and produced `metrics.csv`, `iter_metrics.csv`, and a clean SLURM log with no crash or OOM.

---

## Issues Found

None.

---

## Verdict

APPROVED
