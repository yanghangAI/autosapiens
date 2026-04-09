# Code Review — idea008/design003

**Design_ID:** idea008/design003
**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Summary

The implementation matches the near-emphasized continuous depth PE design. The square-root depth remapping is implemented in the correct module, the interpolation math is consistent with the design, the optimizer wiring keeps the depth-PE parameters in the high-LR group, and the configuration fields match the spec.

---

## Detailed Checks

### 1. `model.py`

- `DepthBucketPE` uses `torch.sqrt(depth_norm) * (num_depth_bins - 1)` exactly as specified.
- The pooled depth map is clamped to `[0, 1]` before the nonlinear remapping, which matches the design.
- Interpolation between neighboring anchors uses `idx_lo`, `idx_hi`, and `alpha` in the expected way.
- `row_emb`, `col_emb`, and `depth_emb` retain the required shapes and initialization behavior from the previous design.
- The custom backbone forward path still injects `row_pe + col_pe + depth_pe` at the patch-token stage, which is the correct place for this change.
- `vit.pos_embed` remains zeroed and frozen as a buffer.

### 2. `train.py`

- The optimizer wiring still separates `model.backbone.depth_bucket_pe.parameters()` into the `lr_depth_pe` group.
- No extra training-loop or dataloader changes were introduced, which is consistent with the design.
- Experiment-specific values remain in `config.py` rather than being hardcoded in the loop.

### 3. `config.py`

- `output_dir` points to `runs/idea008/design003`.
- The requested fields match the design: `arch`, image size, head dimensions, epochs, learning rates, `weight_decay`, `warmup_epochs`, `num_depth_bins`, `grad_clip`, and loss weights.

### 4. Sanity Check

- The design passed the required 2-epoch test and produced `metrics.csv`, `iter_metrics.csv`, and a clean SLURM log.

---

## Issues Found

None.

---

## Verdict

APPROVED
