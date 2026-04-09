# Code Review — idea007/design003

**Design_ID:** idea007/design003  
**Date:** 2026-04-09  
**Verdict:** APPROVED

---

## Summary

The implementation matches the design spec for strong LLRD with earlier unfreezing on the depth-bucket positional embedding backbone. The fixed-value schedule, optimizer grouping, and deterministic training setup are all consistent with the design, and the 2-epoch sanity check completed cleanly.

---

## Detailed Checks

### 1. `config.py`

- `output_dir` points to `runs/idea007/design003`.
- All schedule fields match the design exactly: `lr_backbone=1e-5`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `gamma=0.90`, `unfreeze_epoch=3`, `num_depth_bins=16`.
- The remaining required settings also match the spec: `epochs=20`, `weight_decay=0.03`, `warmup_epochs=3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`.

### 2. `train.py`

- The optimizer is split into the requested frozen and full-backbone phases.
- During epochs `0-2`, `vit.patch_embed` and blocks `0-11` are frozen while blocks `12-23`, the depth-bucket PE, and the head remain trainable.
- At `epoch == unfreeze_epoch`, the code rebuilds the optimizer with the full backbone unfrozen, which matches the design’s earlier unfreeze requirement.
- The per-block LR formula `lr_backbone * gamma ** (23 - i)` is implemented correctly, and the embedding LR uses `lr_backbone * gamma ** 24`.
- The deterministic seed is fixed to `2026`, and the train/val transform setup remains unchanged from the design starting point.

### 3. `model.py` and `transforms.py`

- `model.py` keeps the custom `DepthBucketPE` and the modified backbone forward path unchanged from the starting point.
- `vit.pos_embed` is zeroed out as a buffer, so it is not treated as a trainable optimizer group.
- `transforms.py` remains unchanged, as required by the design.

### 4. Sanity Check

- The 2-epoch test run completed successfully.
- `metrics.csv`, `iter_metrics.csv`, and the SLURM log were generated without crash or OOM.

---

## Issues Found

None.

---

## Verdict

APPROVED
