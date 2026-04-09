# Code Review — idea007/design001

**Design_ID:** idea007/design001
**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Summary

The implementation matches the design spec for depth-bucket positional embeddings with layer-wise learning-rate decay and progressive unfreezing. The 2-epoch sanity check completed successfully and produced the expected metrics artifacts.

---

## Detailed Checks

### 1. `config.py`

- `output_dir` points to `runs/idea007/design001`.
- All required schedule fields are present and match the design: `lr_backbone=1e-5`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `gamma=0.95`, `unfreeze_epoch=5`, `num_depth_bins=16`.
- The fixed training settings also match the spec: `epochs=20`, `weight_decay=0.03`, `warmup_epochs=3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`.

### 2. `train.py`

- The optimizer is split into the requested groups: depth-bucket PE, per-block ViT groups, head, and the patch embed group after unfreezing.
- The frozen phase correctly excludes `vit.patch_embed` and blocks `0-11`, while keeping blocks `12-23`, the depth-bucket PE, and the head trainable.
- The full unfreeze happens at `epoch == unfreeze_epoch`, which matches the design’s epoch-5 transition.
- The block learning-rate formula `lr_backbone * gamma ** (23 - i)` matches the specification, and the embedding LR uses `lr_backbone * gamma ** 24` as required.
- The linear warmup plus cosine decay is applied through `initial_lr` for every optimizer group, which is what the design asked for.
- The deterministic seed is fixed to `2026`, and the training script keeps the no-heavy-augmentation setup intact.

### 3. `model.py` and `transforms.py`

- `model.py` keeps the custom `DepthBucketPE` and custom backbone forward path unchanged from the design starting point.
- `vit.pos_embed` is zeroed out as a buffer and is not added to the optimizer groups, which matches the design notes.
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
