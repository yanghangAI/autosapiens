# Review — idea006/design002

**Design_ID:** idea006/design002  
**Reviewer:** Reviewer agent  
**Date:** 2026-04-08  
**Verdict:** APPROVED

---

## Summary

design002 implements Axis 2 (Scale/Crop Jitter) from idea006: a `RandomScaleJitter` transform that multiplies the person bounding box half-extents by `s ~ Uniform(0.8, 1.2)` around the bbox center, applied before `CropPerson`. This is the correct, minimal change needed to introduce scale invariance without touching labels.

---

## Evaluation

### Idea Alignment
Matches idea006 Axis 2 exactly. No scope creep.

### Mathematical Correctness
- Bbox scaling formula is correct: scale half-extents `(cx - x_min)` and `(cy - y_min)` by `s` around center — this uniformly enlarges or shrinks the crop region.
- Root-relative 3D joint coordinates in metric camera space are independent of pixel crop extent. No label adjustment required. This is verified by examining `SubtractRoot` in `baseline/transforms.py`, which computes `pelvis_uv` from the updated crop-space intrinsics after `CropPerson` runs — the pipeline is self-consistent.
- The design provides the correct formula for both `[x_min, y_min, x_max, y_max]` and `[cx, cy, w, h]` formats with an appropriate Builder note to verify convention.

### Pipeline Order
`RandomScaleJitter → CropPerson → SubtractRoot → ToTensor` is correct. The jittered bbox is consumed by `CropPerson`, which recomputes crop-space intrinsics. `SubtractRoot` then correctly projects the pelvis into the updated crop space.

### Out-of-Bounds Handling
Verified against `baseline/transforms.py`: `CropPerson` already uses `cv2.copyMakeBorder` with zero-padding for crops that extend beyond image boundaries. The design's recommendation to let `CropPerson` handle out-of-bounds is accurate and safe.

### Configuration Completeness
All required fields are explicitly specified in the `config.py` block:
- `arch`, `img_h`, `img_w`
- `head_hidden`, `head_num_heads`, `head_num_layers`, `head_dropout`, `drop_path`
- `epochs = 20`, `lr_backbone = 1e-5`, `lr_head = 1e-4`, `weight_decay = 0.03`
- `warmup_epochs = 3`, `grad_clip = 1.0`
- `lambda_depth = 0.1`, `lambda_uv = 0.2`
- `output_dir` updated correctly to `runs/idea006/design002`

No guessing required by the Builder.

### VRAM Constraint
Model architecture is unchanged. Only `transforms.py` is modified. A 20-epoch run on a single 1080Ti (11GB) is fully feasible.

### Files to Modify
Correctly scoped to `transforms.py` and `config.py` only. `train.py` and `model.py` are unchanged, which is appropriate.

---

## Issues Found

None. The design is complete, mathematically correct, and implementation-ready.

---

## Verdict: APPROVED
