# Review: idea006/design006

**Design_ID:** idea006/design006
**Date:** 2026-04-08
**Reviewer:** Reviewer agent
**Verdict:** APPROVED

---

## Summary

design006 combines all three augmentation strategies tested individually in designs 001, 003, and 004 into a single full-stack pipeline, exactly as specified by idea.md Axis 6. The design is complete, mathematically correct, architecturally feasible, and every configuration value is explicitly specified.

---

## Checklist

### 1. Completeness
- All three augmentation classes (`RandomHorizontalFlip`, `RGBColorJitter`, `DepthAugmentation`) are fully specified with pseudocode-level Python implementations.
- Pipeline ordering is given explicitly with a rationale table.
- `config.py` fields are fully enumerated; all values match baseline except `output_dir`.
- Files to modify are clearly listed: `transforms.py` and `config.py` only. `train.py` and `model.py` are unchanged.

### 2. Mathematical Correctness

**RandomHorizontalFlip** (identical to APPROVED design001):
- `cv2.flip(..., 1)` applied to both RGB and depth — correct.
- `joints[:, 1] *= -1.0` negates the lateral Y camera-space axis — correct.
- Left-right joint pairs swapped via `FLIP_PAIRS` (0..69 remapped indices from infra.py) — correct.
- `pelvis_uv` is NOT manually negated. `SubtractRoot` runs after the flip and recomputes `pelvis_uv` from the negated-Y joints automatically — this is the correct approach, consistent with lesson recorded in Reviewer memory.
- `pelvis_depth` unchanged — flip-invariant, correct.

**RGBColorJitter** (identical to APPROVED design003):
- Parameters: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1 — exactly match idea.md Axis 3.
- Applied after `ToTensor` on float32 RGB tensor — correct.
- Depth and joint labels untouched — correct.

**DepthAugmentation** (identical to APPROVED design004):
- Additive Gaussian noise: sigma=0.02, clamped to [0,1] — correct.
- Random pixel dropout: 10% Bernoulli mask — correct.
- Each perturbation fires independently with p=0.5 — correct.
- Applied after `ToTensor` on normalized float32 depth — correct.
- RGB and joint labels untouched — correct.

### 3. Pipeline Ordering
```
CropPerson → RandomHorizontalFlip → SubtractRoot → ToTensor → RGBColorJitter → DepthAugmentation
```
- Flip before SubtractRoot: required so pelvis_uv is computed on the flipped crop — correct.
- RGBColorJitter after ToTensor: torchvision operates on float tensors — correct.
- DepthAugmentation after ToTensor: requires float tensor in [0,1] — correct.

### 4. VRAM / Compute Feasibility
- Architecture unchanged: `sapiens_0.3b`, well within 1080ti 11GB VRAM.
- 20 epochs with baseline LR schedule — within the 20-epoch proxy limit.
- Augmentation adds negligible per-sample compute overhead.

### 5. Config Fields
All explicitly specified. No guesswork required for Builder:
- `output_dir`, `arch`, `img_h`, `img_w`, `head_hidden`, `head_num_heads`, `head_num_layers`, `head_dropout`, `drop_path`, `epochs`, `lr_backbone`, `lr_head`, `weight_decay`, `warmup_epochs`, `grad_clip`, `lambda_depth`, `lambda_uv`.

### 6. Alignment with idea.md
- Axis 6 specifies: Flip (p=0.5) + Color Jitter + Depth Noise; Scale Jitter excluded — design matches exactly.
- Component designs 001, 003, 004 were all previously APPROVED; this design faithfully composes them.

---

## Issues Found

None.

---

## Verdict: APPROVED

The design is complete, mathematically sound, architecturally feasible, and consistent with the idea.md specification and all three component designs it composes. The Builder may proceed directly to implementation.
