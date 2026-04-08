# Review — idea006/design004

**Design_ID:** idea006/design004
**Reviewer:** Reviewer agent
**Date:** 2026-04-08
**Verdict:** APPROVED

---

## Summary

design004 implements Depth Channel Augmentation (Axis 4 from idea006/idea.md): additive
Gaussian noise (σ=0.02, p=0.5) and random pixel dropout (10% of pixels zeroed, p=0.5),
each applied independently to the normalized depth tensor after `ToTensor`. The design is
complete, mathematically correct, and feasible.

---

## Checklist

### Alignment with idea.md
- [x] Implements exactly Axis 4 as specified: noise σ=0.02, dropout rate=0.10, both at p=0.5.
- [x] `build_val_transform` unchanged.

### Mathematical Correctness
- [x] Gaussian noise: `depth + ε, ε ~ N(0, σ²)` then `clamp(0, 1)` — correct.
- [x] Pixel dropout: `depth * M` where `M ~ Bernoulli(1 - p_drop)` — correct; zeroing
  dropped pixels matches sensor convention (0 = no valid depth reading).
- [x] Each perturbation gated by independent `np.random.random() < p` draw — correct
  for independent application.

### Label / Coordinate Correctness
- [x] `sample["joints"]` not modified — correct, since joints are GT 3D coordinates
  derived from skeleton, not from depth image.
- [x] `pelvis_uv`, `pelvis_depth`, `sample["intrinsic"]`, `sample["rgb"]` all unaffected.
- [x] No label noise introduced.

### Pipeline Order
- [x] `DepthAugmentation` placed after `ToTensor` — operates on float32 tensors in [0,1].
- [x] `SubtractRoot` runs before augmentation, so `pelvis_uv`/`pelvis_depth` are computed
  from clean data before any perturbation.

### Compute / VRAM Feasibility
- [x] No model architecture change; augmentation is pure CPU tensor ops in DataLoader workers.
- [x] Trivially within 1080ti 11GB VRAM constraint at 20 epochs.

### Config Completeness
- [x] All config fields explicitly specified: `output_dir`, `arch`, `img_h`, `img_w`,
  `head_hidden`, `head_num_heads`, `head_num_layers`, `head_dropout`, `drop_path`,
  `epochs`, `lr_backbone`, `lr_head`, `weight_decay`, `warmup_epochs`, `grad_clip`,
  `lambda_depth`, `lambda_uv`.
- [x] Builder has zero ambiguity on any parameter.

### Implementation Quality
- [x] Full `DepthAugmentation` class provided with correct PyTorch idioms.
- [x] `torch.randn_like` and `torch.bernoulli` respect tensor device (CPU in DataLoader
  workers) — no device mismatch possible.
- [x] `np.random.random()` used for the sample-level gate — acceptable in DataLoader
  workers; does not affect reproducibility in a meaningful way.
- [x] Files to modify clearly specified; `train.py` and `model.py` explicitly unchanged.

---

## Issues Found

None.

---

## Verdict: APPROVED

The design is complete, correct, and ready for the Builder to implement.
