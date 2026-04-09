# Code Review — idea006/design004

**Design_ID:** idea006/design004  
**Date:** 2026-04-09  
**Verdict:** APPROVED

---

## Summary

The implementation matches the design for depth-channel augmentation. Both requested perturbations are applied only to the depth tensor after `ToTensor`, RGB and labels are left unchanged, and the configuration remains aligned with the baseline values specified in the design.

---

## Detailed Checks

### 1. `transforms.py`

- `DepthAugmentation` is implemented and inserted into `build_train_transform()` only.
- Gaussian noise uses `noise_sigma=0.02`, matching the design exactly.
- Pixel dropout uses `dropout_rate=0.10`, also matching the design exactly.
- Each perturbation is gated independently with `p=0.5`, so either, both, or neither may fire on a sample, as specified.
- Augmentation is applied after `ToTensor`, so it operates on the normalized float depth tensor in `[0, 1]`.
- `sample["rgb"]`, `sample["joints"]`, `pelvis_uv`, `pelvis_depth`, and `intrinsic` are left unchanged, which is correct for a depth-only input augmentation.
- `build_val_transform()` remains deterministic and unchanged.

### 2. `config.py`

- `output_dir` is set to `runs/idea006/design004`.
- All required configuration values match the design: `arch`, image size, head dimensions, epochs, learning rates, weight decay, warmup, gradient clip, and loss weights.
- No experiment-specific values are hardcoded in `train.py`.

### 3. `train.py` and `model.py`

- `train.py` remains unchanged from the baseline implementation.
- `model.py` remains unchanged from the baseline implementation.
- The training loop still consumes `lambda_depth` and `lambda_uv` from config, so the design’s loss-weight settings are honored correctly.

---

## Issues Found

None.

---

## Verdict

APPROVED
