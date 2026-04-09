
---

## idea006/design002 — Code Review

**Date:** 2026-04-09
**Verdict:** APPROVED

`RandomScaleJitter` class matches the design spec precisely: `low=0.8, high=1.2`, bbox format `[x_min, y_min, x_max, y_max]`, half-extents scaled by `s` around center with center unchanged, `bbox.copy()` to avoid aliasing, and a defensive guard when bbox is absent. Pipeline order is `[RandomScaleJitter, CropPerson, SubtractRoot, ToTensor]` — exactly as specified. Out-of-bound handling delegated to `CropPerson` padding as permitted. `build_val_transform` is unchanged. All 17 config fields match exactly. `train.py` and `model.py` are unchanged from baseline as specified. No bugs found.

---

## idea006/design001 — Code Review

**Date:** 2026-04-08
**Verdict:** APPROVED

The `RandomHorizontalFlip` class matches the design spec precisely: `cv2.flip` with flipCode=1, Y-axis negation via `joints[:, 1] *= -1.0`, FLIP_PAIRS swap with copy-to-avoid-alias, and correct pipeline order (flip after CropPerson, before SubtractRoot). `pelvis_uv` is correctly not manually negated (SubtractRoot recomputes it). All 16 config fields match exactly. `train.py` and `model.py` are unchanged from baseline as specified. One inert dead parameter (`scale_jitter` in `build_train_transform`) does not affect correctness.
