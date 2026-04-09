
---

## idea006/design006 — Code Review

**Date:** 2026-04-09
**Verdict:** APPROVED

`RGBColorJitter` now unnormalizes ImageNet-normalized RGB to `[0,1]` before `ColorJitter` and renormalizes afterward; `RandomHorizontalFlip` and `DepthAugmentation` remain correct, config/train/model are unchanged, and the sanity check passed.

---

## idea006/design005 — Code Review

**Date:** 2026-04-09
**Verdict:** APPROVED

Combined geometric augmentation matches the design: `RandomScaleJitter(0.8, 1.2)` runs before `CropPerson`, `RandomHorizontalFlip` runs after crop and before `SubtractRoot`, RGB/depth flip logic and joint/pair handling are correct, config matches spec, and `train.py`/`model.py` remain unchanged.

---

## idea006/design004 — Code Review

**Date:** 2026-04-09
**Verdict:** APPROVED

DepthAugmentation matches the design: Gaussian noise (`sigma=0.02`) and 10% pixel dropout each fire with `p=0.5` after `ToTensor`; RGB and labels remain untouched; config matches spec; `train.py`/`model.py` unchanged.

---

## idea006/design003 — Code Review

**Date:** 2026-04-09
**Verdict:** APPROVED

RGBColorJitter matches the design: exact jitter params, RGB-only application after `ToTensor`, depth untouched, config matches spec, `train.py`/`model.py` unchanged; wrapper safely unnormalizes and re-normalizes around torchvision jitter.

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
