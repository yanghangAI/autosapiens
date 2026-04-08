
---

## idea006/design001 — Code Review

**Date:** 2026-04-08
**Verdict:** APPROVED

The `RandomHorizontalFlip` class matches the design spec precisely: `cv2.flip` with flipCode=1, Y-axis negation via `joints[:, 1] *= -1.0`, FLIP_PAIRS swap with copy-to-avoid-alias, and correct pipeline order (flip after CropPerson, before SubtractRoot). `pelvis_uv` is correctly not manually negated (SubtractRoot recomputes it). All 16 config fields match exactly. `train.py` and `model.py` are unchanged from baseline as specified. One inert dead parameter (`scale_jitter` in `build_train_transform`) does not affect correctness.
