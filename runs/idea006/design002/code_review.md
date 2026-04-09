## idea006/design002 — Code Review

**Date:** 2026-04-09
**Verdict:** APPROVED

---

### Summary

The implementation faithfully matches the design spec for `RandomScaleJitter` augmentation. All critical points are verified below.

---

### transforms.py

**`RandomScaleJitter` class:**
- Constructor parameters `low=0.8, high=1.2` match the design's `Uniform(0.8, 1.2)` specification exactly.
- Bbox format assumed is `[x_min, y_min, x_max, y_max]` — correct; the design designates this as the primary candidate.
- Scaling formula matches the design exactly:
  - `cx = (x_min + x_max) / 2.0` and `cy = (y_min + y_max) / 2.0` — center unchanged.
  - `half_w = (x_max - x_min) / 2.0 * s` and `half_h = (y_max - y_min) / 2.0 * s` — half-extents scaled by `s`.
  - New corners assigned from `cx ± half_w`, `cy ± half_h` — correct.
- `sample["bbox"].copy()` is used before modification, avoiding aliasing.
- `if "bbox" not in sample: return sample` guard added for robustness — acceptable defensive code.
- Out-of-bound clamp is not applied in the jitter class; instead `CropPerson` handles it via `cv2.copyMakeBorder`. This is explicitly permitted by the design.
- Joint coordinates `sample["joints"]` are not modified — correct, because they are 3D metric root-relative values independent of pixel scale.

**`build_train_transform`:**
- Pipeline order: `[RandomScaleJitter(0.8, 1.2), CropPerson(out_h, out_w), SubtractRoot(), ToTensor()]` — matches the design spec exactly.

**`build_val_transform`:**
- Unchanged: `[CropPerson(out_h, out_w), SubtractRoot(), ToTensor()]` — correct as specified.

---

### config.py

All 16 required config fields verified against the design spec:

| Field | Expected | Actual | Match |
|-------|----------|--------|-------|
| `output_dir` | `runs/idea006/design002` | `/work/.../runs/idea006/design002` | ✓ |
| `arch` | `sapiens_0.3b` | `sapiens_0.3b` | ✓ |
| `img_h` | `IMG_H` (640) | `IMG_H` | ✓ |
| `img_w` | `IMG_W` (384) | `IMG_W` | ✓ |
| `head_hidden` | `256` | `256` | ✓ |
| `head_num_heads` | `8` | `8` | ✓ |
| `head_num_layers` | `4` | `4` | ✓ |
| `head_dropout` | `0.1` | `0.1` | ✓ |
| `drop_path` | `0.1` | `0.1` | ✓ |
| `epochs` | `20` | `20` | ✓ |
| `lr_backbone` | `1e-5` | `1e-5` | ✓ |
| `lr_head` | `1e-4` | `1e-4` | ✓ |
| `weight_decay` | `0.03` | `0.03` | ✓ |
| `warmup_epochs` | `3` | `3` | ✓ |
| `grad_clip` | `1.0` | `1.0` | ✓ |
| `lambda_depth` | `0.1` | `0.1` | ✓ |
| `lambda_uv` | `0.2` | `0.2` | ✓ |

No experiment-specific values are hardcoded in `train.py`.

---

### train.py

Unchanged from baseline as required by the design. No modifications to the training loop, loss computation, or optimizer setup.

---

### model.py

Unchanged from baseline as required by the design.

---

### Issues Found

None.

---

**APPROVED**
