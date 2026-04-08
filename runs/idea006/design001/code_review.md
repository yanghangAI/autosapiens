# Code Review — idea006/design001

**Design_ID:** idea006/design001
**Date:** 2026-04-08
**Verdict:** APPROVED

---

## Summary

The implementation correctly and completely follows the approved design for horizontal flip augmentation.

---

## Detailed Checks

### 1. `RandomHorizontalFlip` class (transforms.py, lines 182–208)

- **Trigger condition:** `np.random.random() >= self.p` → flips when random < p=0.5. Correct.
- **Image flip:** `cv2.flip(sample["rgb"], 1)` and same for depth. flipCode=1 is horizontal flip. Correct.
- **Depth guard:** `if sample.get("depth") is not None` — safely handles missing depth. Correct.
- **Joint Y-negation:** `joints[:, 1] *= -1.0` — negates the lateral (Y) axis of all 70 joints. Matches design spec exactly.
- **FLIP_PAIRS swap:** Iterates over `FLIP_PAIRS` imported from `infra`, swaps each left-right pair using `.copy()` to avoid aliasing. Correct.
- **`pelvis_uv` not manually negated:** Design specifies this is handled implicitly by SubtractRoot after Y-negation. Implementation correctly omits an explicit negation. Correct.

### 2. Pipeline ordering (transforms.py, lines 211–212)

```
CropPerson → RandomHorizontalFlip → SubtractRoot → ToTensor
```
Matches design spec exactly. Flip occurs after crop (on fixed-resolution image) and before SubtractRoot (so SubtractRoot recomputes pelvis_uv from flipped coordinates). Correct.

### 3. `build_val_transform` unchanged (lines 215–216)

Val pipeline is `CropPerson → SubtractRoot → ToTensor`. No augmentation. Correct.

### 4. `FLIP_PAIRS` import (line 15)

Imported from `infra` directly rather than redefined. Correct.

### 5. `config.py` — all required fields

| Field | Design spec | Implementation | Match |
|---|---|---|---|
| output_dir | `/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design001` | Same | ✓ |
| arch | `sapiens_0.3b` | Same | ✓ |
| img_h / img_w | IMG_H / IMG_W | Same (from infra) | ✓ |
| head_hidden | 256 | 256 | ✓ |
| head_num_heads | 8 | 8 | ✓ |
| head_num_layers | 4 | 4 | ✓ |
| head_dropout | 0.1 | 0.1 | ✓ |
| drop_path | 0.1 | 0.1 | ✓ |
| epochs | 20 | 20 | ✓ |
| lr_backbone | 1e-5 | 1e-5 | ✓ |
| lr_head | 1e-4 | 1e-4 | ✓ |
| weight_decay | 0.03 | 0.03 | ✓ |
| warmup_epochs | 3 | 3 | ✓ |
| grad_clip | 1.0 | 1.0 | ✓ |
| lambda_depth | 0.1 | 0.1 | ✓ |
| lambda_uv | 0.2 | 0.2 | ✓ |

All fields match exactly. No hardcoded experiment-specific values in `train.py`.

### 6. `train.py` and `model.py`

Design specifies these are unchanged from baseline. Both files are identical in structure to the baseline. No issues.

### 7. `build_train_transform` signature

Implementation has an extra `scale_jitter: bool = True` keyword argument that is unused (not passed to any transform). This is harmless — it's a dead parameter with a default value and does not affect behavior. No rejection warranted.

---

## Issues Found

**None critical.** One minor observation:

- `build_train_transform` has a `scale_jitter: bool = True` parameter that goes unused. This is inert (no code path reads it) and does not affect correctness or reproducibility. It may be a leftover stub for future use.

---

## Verdict: APPROVED
