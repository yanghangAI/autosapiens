# design005 — Combined Geometric Augmentation (Horizontal Flip + Scale Jitter)

## Starting Point

**baseline/** (`baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Problem

The baseline training transform applies zero augmentation (identical to val transform):
`CropPerson → SubtractRoot → ToTensor`. The model overfits to a single scene, yielding a
35–60 mm train-val MPJPE gap across all completed ideas. design001 (flip only) and
design002 (scale jitter only) address orthogonal aspects of this gap: flip doubles
left-right viewpoint diversity while scale jitter forces scale invariance. This design
combines both geometric augmentations to test whether their effects compound positively.

## Proposed Solution

Apply both `RandomScaleJitter` (±20%, before `CropPerson`) and `RandomHorizontalFlip`
(p=0.5, after `CropPerson`, before `SubtractRoot`) in `build_train_transform()`. The
ordering is critical: scale jitter must precede `CropPerson` (it modifies the bbox), and
flip must follow `CropPerson` (it operates on the already-cropped image array) but precede
`SubtractRoot` (so `SubtractRoot` sees the flipped coordinate system and computes
`pelvis_uv` correctly from the negated-Y joints).

---

## Mathematical / Implementation Detail

### Transform Ordering and Rationale

```
RandomScaleJitter(0.8, 1.2)   ← jitter bbox BEFORE crop
  → CropPerson(out_h, out_w)  ← crop at jittered scale
    → RandomHorizontalFlip(p=0.5) ← flip AFTER crop, BEFORE SubtractRoot
      → SubtractRoot()         ← recomputes pelvis_uv from (possibly flipped) joints
        → ToTensor()
```

This ordering ensures:
1. `RandomScaleJitter` sees the original bbox and modifies it before `CropPerson` consumes it.
2. `RandomHorizontalFlip` operates on the fixed-resolution crop output (H×W array), not
   on raw imagery of varying size.
3. `SubtractRoot` computes `pelvis_uv` after any flip, so it naturally absorbs the
   Y-negation without requiring a separate `pelvis_uv` correction step.

### RandomScaleJitter (carried forward from design002, no changes)

Sample `s ~ Uniform(0.8, 1.2)` and scale bbox half-extents by `s` around the bbox centre:

```
s ~ Uniform(0.8, 1.2)
```

For `[x_min, y_min, x_max, y_max]` convention:
```
cx = (x_min + x_max) / 2
cy = (y_min + y_max) / 2
half_w_new = (x_max - x_min) / 2 * s
half_h_new = (y_max - y_min) / 2 * s
x_min_new = cx - half_w_new
x_max_new = cx + half_w_new
y_min_new = cy - half_h_new
y_max_new = cy + half_h_new
```

Joint 3D root-relative coordinates (metric camera space, mm) are **not affected** — they
are independent of the pixel-space crop scale.

### RandomHorizontalFlip (carried forward from design001, no changes)

When a flip fires (probability 0.5):

1. **Flip images:** `cv2.flip(rgb, 1)` and `cv2.flip(depth, 1)` (flipCode=1 = horizontal).
2. **Negate Y (lateral) axis:** `joints[:, 1] *= -1.0` for all 70 joints. In the SMPLX
   camera-relative convention, `u_px = K[0,0] * (-Y / X) + K[0,2]`, so negating Y mirrors
   the horizontal pixel coordinate.
3. **Swap left-right joint pairs** using `FLIP_PAIRS` (0..69 remapped indices from `infra.py`):

```python
FLIP_PAIRS = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14),
    (16, 17), (18, 19), (20, 21), (23, 24),
    (25, 40), (26, 41), (27, 42),
    (28, 43), (29, 44), (30, 45),
    (31, 46), (32, 47), (33, 48),
    (34, 49), (35, 50), (36, 51),
    (37, 52), (38, 53), (39, 54),
)
```

`pelvis_uv[0]` negation is handled **implicitly**: after Y-negation and left-right swap,
`SubtractRoot` recomputes `pelvis_uv` from the flipped joints via
`u_px = K[0,0]*(-Y/X) + K[0,2]`, which naturally yields the mirrored UV. No explicit
`pelvis_uv` update is required. `pelvis_depth` is flip-invariant and unchanged.

> Note: `sample["intrinsic"]` (K matrix) does **not** need modification. The flip does not
> change focal lengths or the optical centre within a symmetric crop.

### Transform Classes

```python
class RandomScaleJitter:
    """Randomly scale the person bbox by s~Uniform(low, high) around its centre.

    Must be applied before CropPerson. Joint coordinates are unaffected.
    """

    def __init__(self, low: float = 0.8, high: float = 1.2):
        self.low = low
        self.high = high

    def __call__(self, sample: dict) -> dict:
        s = np.random.uniform(self.low, self.high)
        bbox = sample["bbox"].copy()

        # ---- [x_min, y_min, x_max, y_max] convention ----
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        half_w = (x_max - x_min) / 2.0 * s
        half_h = (y_max - y_min) / 2.0 * s
        bbox[0] = cx - half_w
        bbox[1] = cy - half_h
        bbox[2] = cx + half_w
        bbox[3] = cy + half_h
        # --------------------------------------------------
        # If bbox is [cx, cy, w, h] instead, replace the block above with:
        #   bbox[2] = bbox[2] * s   # w
        #   bbox[3] = bbox[3] * s   # h

        sample["bbox"] = bbox
        return sample


class RandomHorizontalFlip:
    """Randomly flip RGB + depth horizontally (p=0.5) and update joint coords."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if np.random.random() >= self.p:
            return sample

        # 1. Flip images
        sample["rgb"]   = cv2.flip(sample["rgb"],   1)
        if sample.get("depth") is not None:
            sample["depth"] = cv2.flip(sample["depth"], 1)

        # 2. Negate Y (lateral) axis of all joints
        joints = sample["joints"].copy()
        joints[:, 1] *= -1.0

        # 3. Swap left-right joint pairs
        for left_idx, right_idx in FLIP_PAIRS:
            joints[left_idx], joints[right_idx] = (
                joints[right_idx].copy(), joints[left_idx].copy()
            )
        sample["joints"] = joints

        return sample
```

### Updated `build_train_transform`

```python
def build_train_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        RandomScaleJitter(low=0.8, high=1.2),
        CropPerson(out_h, out_w),
        RandomHorizontalFlip(p=0.5),
        SubtractRoot(),
        ToTensor(),
    ])
```

`build_val_transform` is unchanged:
```python
def build_val_transform(out_h: int, out_w: int) -> Compose:
    return Compose([CropPerson(out_h, out_w), SubtractRoot(), ToTensor()])
```

---

## Files to Modify

| File | Change |
|------|--------|
| `transforms.py` | Add `RandomScaleJitter` and `RandomHorizontalFlip` classes; import `FLIP_PAIRS` from `infra`; update `build_train_transform` |
| `config.py` | Update `output_dir` to `runs/idea006/design005` |
| `train.py` | No changes needed (unchanged from baseline) |
| `model.py` | No changes needed (unchanged from baseline) |

> **Builder note on bbox format:** As in design002, verify `sample["bbox"]` format in
> `dataset.py` / `infra.py`. The `[x_min, y_min, x_max, y_max]` block is the primary
> candidate. If out-of-bound jitter causes `CropPerson` to fail, add a clamp or confirm
> `CropPerson` already handles it via padding.

---

## config.py Fields

All values are identical to baseline **except** `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design005"
arch        = "sapiens_0.3b"
img_h       = IMG_H          # 640
img_w       = IMG_W          # 384
head_hidden     = 256
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
epochs       = 20
lr_backbone  = 1e-5
lr_head      = 1e-4
weight_decay = 0.03
warmup_epochs= 3
grad_clip    = 1.0
lambda_depth = 0.1
lambda_uv    = 0.2
```

---

## Expected Outcome

This design directly tests whether the geometric augmentations from design001 (flip) and
design002 (scale jitter) compound positively or whether one dominates. The two augmentations
are orthogonal: flip increases left-right viewpoint diversity; scale jitter forces
scale invariance. Neither augmentation modifies the 3D root-relative joint labels, so no
label noise is introduced.

If the effects add roughly linearly, val MPJPE improvement of 8–20 mm over the ~121 mm
baseline is plausible (design001 expected 5–15 mm, design002 expected 3–10 mm). If one
dominates, the gain will be closer to the stronger single augmentation. Diminishing returns
are possible if both augmentations target similar sources of overfitting, but the orthogonal
nature of viewpoint diversity (flip) vs. scale invariance (scale jitter) makes compound
benefit the more likely outcome.
