# design002 — Scale/Crop Jitter Augmentation

## Starting Point

**baseline/** (`baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Problem

The baseline training transform applies zero augmentation (identical to val transform):
`CropPerson → SubtractRoot → ToTensor`. The model overfits to a single scene, yielding a
35–60 mm train-val MPJPE gap across all completed ideas. The person always appears at the
same relative scale within each crop, allowing the model to exploit scale memorization
rather than learning scale-invariant pose features.

## Proposed Solution

Add a `RandomScaleJitter` augmentation that randomly scales the bounding box by a factor
sampled from `Uniform(0.8, 1.2)` **before** `CropPerson`. This makes the person appear at
varying scales within the fixed output crop, teaching the model to be scale-invariant.
Joint root-relative coordinates (3D camera-space metric values) do **not** require
adjustment since they are independent of pixel scale.

---

## Mathematical / Implementation Detail

### Why Joint Coordinates Are Unaffected

`CropPerson` crops the image around the bbox and resizes to a fixed `(out_h, out_w)`.
The labels `sample["joints"]` are 3D root-relative coordinates in metric camera space
(mm), not pixel coordinates. Scale jitter changes which pixel region is cropped and
resized, but the underlying 3D geometry — and thus the root-relative joint positions —
is identical. Consequently:

- `sample["joints"]`: **no change needed**
- `sample["intrinsic"]` (K matrix used inside `CropPerson`): handled internally by
  `CropPerson` which recomputes the crop-space intrinsics from the new bbox extents
- `pelvis_uv`, `pelvis_depth`: computed by `SubtractRoot` after `CropPerson` from the
  (potentially jittered) crop — no explicit correction needed

### Bbox Scaling Formula

Given the original bbox `(cx, cy, w, h)` (centre + half-extents or similar), sample:
```
s ~ Uniform(0.8, 1.2)
```
Scale both width and height by `s` around the existing bbox centre:
```
w_new = s * w
h_new = s * h
```
The centre `(cx, cy)` is unchanged. Clip to image boundaries if needed (or let
`CropPerson` handle out-of-bounds via padding — verify against `CropPerson`'s existing
boundary logic before clipping).

### Determining bbox Format

The `sample["bbox"]` format must be inspected from the dataset loader. Common formats:
- `[x_min, y_min, x_max, y_max]` — scale around center: compute `cx = (x_min+x_max)/2`,
  `cy = (y_min+y_max)/2`, then `x_min_new = cx - s*(cx-x_min)`, etc.
- `[cx, cy, w, h]` — direct: `w_new = s*w`, `h_new = s*h`.

The Builder should check `dataset.py` or `infra.py` for the bbox convention and apply the
appropriate formula. Either way, the intent is to scale half-extents by `s` around center.

### Transform Class

`RandomScaleJitter` is inserted **before `CropPerson`** so the jittered bbox is consumed
by the crop step:

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
        bbox = sample["bbox"].copy()   # shape depends on convention; see note above

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
```

> **Builder note:** Verify the bbox format in `dataset.py` / `infra.py` and use the
> appropriate block. The `[x_min, y_min, x_max, y_max]` block is shown as the primary
> candidate. If out-of-bound jitter causes `CropPerson` to fail, add a clamp:
> `bbox[0] = max(0, bbox[0])`, etc., or confirm `CropPerson` already handles it.

### Updated `build_train_transform`

```python
def build_train_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        RandomScaleJitter(low=0.8, high=1.2),
        CropPerson(out_h, out_w),
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
| `transforms.py` | Add `RandomScaleJitter` class; update `build_train_transform` |
| `config.py` | Update `output_dir` to `runs/idea006/design002` |
| `train.py` | No changes needed (unchanged from baseline) |
| `model.py` | No changes needed (unchanged from baseline) |

---

## config.py Fields

All values are identical to baseline **except** `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design002"
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

Scale jitter forces the model to regress correct 3D poses from a wider range of apparent
person sizes. This combats scale memorization from the single-scene training set. Because
the root-relative 3D labels are scale-invariant by construction, there is no label noise
introduced by this augmentation. Val MPJPE improvement of 3–10 mm over the ~121 mm
baseline is plausible, likely smaller than horizontal flip but orthogonal in effect.
