# design001 — Horizontal Flip Augmentation

## Starting Point

**baseline/** (`baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Problem

The baseline training transform applies zero augmentation (identical to val transform):
`CropPerson → SubtractRoot → ToTensor`. The model overfits to a single scene, yielding a
35–60 mm train-val MPJPE gap across all completed ideas.

## Proposed Solution

Add a `RandomHorizontalFlip` augmentation (p=0.5) into `build_train_transform()` only.
Horizontal flipping doubles the effective viewpoint diversity at near-zero computational cost.
Joint coordinates, pelvis UV, and left-right joint assignments must all be updated
consistently with the image flip.

---

## Mathematical / Implementation Detail

### Coordinate Convention

From `SubtractRoot`:
```
u_px = K[0,0] * (-Y / X) + K[0,2]
```
Horizontal pixel coordinate `u_px` depends on the camera-relative Y (lateral) axis.
When we flip the image horizontally:
- `u_px → (crop_w - 1 - u_px)` i.e. left-right mirror around image center.
- This is equivalent to negating Y in camera space: `Y → -Y`.
- X (depth forward axis) and Z (vertical) are unchanged.

Therefore the joint transform under horizontal flip is:
```
joints[:, 1] *= -1   # negate Y (lateral) component of all 70 joints
```

Left-right joint pairs must be swapped. `FLIP_PAIRS` from `infra.py` (already in
remapped 0..69 index space) provides exactly these pairs:
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

For `pelvis_uv`: `pelvis_uv[0]` is the normalized horizontal UV coordinate
(`u_px / crop_w * 2 - 1`). Under horizontal flip this negates: `pelvis_uv[0] *= -1`.
`pelvis_uv[1]` (vertical) is unchanged. `pelvis_depth` (metric depth) is flip-invariant.

### Transform Class

`RandomHorizontalFlip` is inserted **after `CropPerson` and before `SubtractRoot`** so
that the crop is already at output resolution and SubtractRoot can then compute
`pelvis_uv` correctly from the (already-flipped) image dimensions.

```python
class RandomHorizontalFlip:
    """Randomly flip RGB + depth horizontally (p=0.5) and update joint coords."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if np.random.random() >= self.p:
            return sample

        # 1. Flip images
        sample["rgb"]   = cv2.flip(sample["rgb"],   1)  # flipCode=1 → horizontal
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

`pelvis_uv[0]` negation is handled **implicitly**: after the Y-negate in step 2 and
swap in step 3, `SubtractRoot` recomputes `pelvis_uv` from the flipped joints and the
flipped intrinsics — the formula `u_px = K[0,0]*(-Y/X) + K[0,2]` naturally produces the
mirrored UV because Y has been negated.

> Note: `sample["intrinsic"]` (K matrix) does **not** need modification. The crop
> intrinsics are already in the cropped-image frame; the horizontal flip does not
> change focal lengths or the optical centre within a symmetric crop. The u_px formula
> applied on negated-Y joints will land at the correctly mirrored pixel location.

### Updated `build_train_transform`

```python
def build_train_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
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
| `transforms.py` | Add `RandomHorizontalFlip` class; import `FLIP_PAIRS` from `infra`; update `build_train_transform` |
| `config.py` | Update `output_dir` to `runs/idea006/design001` |
| `train.py` | No changes needed (unchanged from baseline) |
| `model.py` | No changes needed (unchanged from baseline) |

---

## config.py Fields

All values are identical to baseline **except** `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design001"
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

Horizontal flip is the single highest-leverage geometric augmentation for human pose
estimation. Doubling effective scene diversity should reduce the train-val gap.
Val MPJPE improvement of 5–15 mm over the baseline ~121 mm is plausible.
