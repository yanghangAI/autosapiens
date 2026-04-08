# design006 — Full Augmentation Stack (Horizontal Flip + Color Jitter + Depth Noise)

## Starting Point

**baseline/** (`baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Problem

The baseline training transform applies zero augmentation (identical to val transform):
`CropPerson → SubtractRoot → ToTensor`. The model overfits to a single scene, yielding a
35–60 mm train-val MPJPE gap across all completed ideas. Designs 001, 003, and 004 each
tested one augmentation type in isolation:

- design001: Horizontal Flip (p=0.5) — geometric diversity
- design003: Color Jitter (RGB) — photometric diversity
- design004: Depth Noise + Dropout — depth-modality robustness

This design combines all three into a single full-stack augmentation pipeline to assess
whether the benefits compound and represent the effective upper bound of augmentation gain
within the 20-epoch budget. Scale jitter (design002) is intentionally excluded to avoid
crop instability when combined with the other augmentations (as specified in idea.md Axis 6).

## Proposed Solution

Add `RandomHorizontalFlip`, `RGBColorJitter`, and `DepthAugmentation` to
`build_train_transform()` only. Each augmentation is applied independently at its
appropriate stage in the pipeline:

- `RandomHorizontalFlip` (p=0.5): inserted **after `CropPerson`, before `SubtractRoot`**
  so that `SubtractRoot` operates on the already-flipped image and correctly recomputes
  `pelvis_uv` from the negated-Y joint coordinates.
- `RGBColorJitter`: inserted **after `ToTensor`** to operate on float tensors; depth is
  untouched.
- `DepthAugmentation`: inserted **after `ToTensor`** (after `RGBColorJitter`); RGB and
  joint labels are untouched.

`build_val_transform` is entirely unchanged.

---

## Mathematical / Implementation Detail

### Transform 1 — RandomHorizontalFlip

Identical to design001. Under horizontal flip:

- `sample["rgb"]`: flipped with `cv2.flip(rgb, 1)`.
- `sample["depth"]`: flipped with `cv2.flip(depth, 1)`.
- `joints[:, 1] *= -1.0`: negates the lateral (Y) camera-space coordinate.
- Left-right joint pairs are swapped using `FLIP_PAIRS` (0..69 remapped indices from
  `infra.py`):

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

`pelvis_uv[0]` negation is handled **implicitly**: `SubtractRoot` runs after the flip and
recomputes `pelvis_uv` from the negated-Y joints via
`u_px = K[0,0]*(-Y/X) + K[0,2]`, which naturally produces the mirrored UV.
`pelvis_depth` (metric depth) is flip-invariant. `sample["intrinsic"]` (K matrix) does
not need modification.

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

### Transform 2 — RGBColorJitter

Identical to design003. Applied after `ToTensor` on the float RGB tensor only.

Parameters (from idea.md Axis 3):
```
brightness = 0.4   # [0.6, 1.4]
contrast   = 0.4   # [0.6, 1.4]
saturation = 0.2   # [0.8, 1.2]
hue        = 0.1   # [-0.1, 0.1]
```

```python
import torchvision.transforms as T

class RGBColorJitter:
    """Apply ColorJitter to the RGB tensor only; depth is unchanged."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample: dict) -> dict:
        # sample["rgb"] is a float32 tensor of shape (3, H, W) in [0, 1]
        sample["rgb"] = self.jitter(sample["rgb"])
        return sample
```

### Transform 3 — DepthAugmentation

Identical to design004. Applied after `ToTensor` (after `RGBColorJitter`) on the
normalized depth tensor. Each perturbation fires independently with `p=0.5`:

1. **Additive Gaussian noise**: `depth += N(0, 0.02²)`, clamped to `[0, 1]`.
2. **Random pixel dropout**: 10% of depth pixels set to 0 (Bernoulli mask).

```python
class DepthAugmentation:
    """
    Apply Gaussian noise and/or random pixel dropout to the depth channel only.

    Each perturbation fires independently with probability p (default 0.5).
    RGB and joint labels are untouched.
    """

    def __init__(
        self,
        noise_sigma: float = 0.02,
        dropout_rate: float = 0.10,
        p: float = 0.5,
    ):
        self.noise_sigma = noise_sigma
        self.dropout_rate = dropout_rate
        self.p = p

    def __call__(self, sample: dict) -> dict:
        depth = sample["depth"]  # float32 tensor, shape (1, H, W) or (H, W), in [0, 1]

        # --- Perturbation 1: Additive Gaussian noise ---
        if np.random.random() < self.p:
            noise = torch.randn_like(depth) * self.noise_sigma
            depth = (depth + noise).clamp(0.0, 1.0)

        # --- Perturbation 2: Random pixel dropout ---
        if np.random.random() < self.p:
            keep_mask = torch.bernoulli(
                torch.full(depth.shape, 1.0 - self.dropout_rate, dtype=depth.dtype)
            )
            depth = depth * keep_mask

        sample["depth"] = depth
        return sample
```

### Full Training Pipeline

```python
def build_train_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        CropPerson(out_h, out_w),
        RandomHorizontalFlip(p=0.5),          # geometric — must be before SubtractRoot
        SubtractRoot(),
        ToTensor(),
        RGBColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        DepthAugmentation(noise_sigma=0.02, dropout_rate=0.10, p=0.5),
    ])
```

`build_val_transform` is unchanged:
```python
def build_val_transform(out_h: int, out_w: int) -> Compose:
    return Compose([CropPerson(out_h, out_w), SubtractRoot(), ToTensor()])
```

### Pipeline Ordering Rationale

| Stage | Reason |
|-------|--------|
| `CropPerson` first | Crops the RGBD image to the person bounding box |
| `RandomHorizontalFlip` before `SubtractRoot` | Flip must happen before `SubtractRoot` so `pelvis_uv` is computed on the already-flipped crop; Y-negation is handled implicitly |
| `SubtractRoot` | Computes root-relative joint coordinates and `pelvis_uv` / `pelvis_depth` |
| `ToTensor` | Converts numpy arrays to float32 tensors in `[0, 1]` |
| `RGBColorJitter` after `ToTensor` | torchvision operates on float tensors; depth untouched |
| `DepthAugmentation` after `ToTensor` | Requires float tensor in `[0, 1]`; RGB untouched |

---

## Files to Modify

| File | Change |
|------|--------|
| `transforms.py` | Add `RandomHorizontalFlip`, `RGBColorJitter`, `DepthAugmentation` classes; import `FLIP_PAIRS` from `infra`, `cv2`, `numpy`, `torch`, `torchvision.transforms`; update `build_train_transform` |
| `config.py` | Update `output_dir` to `runs/idea006/design006` |
| `train.py` | No changes needed (unchanged from baseline) |
| `model.py` | No changes needed (unchanged from baseline) |

---

## config.py Fields

All values are identical to baseline **except** `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design006"
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

Design006 is the "kitchen sink" experiment combining all three orthogonal augmentation
strategies tested individually in designs 001, 003, and 004:

- **Horizontal flip** (design001): addresses geometric/viewpoint diversity, the highest-
  leverage augmentation for body pose estimation.
- **Color jitter** (design003): addresses photometric/appearance diversity, teaches the
  backbone to be invariant to lighting.
- **Depth noise + dropout** (design004): addresses depth-modality robustness, teaches
  the model to handle realistic sensor noise.

Since the three augmentations operate on independent aspects of the input signal
(geometric structure, RGB appearance, depth quality) and none of them introduce label
noise, their effects are expected to be largely additive. If each individually provides
2–15 mm improvement, the combined stack could yield 8–25 mm improvement over the ~121 mm
baseline val MPJPE — potentially the best-performing design in this idea.

The main risk is that the combined augmentation may slow convergence within the 20-epoch
budget, since the model sees a more varied distribution each epoch. However, given that
the training set is small (6,314 frames) and heavily overfits without any augmentation,
the regularisation benefit should outweigh the convergence cost.
