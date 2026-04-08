# design004 — Depth Channel Augmentation (Gaussian Noise + Pixel Dropout)

## Starting Point

**baseline/** (`baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Problem

The baseline training transform applies zero augmentation (identical to val transform):
`CropPerson → SubtractRoot → ToTensor`. The model overfits to a single scene, yielding a
35–60 mm train-val MPJPE gap across all completed ideas. Because the training data comes
from a single RGBD capture, the model may over-rely on clean, noise-free depth maps that
do not reflect real-world RGBD sensor behaviour (missing pixels, quantisation noise,
reflective-surface dropouts).

## Proposed Solution

Add two stochastic depth-channel perturbations to `build_train_transform()` only, applied
**after `ToTensor`** on the normalized depth tensor:

1. **Additive Gaussian noise** (`σ = 0.02`): simulates sensor measurement noise.
2. **Random pixel dropout** (10% of pixels zeroed): simulates sensor dropouts on
   reflective or out-of-range surfaces.

Each perturbation is applied independently with `p = 0.5`, so either, both, or neither
may fire on any given sample. RGB channels and all joint labels are completely unaffected.

---

## Mathematical / Implementation Detail

### Why Depth Augmentation Does Not Affect Joint Labels

`sample["joints"]` contains 3D root-relative coordinates in metric camera space (mm),
computed by `SubtractRoot` from the ground-truth skeleton — not from the depth image.
The depth map is a model **input** used by the backbone; it does not determine labels.
Therefore:

- `sample["joints"]`: **no change needed**
- `sample["intrinsic"]` (K matrix): **no change needed**
- `pelvis_uv`, `pelvis_depth`: **no change needed** (these come from skeleton GT, not
  the depth image)
- `sample["rgb"]`: **not modified**

### Depth Tensor Convention

After `ToTensor`, `sample["depth"]` is a float32 tensor of shape `(1, H, W)` normalized
to `[0, 1]` (where 0 = no depth / clipped minimum, 1 = maximum range value). Both
perturbations operate in this normalized space and clamp the result back to `[0, 1]`.

### Perturbation 1 — Additive Gaussian Noise

```
depth_noisy = depth + ε,   ε ~ N(0, σ²)  with σ = 0.02
depth_noisy = clamp(depth_noisy, 0.0, 1.0)
```

`σ = 0.02` corresponds to roughly ±4 cm of noise for a 2-metre depth range — consistent
with the noise floor of commodity structured-light and ToF sensors.

Applied independently with `p = 0.5` per sample.

### Perturbation 2 — Random Pixel Dropout

A random binary mask `M` of shape `(1, H, W)` is sampled with Bernoulli probability
`p_drop = 0.10` of being 0 (dropout) and `1 - p_drop = 0.90` of being 1 (keep):

```
M ~ Bernoulli(1 - p_drop)    element-wise, p_drop = 0.10
depth_dropped = depth * M
```

Setting dropped pixels to 0 matches the sensor convention where 0 means "no valid
measurement". Applied independently with `p = 0.5` per sample.

### Transform Class

Both perturbations are combined in a single `DepthAugmentation` class for simplicity.
Each sub-perturbation is gated by its own independent Bernoulli draw at the sample level:

```python
import torch
import numpy as np

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
            # keep_mask: 1 = keep, 0 = drop
            keep_mask = torch.bernoulli(
                torch.full(depth.shape, 1.0 - self.dropout_rate, dtype=depth.dtype)
            )
            depth = depth * keep_mask

        sample["depth"] = depth
        return sample
```

> **Builder note:** `torch.randn_like` and `torch.bernoulli` respect the tensor's device
> (CPU during data loading). No manual seed management is required; PyTorch's DataLoader
> worker seeds handle reproducibility.

### Placement in the Pipeline

`DepthAugmentation` is inserted **after `ToTensor`** so it operates on float tensors
already in `[0, 1]`. Placing it after `SubtractRoot` ensures that `pelvis_uv` and
`pelvis_depth` (which are derived from GT skeleton, not from the depth image) have already
been computed from clean data:

```python
def build_train_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        CropPerson(out_h, out_w),
        SubtractRoot(),
        ToTensor(),
        DepthAugmentation(noise_sigma=0.02, dropout_rate=0.10, p=0.5),
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
| `transforms.py` | Add `DepthAugmentation` class; update `build_train_transform` |
| `config.py` | Update `output_dir` to `runs/idea006/design004` |
| `train.py` | No changes needed (unchanged from baseline) |
| `model.py` | No changes needed (unchanged from baseline) |

---

## config.py Fields

All values are identical to baseline **except** `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design004"
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

Depth channel augmentation teaches the model to be robust to sensor noise and missing
depth pixels — both of which occur in real RGBD deployments. Because the perturbations
are applied only to the model input (not the labels), no label noise is introduced. The
effect is complementary to geometric (flip, scale jitter) and photometric (color jitter)
augmentations: it specifically targets the depth modality.

Expected val MPJPE improvement: 2–6 mm over the ~121 mm baseline. The gain is likely
smaller than horizontal flip (design001) since depth noise alone does not increase
viewpoint or pose diversity, but it should reduce the model's sensitivity to clean-depth
overfitting and improve generalization to noisier real-world sensors.
