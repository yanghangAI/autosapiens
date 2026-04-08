# design003 — Color Jitter Augmentation (RGB Only)

## Starting Point

**baseline/** (`baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Problem

The baseline training transform applies zero augmentation (identical to val transform):
`CropPerson → SubtractRoot → ToTensor`. The model overfits to a single scene, yielding a
35–60 mm train-val MPJPE gap across all completed ideas. Since the training set comes from
a single scene with fixed lighting conditions, the model likely memorizes appearance
statistics (brightness, contrast, colour tone) that do not generalise to held-out scenes.

## Proposed Solution

Add `torchvision.transforms.ColorJitter` to `build_train_transform()` only, applied to
the **RGB channels** after `ToTensor`. The depth channel is left entirely unchanged.
Color jitter randomly perturbs brightness, contrast, saturation, and hue, making the
backbone features invariant to photometric variation — which is irrelevant to 3D pose
geometry. No joint coordinates or intrinsics are affected.

---

## Mathematical / Implementation Detail

### Why Color Jitter Does Not Affect Joint Labels

`sample["joints"]` contains 3D root-relative coordinates in metric camera space (mm).
Color jitter operates purely on pixel intensities; it does not change the 3D geometry or
camera parameters. Therefore:

- `sample["joints"]`: **no change needed**
- `sample["intrinsic"]` (K matrix): **no change needed**
- `pelvis_uv`, `pelvis_depth`: **no change needed**
- `sample["depth"]`: **not modified** (depth encodes metric distance, not photometric info)

### ColorJitter Parameters

Following the specification in `idea.md` (Axis 3):

```
brightness = 0.4   # multiplicative factor sampled from [1-0.4, 1+0.4] = [0.6, 1.4]
contrast   = 0.4   # similarly [0.6, 1.4]
saturation = 0.2   # [0.8, 1.2]
hue        = 0.1   # shift in [-0.1, 0.1] (fraction of full hue circle)
```

These are moderate values: strong enough to introduce meaningful photometric diversity
across 20 training epochs, but not so extreme as to make RGB features uninformative.

### Applying ColorJitter to RGB Only

After `ToTensor`, the sample dict contains:
- `sample["rgb"]`: float32 tensor, shape `(3, H, W)`, range `[0, 1]`
- `sample["depth"]`: float32 tensor (depth map), shape `(1, H, W)` or `(H, W)`

`torchvision.transforms.ColorJitter` operates on PIL images or `(C, H, W)` float tensors.
Applying it directly to `sample["rgb"]` (3-channel) leaves `sample["depth"]` untouched.

A thin wrapper is used to integrate ColorJitter into the sample-dict pipeline:

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

> **Builder note:** `T.ColorJitter` accepts `torch.Tensor` directly (torchvision >= 0.8).
> Each `__call__` applies a freshly sampled random transform (per-image randomisation is
> handled internally). No manual seed management is required.

### Updated `build_train_transform`

`RGBColorJitter` is inserted **after `ToTensor`** so it operates on float tensors
(required by torchvision's tensor API) and after `SubtractRoot` has already computed
`pelvis_uv` and `pelvis_depth` from unaltered data:

```python
def build_train_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        CropPerson(out_h, out_w),
        SubtractRoot(),
        ToTensor(),
        RGBColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
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
| `transforms.py` | Add `RGBColorJitter` class; import `torchvision.transforms`; update `build_train_transform` |
| `config.py` | Update `output_dir` to `runs/idea006/design003` |
| `train.py` | No changes needed (unchanged from baseline) |
| `model.py` | No changes needed (unchanged from baseline) |

---

## config.py Fields

All values are identical to baseline **except** `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea006/design003"
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

Color jitter teaches the backbone to extract pose-relevant features rather than
lighting-specific appearance statistics. Because 3D joint positions are photometrically
invariant by definition, no label noise is introduced. The effect is complementary to
geometric augmentations (flip, scale jitter): those address viewpoint diversity, while
color jitter addresses appearance diversity. Val MPJPE improvement of 2–8 mm over the
~121 mm baseline is plausible. The gain may be smaller than horizontal flip (design001)
since the train-val domain gap is primarily geometric rather than photometric, but the
augmentation is zero-cost in terms of label correctness.
