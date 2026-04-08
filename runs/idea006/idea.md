# idea006 — Training Data Augmentation for Generalization

**Expected Designs:** 6

**Baseline starting point:** `baseline.py` (equivalently `baseline/train.py`, `baseline/model.py`, `baseline/transforms.py`, `baseline/config.py`)

---

## Motivation & Broader Reflection

### Key finding from results.csv

| Idea | Best val MPJPE | Best train MPJPE | Train–val gap |
|------|---------------|-----------------|---------------|
| idea005/design001 | 121.4 mm | 86.1 mm | ~35 mm |
| idea004/design002 | 130.7 mm | 70.4 mm | ~60 mm |
| idea002/design003 | 139.5 mm | 92.1 mm | ~47 mm |
| idea001/design002 | 139.6 mm | 86.9 mm | ~53 mm |

Across **all** five completed ideas, the training MPJPE is consistently much lower than validation MPJPE — a persistent 35–60 mm train-val gap. This pattern strongly indicates that the model is overfitting to the small training set (6,314 frames from a single scene `20241213_1_250_rome_tracking`).

### The unexplored axis: Data Augmentation

The baseline `build_train_transform` is **identical** to `build_val_transform`:
```
CropPerson → SubtractRoot → ToTensor
```
**Zero augmentation is applied during training.** The README explicitly notes: *"Transforms (train = val, no augmentation)"*.

This is the most prominent unexplored axis. No prior idea has touched augmentation. In pose estimation literature, geometric augmentations (flip, scale jitter, crop jitter) and photometric augmentations (color jitter, depth noise) are standard practice and consistently reduce generalization error.

### Why augmentation could improve val MPJPE meaningfully

1. **The training set is small and from a single scene.** The model likely memorizes appearance and viewpoint distributions. Augmentation forces invariance.
2. **Geometric augmentations** (flip, scale jitter) increase effective pose/viewpoint diversity without requiring additional data.
3. **Photometric augmentations** (color jitter) make the RGB features more invariant to lighting, which is irrelevant to 3D pose.
4. **Depth-channel perturbations** (noise, dropout) teach the model to be robust to depth sensor noise, which is realistic in RGBD data.

### Constraint: Joint correctness under geometric transforms

When flipping horizontally or scaling the crop, **joint coordinates must be transformed consistently** with the image. This requires careful implementation. The Designer must ensure:
- Horizontal flip: negate `joints[:, 1]` (Y-axis in the SMPLX camera-relative convention) and swap left-right joint indices using `SMPLX_FLIP_MAP` from `infra.py` (if available) or manually constructed from `SMPLX_SKELETON`.
- Scale jitter: jitter the bbox scale before passing to `CropPerson`; joint coordinates in the output are root-relative so they do **not** need scaling (root-relative coordinates are camera-space 3D, not pixel-space).
- Color jitter: RGB only, depth unchanged.
- Depth noise: depth channel only, RGB unchanged.

---

## Design Axes

### Axis 1 — Horizontal Flip
Random horizontal flip of the RGB+depth crop with joint left-right swap. Classic and high-impact for body pose estimation. This doubles effective data diversity at near-zero cost.

### Axis 2 — Scale/Crop Jitter
Random bbox scale jitter (±20%) before `CropPerson`. Varies the person scale in the image, teaching the model to be scale-invariant. Implementation: sample a random scale factor in [0.8, 1.2] and multiply the bbox side lengths before crop.

### Axis 3 — Color Jitter (RGB only)
Apply `torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)` to the RGB tensor after `ToTensor`. Depth is not modified. Makes the backbone more invariant to lighting conditions.

### Axis 4 — Depth Channel Augmentation
Two sub-perturbations applied to the normalized depth map:
- Additive Gaussian noise: `depth += N(0, 0.02)` then clamp to [0, 1]
- Random pixel dropout: set a random 10% of depth pixels to 0 (simulate sensor dropouts)
Applied independently with p=0.5 each.

### Axis 5 — Combined Geometric Augmentation (Flip + Scale Jitter)
Both flip (p=0.5) and scale jitter (±20%) applied together. Tests whether geometric augmentations compound positively or if one dominates.

### Axis 6 — Full Augmentation Stack (Flip + Color Jitter + Depth Noise)
The full suite: horizontal flip (p=0.5) + color jitter + depth noise. This is the "kitchen sink" design to assess the upper bound of augmentation benefit under the 20-epoch budget. Scale jitter excluded to avoid crop instability when combined with other augmentations.

---

## Implementation Notes for the Designer

- All augmentations go into `build_train_transform()` in `transforms.py`. `build_val_transform()` stays unchanged.
- Horizontal flip: after `CropPerson` and before `ToTensor`, implement a `RandomHorizontalFlip` transform class. It must:
  1. Flip `sample["rgb"]` using `cv2.flip(rgb, 1)`.
  2. Flip `sample["depth"]` similarly.
  3. Negate the Y (lateral) component of all joints: `joints[:, 1] *= -1` (in the SMPLX convention where +X is forward/depth, +Y is left, +Z is up in camera frame — verify by checking `infra.py` comments or `SubtractRoot`).
  4. Swap left-right joint pairs. Consult `infra.py` for `ACTIVE_JOINT_INDICES` and `SMPLX_SKELETON`; or manually define the mirror pairs from SMPLX body topology (left/right hip, knee, ankle, shoulder, elbow, wrist).
  5. Update `pelvis_uv[0] *= -1` (horizontal UV coordinate flips).
  6. No change to `pelvis_depth` (depth/distance is flip-invariant).
- Scale jitter: implement as a `RandomScaleJitter` that modifies `sample["bbox"]` before passing to `CropPerson`. Randomly sample `s ~ Uniform(0.8, 1.2)` and scale bbox width and height by `s` around the bbox center. The joint root-relative coordinates do **not** need adjustment (they are in metric 3D space, not pixels).
- Color jitter: apply after `ToTensor` (operates on float tensors), using `torchvision.transforms.functional` for per-image randomization.
- Depth noise: apply after `ToTensor` on the `sample["depth"]` tensor (already in [0,1]).
- Each design has its own `train.py` / `transforms.py` pair in its design folder under `runs/idea006/designXXX/`.
- Baseline config (lr, epochs, model arch) stays at default values from `baseline/config.py`.

---

## Expected Number of Designs

**6 novel designs** (all augmentation variants, no re-implementation of the no-augmentation baseline):

| Design | Augmentation Applied |
|--------|----------------------|
| design001 | Horizontal Flip only (p=0.5) |
| design002 | Scale/Crop Jitter only (±20%) |
| design003 | Color Jitter only (RGB) |
| design004 | Depth Channel Augmentation only (noise + dropout) |
| design005 | Combined Geometric: Flip + Scale Jitter |
| design006 | Full Stack: Flip + Color Jitter + Depth Noise |
