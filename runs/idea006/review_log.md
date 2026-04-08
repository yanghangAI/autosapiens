# Review Log — idea006

---

## Entry: design001 | 2026-04-08 | APPROVED

**Design_ID:** idea006/design001  
**Design:** Horizontal Flip Augmentation (p=0.5)

**Summary:** Adds `RandomHorizontalFlip` after `CropPerson` and before `SubtractRoot`. Negates `joints[:, 1]`, swaps FLIP_PAIRS, flips RGB and depth images. `pelvis_uv` recomputed correctly by downstream `SubtractRoot`. No changes to `train.py` or `model.py`. All config fields explicitly specified. Computationally trivial; fits within 20-epoch / 1080Ti constraint.

**Verdict:** APPROVED — No issues found. Design is complete, mathematically correct, and unambiguous.

---

## Entry: design002 | 2026-04-08 | APPROVED

**Design_ID:** idea006/design002  
**Design:** Scale/Crop Jitter Augmentation (s ~ Uniform(0.8, 1.2))

**Summary:** Adds `RandomScaleJitter` before `CropPerson`. Scales bbox half-extents by `s` around center. Root-relative 3D joint coordinates require no adjustment (metric camera space). CropPerson handles out-of-bounds via existing zero-padding. Both `[x_min,y_min,x_max,y_max]` and `[cx,cy,w,h]` bbox formats covered with a Builder note to verify convention. All config fields explicitly specified. No model or train.py changes.

**Verdict:** APPROVED — No issues found. Design is complete, mathematically correct, and implementation-ready.

---

## Entry: design003 | 2026-04-08 | APPROVED

**Design_ID:** idea006/design003  
**Design:** Color Jitter Augmentation (RGB Only)

**Summary:** Adds `RGBColorJitter` wrapper class after `ToTensor` in `build_train_transform`. Applies `T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)` to `sample["rgb"]` only; depth channel left untouched. Joints and intrinsics correctly identified as photometrically invariant — no label adjustment needed. Full class code provided verbatim. Only `transforms.py` and `config.py` modified. All config fields explicitly specified.

**Verdict:** APPROVED — No issues found. Design is complete, mathematically correct, and unambiguous for the Builder.

---

## Entry: design006 | 2026-04-08 | APPROVED

**Design_ID:** idea006/design006  
**Design:** Full Augmentation Stack (Horizontal Flip + Color Jitter + Depth Noise)

**Summary:** Composes all three component augmentations from APPROVED designs 001, 003, and 004 into a single `build_train_transform` pipeline. Ordering: `CropPerson → RandomHorizontalFlip → SubtractRoot → ToTensor → RGBColorJitter → DepthAugmentation`. Each class is identical to its previously approved standalone version. `pelvis_uv` correctly not manually negated (SubtractRoot handles implicitly). All parameters match idea.md Axis 6 exactly. Scale jitter correctly excluded per spec. Only `transforms.py` and `config.py` modified. All config fields explicitly specified. Fits 20 epochs on 1080Ti.

**Verdict:** APPROVED — No issues found. Design is complete, mathematically correct, and implementation-ready.

---

## Entry: design005 | 2026-04-08 | APPROVED

**Design_ID:** idea006/design005  
**Design:** Combined Geometric Augmentation (Horizontal Flip p=0.5 + Scale Jitter ±20%)

**Summary:** Combines `RandomScaleJitter` (before `CropPerson`) and `RandomHorizontalFlip` (after `CropPerson`, before `SubtractRoot`) in `build_train_transform`. Transform ordering is correct. `RandomScaleJitter` formula identical to APPROVED design002; `RandomHorizontalFlip` logic identical to APPROVED design001 including FLIP_PAIRS. `pelvis_uv` is not manually negated — SubtractRoot recomputes implicitly (consistent with lesson learned). All config fields explicitly specified. Only `transforms.py` and `config.py` modified. Computationally trivial; fits 20 epochs on 1080Ti.

**Verdict:** APPROVED — No issues found. Design is complete, mathematically correct, and implementation-ready.

---

## Entry: design004 | 2026-04-08 | APPROVED

**Design_ID:** idea006/design004  
**Design:** Depth Channel Augmentation (Gaussian Noise σ=0.02, p=0.5 + Pixel Dropout 10%, p=0.5)

**Summary:** Adds `DepthAugmentation` class after `ToTensor` in `build_train_transform`. Applies additive Gaussian noise (clamped to [0,1]) and random pixel dropout (zeroing 10% of pixels) independently, each with p=0.5. Operates entirely on the depth tensor in float32 [0,1] space. Joint labels, intrinsics, pelvis_uv, and pelvis_depth all unaffected. `build_val_transform` unchanged. No model or train.py changes. Full class code provided. All config fields explicitly specified.

**Verdict:** APPROVED — No issues found. Design is complete, mathematically correct, and unambiguous for the Builder.
