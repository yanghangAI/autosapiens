# Reviewer Memory

This file serves as the persistent memory storage for the Reviewer agent to store state, notes, and lessons learned across workflow executions. Keep it concise.

---

## Completed Reviews

| Design_ID | Date | Verdict | Notes |
|-----------|------|---------|-------|
| idea006/design001 | 2026-04-08 | APPROVED | Horizontal Flip only; mathematically correct; FLIP_PAIRS matches infra.py; pipeline order correct (flip before SubtractRoot); all config fields specified. |
| idea006/design001 (code) | 2026-04-08 | APPROVED | Implementation matches design exactly; all 16 config fields correct; no hardcoded params in train.py; one inert dead param (scale_jitter) in build_train_transform — harmless. |
| idea006/design002 | 2026-04-08 | APPROVED | Scale/Crop Jitter only; bbox half-extents scaled by s~Uniform(0.8,1.2) around center; 3D metric joints unaffected; CropPerson handles OOB via padding; both bbox formats covered; all config fields specified. |
| idea006/design002 (code) | 2026-04-09 | APPROVED | Implementation matches design exactly; RandomScaleJitter formula correct; pipeline order correct; bbox.copy() prevents aliasing; OOB handled by CropPerson padding; all 17 config fields correct; train.py and model.py unchanged. |
| idea006/design003 | 2026-04-08 | APPROVED | Color Jitter only (RGB); RGBColorJitter wrapper applied after ToTensor; depth and joints unchanged; params match idea.md spec exactly; full class code provided; all config fields specified. |
| idea006/design003 (code) | 2026-04-09 | APPROVED | RGBColorJitter matches design: exact jitter params, RGB-only application after ToTensor, depth untouched, config matches spec, train.py/model.py unchanged; wrapper safely unnormalizes and re-normalizes around torchvision jitter. |
| idea006/design004 | 2026-04-08 | APPROVED | Depth Channel Augmentation only (Gaussian noise σ=0.02 + pixel dropout 10%, each p=0.5); applied after ToTensor on float32 depth in [0,1]; labels and RGB untouched; full class code provided; all config fields specified. |
| idea006/design005 | 2026-04-08 | APPROVED | Combined Geometric (Flip + Scale Jitter); ordering correct (ScaleJitter→CropPerson→Flip→SubtractRoot→ToTensor); both classes identical to APPROVED design001/002; pelvis_uv not manually negated (SubtractRoot handles implicitly); all config fields specified. |
| idea006/design006 | 2026-04-08 | APPROVED | Full Stack (Flip + Color Jitter + Depth Noise); composes APPROVED designs 001, 003, 004; pipeline ordering correct; all params match idea.md Axis 6; Scale Jitter correctly excluded; all config fields specified. |
| idea006/design006 (code) | 2026-04-09 | APPROVED | Builder fixed the rejected value-domain bug in RGBColorJitter by unnormalizing to [0,1], applying ColorJitter, then renormalizing; full stack now matches design, config/train/model remain aligned, and the sanity check passed. |

---

## Lessons Learned

- For horizontal flip designs: verify that pelvis_uv is NOT manually negated when SubtractRoot follows the flip transform — SubtractRoot recomputes it automatically from the negated-Y joints.
- FLIP_PAIRS in infra.py is in remapped 0..69 index space; designs should import directly from infra rather than redefine.
- For RGBColorJitter after ToTensor: if ToTensor normalizes RGB, the implementation must unnormalize before calling torchvision ColorJitter and then renormalize afterward.
idea007/design002 reviewed on 2026-04-09. Approved: strong LLRD on depth-bucket PE, correct freeze/unfreeze and optimizer grouping, config matches spec, sanity check passed.
idea007/design003 reviewed on 2026-04-09. Approved: earlier unfreeze at epoch 3 is implemented correctly; LLRD formulas and optimizer groups match design; config matches spec; sanity check passed.
