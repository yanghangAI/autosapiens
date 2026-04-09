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
idea009/design001 reviewed on 2026-04-09. Approved (design + code): 6-layer decoder (Axis A1); param math correct (+2.1M, head ~7.9M); all 16 config fields specified; single-integer change in config.py; train.py passes head_num_layers through correctly; LLRD grouping automatically covers new layers; 2-epoch test passed.
idea009/design002 reviewed on 2026-04-09. Approved (design + code): wide head hidden_dim=384 (Axis A2); divisibility 384/8=48 valid; param math correct (+4.4M, head ~9.88M, total ~303M); all 16 config fields specified; no structural code change needed; head_hidden propagates automatically through Pose3DHead; LLRD grouping auto-covers wider layers; stale docstring in train.py cosmetic only; 2-epoch test passed.
idea009/design003 reviewed on 2026-04-09. Approved: sine-cosine joint query init (Axis B1); PE formula and log-space numerics correct; initialization-only change, no parameter count or VRAM impact; isolated to model.py _init_weights; all 16 config fields specified; Builder instructions unambiguous.
idea009/design003 (code) reviewed on 2026-04-09. Approved: _sinusoidal_init correctly implemented as @staticmethod; formula and log-space numerics match design exactly; torch.no_grad() wraps copy_(); trunc_normal_ preserved for regression heads; all 16 config fields correct; 2-epoch smoke test passed with sanity check [0.0, 1.0, 0.0, 1.0].
idea009/design004 reviewed on 2026-04-09. Approved: per-layer input feature gate (Axis B2); 4 scalar gates via nn.ParameterList in Pose3DHead; sigmoid(4.6)≈0.99 init is honest and acceptable; forward loop over decoder.layers with norm guard is correct; 0-dim scalar broadcasts over (B,S,D) memory correctly; gates auto-include in lr_head group; 4 scalar params negligible; all 16 config fields specified.
idea009/design005 reviewed on 2026-04-09. Approved: output LayerNorm before regression (Axis B3); single nn.LayerNorm(256) after decoder output, before pelvis_token extraction; correct for pre-norm TransformerDecoder with no final norm arg; 512 params auto-included in lr_head; no VRAM impact; two one-line changes to model.py only; all 16 config fields specified.
