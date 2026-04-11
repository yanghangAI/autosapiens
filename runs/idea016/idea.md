# idea016 -- 2.5D Heatmap Representation with Soft-Argmax Decoding

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea014/design003/code/`

Current SOTA: val_mpjpe_body = **106.85 mm**, val_pelvis = **96.73 mm**, val_mpjpe_weighted = **103.51 mm**. This idea replaces the *output representation* of the head — moving from direct 3D coordinate regression to a 2.5D heatmap representation (per-joint 2D spatial heatmap + per-joint depth) decoded via soft-argmax.

## Concept

Every prior idea in this pipeline (idea001-014) uses the same output protocol: the decoder produces 70 hidden vectors `(B, 70, 256 or 384)` and a final `Linear(hidden->3)` directly regresses `(x, y, z)` for each joint. **No idea has varied the output representation.**

In 2D and 3D human pose estimation literature, *heatmap-based* outputs with soft-argmax decoding (Integral Pose Regression; Sun et al. 2018) consistently outperform direct regression because:
1. Heatmaps provide a dense spatial inductive bias that matches how ViT features are spatially organized.
2. Soft-argmax gives sub-pixel accuracy while remaining fully differentiable.
3. The supervision signal is spatially local, which regularizes the backbone better than a global L1 on coordinates.

The hypothesis: given that Sapiens ViT produces a spatially-structured `(1024, 40, 24)` feature map, a heatmap decoder that predicts per-joint 2D *spatial* heatmaps (on the 40×24 grid) plus per-joint *depth* values should exploit this spatial structure more naturally than direct 3D regression.

## Broader Reflection

### Strong prior results context

- **idea014/design003** (current SOTA 106.85/96.73) — best direct-regression setup. Sets the bar this new representation must beat.
- **idea008/design003** (pelvis 93.7 mm) — shows that spatially-aware positional encoding (continuous depth PE) already helps pelvis localization substantially. A spatial heatmap representation doubles down on the same insight: let the head reason spatially instead of regressing a coordinate.
- **idea010** (multi-scale features) — mixed results (best val_body 111.4 mm) but confirms the backbone has rich spatial content at the 40×24 resolution we'd decode to. Soft-argmax over this grid is natively supported.

### Patterns to avoid

- **idea002** (attention masking) — structural rigidity hurts. The heatmap decoder here still uses the exact decoder of idea014/design003 (wide head, LLRD); only the final output projection changes. No rigid masking is introduced.
- **idea006/design001** (flip) — pelvis catastrophic failure. We keep the same transforms as idea014/design003.
- **idea013/design001-004** (loss reformulation) — showed that aggressive loss changes (bone length, hard-joint reweighting) did not beat the Smooth L1 baseline. The heatmap loss in this idea is a *representation* change, not a philosophical shift in what we penalize — we still minimize MPJPE in mm via the same Smooth L1 applied *after* soft-argmax decoding.

### Why this is novel and promising

- Output representation has never been varied in 14 prior ideas.
- Soft-argmax converts any heatmap into differentiable 2D coordinates with *sub-pixel* accuracy — training is end-to-end in mm units.
- The grid `(40, 24)` is exactly the right resolution to match Sapiens features; no upsampling is required (unlike classic HRNet pipelines).
- The depth axis can be handled in two complementary ways (A3, A4): a full 3D volumetric heatmap (like Integral Pose Regression), or a scalar-per-joint depth head conditioned on the 2D heatmap peak.

## Design Axes

### Category B -- Novel Exploration

All 4 designs are Category B — no prior idea has varied the output representation. Two categories:

**Axis B1: 2D heatmap + scalar depth regression.**
Replace the wide-head output `Linear(384 -> 3)` with two branches sharing the wide head trunk:
- **2D branch:** project decoder output `(B, 70, 384)` through `Linear(384 -> 40*24)` -> reshape to `(B, 70, 40, 24)` -> softmax over spatial dims -> soft-argmax to produce `(u, v)` in grid coordinates -> normalize to `[0,1]` image space -> reproject to root-relative metric `(x, y)`.
- **Depth branch:** `Linear(384 -> 1)` predicts scalar depth per joint in metres (root-relative).
Final 3D joint is `(x, y, z)` concatenation. Smooth L1 (beta=0.05) on body joints against GT is identical to idea014/design003. The `(x,y)` metric conversion uses the same `pelvis_uv` / normalization constants already present in the data pipeline (Designer must locate these — e.g. from `CropPerson` bbox stored in the sample dict).

*Justification:* This is the minimal change: swap coordinate regression for soft-argmax spatial decoding on the `(x, y)` axes while leaving depth as direct regression. Pure integral pose regression on 2D only.

**Axis B2: 2D heatmap + scalar depth regression, higher-resolution upsampled heatmap.**
Same as B1 but the decoder output is reshaped to `(B, 70, 40, 24)`, then bilinearly upsampled to `(B, 70, 80, 48)` before softmax + soft-argmax. Higher resolution improves sub-pixel accuracy without retraining the backbone, at the cost of a 4x softmax. Depth branch unchanged.

*Justification:* Tests whether the grid resolution limits soft-argmax accuracy.

**Axis B3: 3D volumetric heatmap (Integral Pose Regression, Sun 2018).**
Replace the output with a full 3D heatmap: `Linear(384 -> 40*24*16)` -> reshape to `(B, 70, 40, 24, 16)` -> softmax over the 3D volume -> soft-argmax on all three axes. The 16-bin depth axis matches the `num_depth_bins` from continuous depth PE (coherent design choice). Depth bin centres are the sqrt-spaced anchors already defined in `idea008/design003`. Final `(x, y, z)` produced by single soft-argmax over the 3D volume, then converted to metric coordinates via the same (x,y) pipeline as B1 and the depth bin centres for z.

*Justification:* This is the canonical Integral Pose Regression approach but using the depth bin geometry we already validated. Single unified output pathway.

**Axis B4: 2D heatmap + scalar depth regression + heatmap auxiliary MSE supervision.**
Same as B1 but add an *auxiliary* MSE loss on the spatial heatmap against a Gaussian target centred at the ground-truth (u, v) with sigma=2.0 grid cells, weight `lambda_hm=0.1`. This provides dense spatial supervision in addition to the soft-argmax coordinate loss, which often stabilizes early-epoch heatmap learning.

*Justification:* Tests whether dense heatmap supervision helps convergence within 20 epochs (a concern for soft-argmax-only training).

## Expected Designs

The Designer should generate **4** novel designs (count only these new variations; idea014/design003 serves as the baseline and is NOT re-designed):

1. **2D heatmap + scalar depth (grid-resolution, 40x24).**
2. **2D heatmap + scalar depth (upsampled to 80x48).**
3. **Full 3D volumetric heatmap (40x24x16, sqrt depth bins).**
4. **2D heatmap + scalar depth + auxiliary Gaussian MSE supervision.**

## Design Constraints

- All designs start from `runs/idea014/design003/code/` and copy its optimizer, LR schedule, LLRD config (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.
- The continuous sqrt-spaced depth PE, wide head trunk (hidden_dim=384, num_heads=8, num_layers=4), and data transforms MUST NOT be modified. Only the *final output projection and loss computation* change.
- **Metric conversion (critical).** For all designs, the (u,v) soft-argmax output is in normalized image coordinates `[0, 1]`. The Designer MUST convert to the same root-relative metric `(x, y)` used in the training targets. Consult `baseline/transforms.py -> CropPerson` and `infra.py` to find how targets are constructed; reproduce the inverse transform inside the head loss. If this proves infeasible, the Designer may use a simpler proxy: let the head still regress a scalar (x,y) offset per joint but warm-start it from the soft-argmax location (documented fallback).
- **Depth bin centres for B3** must use the same sqrt-spaced anchors already in the depth PE module (reuse, don't re-derive).
- **Soft-argmax implementation:** standard — compute softmax over the spatial dims, then sum over `argmax_coord = sum(softmax * coord_grid)` with precomputed coordinate buffers registered as `register_buffer`. Do NOT recompute the grid each forward.
- **Losses:** keep Smooth L1 (beta=0.05) on body joints only (BODY_IDX). lambda_depth=0.1, lambda_uv=0.2 stay unchanged (the `uv` auxiliary signal still applies to the pelvis_uv head input from the data pipeline). For design 4, add `lambda_hm=0.1 * MSE(heatmap, gaussian_target)` summed over body joints. Do not modify `infra.py`.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8`, `epochs=20` fixed.
- Memory check: the largest output is design 3 at `(B=4, 70, 40, 24, 16) = ~4.3M floats = 17 MB per batch` — well within 11GB. The softmax over this volume is also cheap.
- Output layer dimensions: design 1/2/4 => `Linear(384, 40*24) = 369K params`; design 3 => `Linear(384, 40*24*16) = 5.9M params`. All fit.
- Initialization: zero-init the output heatmap projection bias to avoid degenerate early softmax.
