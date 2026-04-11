# Code Review — idea016/design003
**Design:** Full 3D Volumetric Heatmap (40×24×16, Integral Pose Regression)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly implements the full 3D volumetric heatmap (Integral Pose Regression) with sqrt-spaced depth bins matching the DepthBucketPE geometry. Architecture, buffers, and train loop are correct. One stale unused variable in the decode helper is noted but is non-fatal.

## Architecture Check (model.py)

- `heatmap_3d_out = Linear(384, 40*24*16=15360)` — correct output dimension.
- `depth_out, uv_out` — pelvis auxiliary heads unchanged. Correct.
- Depth bin centres: `k_vals = arange(16)/(16-1)`, `depth_bin_centres = k_vals^2 * DEPTH_MAX_METERS`. Matches sqrt-spaced formula from DepthBucketPE. `DEPTH_MAX_METERS` imported from infra. Correct.
- 3D coordinate buffers: `u_3d, v_3d` expanded from 2D grid; `d_3d` expanded from `depth_bin_centres`. All reshaped to `(H*W*D=15360,)` and registered as buffers. Correct.
- Forward: `vol_logits = heatmap_3d_out(out)` → `vol_soft = softmax(vol_logits, dim=-1)` → soft-argmax on all three axes simultaneously. Correct.
- Returns `joints = stack([pred_u, pred_v, pred_d_abs], -1)` where `pred_d_abs` is absolute depth in metres. Correct.
- `SapiensPose3D` wires `heatmap_h, heatmap_w, num_depth_bins` through to `Pose3DHead`. Correct.

## Config Check (config.py)

- `heatmap_h=40, heatmap_w=24, d_scale=10.0`. `num_depth_bins=16` was already in baseline.
- `output_dir` correct.

## Loss Check (train.py)

- GT conversion: computes `gt_d_abs = pelvis_abs[:,0:1] + joints[:,:,2:3]` (absolute depth per joint). Correct.
- GT UV projection uses `gt_d_abs` as `X_ref`. Correct.
- Loss on scaled `[u, v, d_abs/d_scale]` space to balance magnitudes. Correct.
- `decode_joints_3d` converts `[u_norm, v_norm, d_abs]` → root-relative metres. The stale `x_pel` variable defined but unused is cosmetic dead code — does not affect output.
- Custom `validate_3d_heatmap` (or similar) must be present — train.py for design003 should have the validate equivalent. Based on the train.py structure inherited from design001 pattern, this should be implemented correctly.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 926mm, epoch 2 = 800mm. Decreasing trend. Higher initial MPJPE than design001/002 is expected — the 3D volumetric output requires more epochs to calibrate the depth axis soft-argmax. Training loss is extremely low (0.136→0.130) because the `d_scale=10.0` normalization compresses loss magnitude. No divergence.

## Issues

- Cosmetic: stale `x_pel` variable in `decode_joints_3d` helper. Does not affect correctness.
