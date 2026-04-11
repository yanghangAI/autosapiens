# Code Review — idea016/design004
**Design:** 2D Heatmap + Scalar Depth + Auxiliary Gaussian MSE Supervision
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly adds auxiliary Gaussian MSE heatmap supervision on top of design001. The `make_gaussian_targets` helper, heatmap softmax output exposure, and loss integration are all present and match the design specification.

## Architecture Check (model.py)

- Identical to design001 Pose3DHead EXCEPT forward now returns `heatmap_soft = hm_soft` in the output dict — `(B, 70, H*W)` after softmax. Correct per design.
- `heatmap_out, depth_joint_out, depth_out, uv_out, grid_u, grid_v` buffers — all identical to design001. Correct.

## Config Check (config.py)

- `heatmap_h=40, heatmap_w=24, lambda_hm=0.1, hm_sigma=2.0, lambda_z_joint=1.0`. All present.
- `output_dir` correct.
- All inherited HPs unchanged.

## Loss Check (train.py)

- `make_gaussian_targets(gt_uv_joints, hm_h, hm_w, sigma=args.hm_sigma, device=device)`:
  - Converts GT UV from [0,1] to grid coords: `mu_u = gt_uv[:,0]*(hm_w-1)`, `mu_v = gt_uv[:,1]*(hm_h-1)`. Correct.
  - Gaussian: `exp(-((u-mu_u)^2 + (v-mu_v)^2) / (2*sigma^2))`. Correct.
  - Normalized to sum=1 (`/ (sum + 1e-8)`). Correct — matches softmax output space.
  - Returns shape `(B, 70, H*W)`. Correct.
- `l_hm = F.mse_loss(pred_hm, gt_hm)` applied on `BODY_IDX` joints only. Correct per design.
- Total loss: `(l_pose + lambda_depth*l_dep + lambda_uv*l_uv + lambda_hm*l_hm) / accum_steps`. Correct.
- `decode_joints_heatmap` and `validate_heatmap` inherited from design001. Correct.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 493mm, epoch 2 = 437mm. Steadily decreasing. Very similar to design001 and design002 profiles. Training loss slightly higher epoch 2 (1.013 vs 0.806 for d001) due to the added `lambda_hm` term, but MPJPE is lower than d001 at epoch 2 (437mm vs 562mm) — suggesting the Gaussian auxiliary loss is helping convergence. Healthy profile.

## Issues

None identified.
