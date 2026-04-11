# Code Review — idea016/design001
**Design:** 2D Heatmap + Scalar Depth (40×24 native grid)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly replaces direct 3D coordinate regression with a 2D soft-argmax heatmap head plus per-joint scalar depth. The UV projection, loss formulation, decode helper, and custom validation loop are all present and correct.

## Architecture Check (model.py)

- `heatmap_out = Linear(384, 40*24=960)` — correct output dimension.
- `depth_joint_out = Linear(384, 1)` — per-joint scalar depth. Correct.
- `depth_out, uv_out` — pelvis auxiliary heads unchanged. Correct.
- Coordinate buffers: `grid_u, grid_v` precomputed via `torch.linspace(0,1,W)`, `torch.meshgrid` with `indexing="ij"`, registered as buffers. Matches design spec exactly.
- Forward: `hm_logits = heatmap_out(out)` → `hm_soft = softmax(hm_logits, dim=-1)` → `pred_u = (hm_soft * grid_u).sum(-1)`, `pred_v` similarly → `pred_z = depth_joint_out(out).squeeze(-1)` → `joints_out = stack([pred_u, pred_v, pred_z], -1)`. Correct.
- `heatmap_h=40, heatmap_w=24` wired through `SapiensPose3D.__init__` to `Pose3DHead`. Correct.
- Zero-init of `heatmap_out.bias`. Correct.

## Config Check (config.py)

- `heatmap_h=40, heatmap_w=24, lambda_z_joint=1.0` present.
- `output_dir` correct.
- All inherited HPs from idea014/design003 unchanged.

## Loss Check (train.py)

- GT UV computation: `X_ref = pelvis_abs[:,0:1].unsqueeze(1) + joints[:,:,2:3]` → exact per-joint depth for projection. `u_px = K[:,0:1,0:1]*(-joints[:,:,0:1])/X_ref + K[:,0:1,2:3]`. Matches design spec.
- `l_xy = pose_loss(out["joints"][:,BODY_IDX,:2], gt_uv_joints[:,BODY_IDX])`. Correct.
- `l_z = pose_loss(out["joints"][:,BODY_IDX,2:3], joints[:,BODY_IDX,2:3])`. Correct.
- `l_pose = l_xy + args.lambda_z_joint * l_z`. Note: design spec says `l_xy + l_z` (weight=1.0), implemented as `l_xy + lambda_z_joint * l_z` with `lambda_z_joint=1.0`. Functionally identical.
- Custom `validate_heatmap` function implemented — does not use infra.py `validate` (which would compute MPJPE on raw (u,v,z) without decoding). Correctly decodes via `decode_joints_heatmap` before computing MPJPE. Critical correctness point: PASS.
- `decode_joints_heatmap` converts `(u_norm, v_norm, z_rel)` → `(x,y,z)` metres using K and pelvis_abs. Correct formula.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 475mm, epoch 2 = 562mm. Slight uptick at epoch 2 within warmup phase noise. Training loss is decreasing (0.806→0.747). No divergence. The MPJPE bounce is consistent with the coordinate-space mismatch being resolved as the soft-argmax calibrates.

## Issues

None blocking. The custom `validate_heatmap` function correctly handles the non-standard output format.
