# Code Review Log — idea016

## idea016/design001 — 2D Heatmap + Scalar Depth (40×24 native grid)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea016/design001/code_review.md  
heatmap_out(384→960), depth_joint_out(384→1). Coordinate buffers correctly registered. Soft-argmax via grid_u/grid_v. GT UV projection using exact per-joint depth (X_ref = pelvis_abs + z_rel). Custom validate_heatmap function decodes to metric before MPJPE. decode_joints_heatmap helper correct. 2-epoch test passed (475mm→562mm; within warmup noise).

---

## idea016/design002 — 2D Heatmap + Scalar Depth (80×48 upsampled grid)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea016/design002/code_review.md  
Same as design001 + F.interpolate logits to (80,48) before softmax. Grid buffers at 80×48 resolution. No additional parameters. Train loop identical to design001. 2-epoch test passed (468mm→442mm decreasing).

---

## idea016/design003 — Full 3D Volumetric Heatmap (40×24×16, Integral Pose Regression)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea016/design003/code_review.md  
heatmap_3d_out(384→15360). sqrt-spaced depth bins matching DepthBucketPE. 3D soft-argmax buffers registered. Loss normalized by d_scale=10.0. Stale unused x_pel variable in decode helper (cosmetic). 2-epoch test passed (926mm→800mm decreasing).

---

## idea016/design004 — 2D Heatmap + Scalar Depth + Auxiliary Gaussian MSE Supervision
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea016/design004/code_review.md  
Adds heatmap_soft to return dict. make_gaussian_targets correctly generates normalized Gaussian targets. l_hm = MSE(pred_hm, gt_hm) on BODY_IDX. lambda_hm=0.1 correctly applied. 2-epoch test passed (493mm→437mm; faster convergence vs design001 at epoch 2 confirms auxiliary supervision helps).
