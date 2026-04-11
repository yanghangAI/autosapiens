# idea016 Design Review Log

---

## idea016/design001 — 2D Heatmap + Scalar Depth (40×24 native grid)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Replaces Linear(384,3) with heatmap branch (Linear 384→960) + depth branch (Linear 384→1). Soft-argmax on 40×24 grid. GT UV conversion via intrinsics + per-joint depth (exact, not proxy). decode_joints_heatmap helper for MPJPE. Minor scale mismatch (UV vs. metre loss units) is an experimental choice, not fatal. ~369K new params. Complete.

---

## idea016/design002 — 2D Heatmap + Scalar Depth (80×48 upsampled grid)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Extends design001 by upsampling logit map to 80×48 before softmax (4× more bins). Bilinear interpolation in logit space. Coordinate buffers correctly sized for 80×48. No parameter increase. Clean extension. No issues.

---

## idea016/design003 — Full 3D Volumetric Heatmap (40×24×16)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Full 3D integral pose regression. Linear(384,15360), 3D softmax, 3D soft-argmax → (u,v,d_abs). Depth bins reuse sqrt-spaced anchors from depth PE. Loss normalized by DEPTH_MAX_METERS for scale balance. Stale unused `x_pel` variable in decode helper — Builder should delete. No fatal issues. ~5.9M new params.

---

## idea016/design004 — 2D Heatmap + Scalar Depth + Gaussian MSE Supervision
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Extends design001 with auxiliary Gaussian MSE on heatmap soft output. sigma=2.0 grid cells, lambda_hm=0.1, body joints only. Gaussian normalization (sum=1) is consistent with softmax output. make_gaussian_targets helper fully specified. Clean design, no issues.
