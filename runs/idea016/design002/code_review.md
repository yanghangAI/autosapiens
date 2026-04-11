# Code Review — idea016/design002
**Design:** 2D Heatmap + Scalar Depth (80×48 upsampled grid)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly extends design001 with bilinear upsampling of heatmap logits to 80×48 before softmax+soft-argmax. The coordinate buffers match the upsampled resolution and the rest of the train loop is identical to design001.

## Architecture Check (model.py)

- `heatmap_out = Linear(384, 960)` — predicts at native 40×24. Correct.
- `depth_joint_out = Linear(384, 1)` — unchanged. Correct.
- `up_h=80, up_w=48` computed as `heatmap_h*upsample_factor`. Correct.
- Coordinate buffers: `grid_u, grid_v` at upsampled resolution (80×48 = 3840 bins). Correct — `torch.linspace(0,1,48)` and `torch.linspace(0,1,80)`.
- Forward: `hm_native = heatmap_out(out)` → reshape to `(B,70,40,24)` → `F.interpolate(..., size=(80,48), mode="bilinear", align_corners=False)` reshaping back to `(B,70,3840)` → softmax → soft-argmax with 80×48 buffers. Correct.
- No parameter increase vs design001 (Linear projection unchanged). Correct.

## Config Check (config.py)

- `heatmap_h=40, heatmap_w=24, upsample_factor=2, lambda_z_joint=1.0`. All present.
- `output_dir` correct.
- All inherited HPs unchanged.

## Loss/Train Check (train.py)

- Identical to design001 train.py: same GT UV computation, same `l_xy + lambda_z_joint * l_z` formulation, same `decode_joints_heatmap` helper, same `validate_heatmap`. All correct — the upsampling is internal to the head, transparent to the training loop.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 468mm, epoch 2 = 442mm. Steadily decreasing. Very similar to design001 profile (475mm→562mm for d001 vs 468mm→442mm here). Both are within normal variation for 2-epoch warmup.

## Issues

None identified.
