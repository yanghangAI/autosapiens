# Code Review — idea015/design001
**Design:** Two-Pass Shared-Decoder Refinement (Query Injection)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation faithfully matches design001 specification. Architecture, loss weights, config fields, and optimizer grouping are all correct.

## Architecture Check (model.py)

- `Pose3DHead` adds `refine_mlp = Sequential(Linear(3,384), GELU, Linear(384,384))` — matches design exactly.
- `joints_out2 = Linear(384,3)` — second output head present and correct.
- Forward pass: `out1 = decoder(queries, memory)` → `J1 = joints_out(out1)` → `R = refine_mlp(J1)` → `queries2 = out1 + R` → `out2 = decoder(queries2, memory)` → `J2 = joints_out2(out2)`. This matches design spec exactly.
- `pelvis_depth` and `pelvis_uv` derive from `out2[:,0,:]` (refined pelvis token). Correct per design.
- Return dict includes `joints=J2`, `joints_coarse=J1`, `pelvis_depth`, `pelvis_uv`. Correct.
- Weight init: trunc_normal_(std=0.02) for all heads including `joints_out2`; refine_mlp layers initialized with trunc_normal_ + zero bias. Correct.

## Config Check (config.py)

- `output_dir` = correct idea015/design001 path.
- `refine_passes = 2`, `refine_loss_weight = 0.5` — informational fields present.
- All inherited HPs match spec: `head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, llrd_gamma=0.90, unfreeze_epoch=5, lr_backbone=1e-4, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, num_depth_bins=16, epochs=20`.
- No experiment-specific values hardcoded in train.py.

## Loss Check (train.py)

- `l_pose1 = pose_loss(out["joints_coarse"][:,BODY_IDX], joints[:,BODY_IDX])` — correct.
- `l_pose2 = pose_loss(out["joints"][:,BODY_IDX], joints[:,BODY_IDX])` — correct.
- `l_pose = 0.5*l_pose1 + 1.0*l_pose2` — matches design weights exactly.
- Pelvis depth and UV losses unchanged. Final loss formula correct.
- `del` cleanup includes `l_pose1, l_pose2`. No memory leaks.
- MPJPE computed on `out["joints"]` (final refined prediction). Correct.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test run completed: val_mpjpe_body epoch 1 = 2006mm, epoch 2 = 1499mm. High values as expected for a 2-epoch warmup run (model is still converging). Decreasing trend confirms training is not diverging.
- Epoch times ~32s. LLRD LR ramp correct (1/3→2/3 of 1e-4).

## Issues

None identified.
