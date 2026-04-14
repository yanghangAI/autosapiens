# Code Review — idea020 / design001 — Stop-Gradient on Coarse J1

**Design_ID:** idea020/design001
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- `Pose3DHead.forward()` correctly applies `J1.detach()` before `self.refine_mlp()` on line 217:
  ```python
  R = self.refine_mlp(J1.detach())  # stop gradient through J1
  ```
- This is exactly the one-line change specified in the design.
- All other model components (decoder, refine_decoder, joints_out, joints_out2, depth_out, uv_out) are unchanged from the idea015/design004 baseline.

### train.py
- Loss formula is `0.5 * l_pose1 + 1.0 * l_pose2`, matching the design spec (unchanged from baseline).
- No other changes to the training loop.

### config.py
- `output_dir` correctly set to `runs/idea020/design001`.
- `refine_loss_weight = 0.5` (correct, unchanged).
- All other fields match design spec: head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, epochs=20, lr_backbone=1e-4, base_lr_backbone=1e-4, llrd_gamma=0.90, unfreeze_epoch=5, lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 803.6mm.
- GPU memory within expected bounds (allocated=3.07GB, reserved=7.41GB).

## Issues Found
None.
