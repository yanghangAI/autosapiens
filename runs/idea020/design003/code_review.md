# Code Review — idea020 / design003 — L1 Loss on Refinement Pass Only

**Design_ID:** idea020/design003
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- Unchanged from baseline (idea015/design004). Correct.

### train.py
- `l_pose1` uses `pose_loss()` (Smooth L1), unchanged from baseline.
- `l_pose2` correctly replaced with `F.l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])` — pure L1 for the refinement pass.
- `l_pose = 0.5 * l_pose1 + 1.0 * l_pose2` — coarse weight unchanged at 0.5.
- `torch.nn.functional as F` is imported at the top of the file.
- All other training loop elements unchanged.

### config.py
- `output_dir` correctly set to `runs/idea020/design003`.
- `refine_loss_weight = 0.5` (coarse pass weight unchanged, correctly reflects the baseline coarse weight).
- All other config fields match design spec: head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, epochs=20, lr_backbone=1e-4, base_lr_backbone=1e-4, llrd_gamma=0.90, unfreeze_epoch=5, lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 787.6mm.

## Issues Found
None.
