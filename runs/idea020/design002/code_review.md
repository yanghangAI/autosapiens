# Code Review — idea020 / design002 — Reduced Coarse Supervision Weight 0.1

**Design_ID:** idea020/design002
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- Unchanged from baseline (idea015/design004). No `.detach()` applied — correct for this design, which only modifies the loss weight.

### train.py
- Loss formula correctly changed to `0.1 * l_pose1 + 1.0 * l_pose2` (line 79), matching the design spec.
- All other training loop elements unchanged.

### config.py
- `output_dir` correctly set to `runs/idea020/design002`.
- `refine_loss_weight = 0.1` — correctly reflects the new coarse pass weight.
- All other config fields match design spec: head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, epochs=20, lr_backbone=1e-4, base_lr_backbone=1e-4, llrd_gamma=0.90, unfreeze_epoch=5, lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 825.9mm.

## Issues Found
None.
