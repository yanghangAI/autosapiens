# Code Review — idea021 / design003 — Bone-Length Loss on J2 with lambda=0.05

**Design_ID:** idea021/design003
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- Unchanged from baseline (idea015/design004). Correct.

### train.py
- `SMPLX_SKELETON` correctly imported from `infra`.
- `BODY_EDGES = [(a, b) for (a, b) in SMPLX_SKELETON if a < 22 and b < 22]` computed at module level (before epoch loop). Correct.
- `bone_length_loss()` helper function correctly implements the design formula:
  ```python
  pred_len = torch.norm(pred_joints[:, i] - pred_joints[:, j], dim=-1)
  gt_len   = torch.norm(gt_joints[:, i]   - gt_joints[:, j],   dim=-1)
  losses.append(torch.abs(pred_len - gt_len))
  return torch.stack(losses, dim=1).mean()
  ```
- In `train_one_epoch()`:
  - `l_bone` computed on `out["joints"][:, BODY_IDX]` (J2) and `joints[:, BODY_IDX]` with `BODY_EDGES`. Correct — applied to J2 only, not J1.
  - `l_pose = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_bone * l_bone` — matches design spec.
  - `args.lambda_bone` reads from config.

### config.py
- `output_dir` correctly set to `runs/idea021/design003`.
- `lambda_bone = 0.05` — ADDED as specified.
- All other fields match design spec: refine_loss_weight=0.5, head_hidden=384, all training hyperparameters unchanged.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 795.6mm.

## Issues Found
None.
