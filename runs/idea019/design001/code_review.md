# Code Review — idea019/design001
## Bone-Length Auxiliary Loss on Refinement Output (Axis A1)

**Design_ID:** idea019/design001
**Verdict: APPROVED**

---

## Summary

The implementation faithfully reproduces design001 exactly. The bone-length auxiliary loss is applied only to the refined prediction J2, the formula matches the design, all config parameters are present in config.py and not hardcoded in train.py, and the model.py is unchanged from the baseline (correct per design).

---

## config.py

All required fields are present and correct:
- `lambda_bone = 0.1` — matches design spec exactly
- All baseline hyperparameters preserved: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `epochs=20`, `warmup_epochs=3`, `llrd_gamma=0.90`, `unfreeze_epoch=5`, `weight_decay=0.3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`, `num_depth_bins=16`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `base_lr_backbone=1e-4`
- No config values are hardcoded in train.py; all accessed via `args.lambda_bone`, `args.lambda_depth`, `args.lambda_uv`

---

## train.py

- `BODY_EDGES` is computed at module level as `[(a, b) for (a, b) in SMPLX_SKELETON if a < 22 and b < 22]` — matches design spec exactly
- `bone_length_loss(pred_joints, gt_joints, edges)` function signature and implementation match design exactly:
  - Per-edge `norm(dim=-1)` for predicted and GT bone lengths
  - `.abs().mean()` per edge, averaged over total edges
  - Division by `max(len(edges), 1)` prevents zero-division
- Loss composition in `train_one_epoch`:
  - `l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])` — coarse
  - `l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])` — refined
  - `l_bone = bone_length_loss(out["joints"], joints, BODY_EDGES)` — applied to J2
  - `l_pose = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_bone * l_bone` — matches design
  - Final: `(l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps`
- `l_bone` is included in the `del` statement at end of batch loop (no memory leak)
- SMPLX_SKELETON imported from infra — correct
- LLRD optimizer logic, freeze/unfreeze at epoch 5, cosine warmup schedule all preserved unchanged from baseline

---

## model.py

Design specifies no changes to model.py. The file is the baseline Pose3DHead (two-pass shared-decoder with query injection MLP, no bone-length related changes). Confirmed correct.

---

## Issues

None. Implementation is clean and matches the design spec exactly.
