# Code Review — idea019/design003
## Left-Right Symmetry Loss (Axis B1)

**Design_ID:** idea019/design003
**Verdict: APPROVED**

---

## Summary

The implementation correctly adds the left-right symmetry auxiliary loss applied to the refined prediction J2. The SYM_PAIRS definition, loss function, and loss composition all match the design spec exactly. Config has `lambda_sym=0.05`. model.py is unchanged (correct per design). No hardcoded values in train.py.

---

## config.py

All required fields present and correct:
- `lambda_sym = 0.05` — matches design spec exactly
- All baseline hyperparameters preserved: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `epochs=20`, `warmup_epochs=3`, `llrd_gamma=0.90`, `unfreeze_epoch=5`, `weight_decay=0.3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`, `num_depth_bins=16`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `base_lr_backbone=1e-4`
- `lambda_sym` used as `args.lambda_sym` in train.py — not hardcoded

---

## train.py

- `SYM_PAIRS` defined at module level with 6 left-right limb pairs:
  ```python
  SYM_PAIRS = [
      (13, 16, 14, 17),   # upper arm
      (16, 18, 17, 19),   # forearm
      (18, 20, 19, 21),   # hand root
      (1,  4,  2,  5),    # thigh
      (4,  7,  5,  8),    # shin
      (7,  10, 8,  11),   # foot
  ]
  ```
  These match the design spec exactly. All indices are within BODY_IDX range (0-21) and correspond to valid SMPL-X body joint pairs.

- `symmetry_loss(pred_joints, sym_pairs)` function:
  - Per-pair: `left_len` and `right_len` computed as `norm(dim=-1)` of the joint difference vector
  - `(left_len - right_len).abs().mean()` per pair
  - Averaged over all pairs with `max(len(sym_pairs), 1)` guard
  - Applied to `out["joints"]` (J2) only — correct

- Loss composition:
  - `l_sym = symmetry_loss(out["joints"], SYM_PAIRS)` — J2 only, correct
  - `l_pose = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_sym * l_sym` — matches design formula
  - `loss = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps`
  - `l_sym` included in `del` statement at end of batch loop — no memory leak

- LLRD optimizer, freeze/unfreeze, cosine schedule all preserved unchanged

---

## model.py

Design specifies no changes. The file is the baseline Pose3DHead with two-pass shared-decoder and query injection MLP. Confirmed unchanged.

---

## Issues

None. Implementation is clean and correct. All 6 symmetric pairs are within the BODY_IDX range (max index used is 21), so there is no risk of out-of-bounds indexing or index confusion.
