# Code Review — idea019/design004
## Joint-Group Query Initialization in Refinement Pass (Axis B2)

**Design_ID:** idea019/design004
**Verdict: APPROVED**

---

## Summary

The implementation correctly adds learnable group embeddings (zero-initialized) to the joint queries before the second decoder pass. The group_emb Embedding(4, 384) and joint_group_ids buffer are implemented as specified. Config contains all required informational fields. train.py loss is unchanged. One minor index assignment issue (inherited from the design) affects only two non-body joints and has no functional impact on the training objective.

---

## config.py

All required fields present and correct:
- `group_emb_init = 0.0` — informational, matches design
- `num_joint_groups = 4` — informational, matches design
- All baseline hyperparameters preserved: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `epochs=20`, `warmup_epochs=3`, `llrd_gamma=0.90`, `unfreeze_epoch=5`, `weight_decay=0.3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`, `num_depth_bins=16`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `base_lr_backbone=1e-4`

---

## model.py

Architecture changes:

1. Group assignment in `Pose3DHead.__init__`:
   - `_TORSO = [0, 3, 6, 9, 12, 15, 23, 24]` — torso joints including eyes at indices 23, 24
   - `_ARMS = [13, 14, 16, 17, 18, 19, 20, 21]` — arm joints
   - `_LEGS = [1, 2, 4, 5, 7, 8, 10, 11]` — leg joints
   - Assignments use `if j < num_joints:` guard — safe for all indices
   - Indices 22-69 not in `_TORSO` assigned to group 3 (extremities) — correct
   - **Minor issue (inherited from design):** Joint indices 23 and 24 appear in `_TORSO`, but joints 22-69 are overridden in the loop `for j in range(22, num_joints)`. However, the loop condition `if j not in _TORSO` correctly excludes indices 23 and 24 from group 3 reassignment. So 23→group0, 24→group0 from the `_TORSO` loop, and the `range(22, ...)` loop won't overwrite them. Joint 22 (unassigned earlier, default 0) will be assigned group 3. This is correct behavior per the design intent.
   - All body joints (0-21) are correctly assigned: 0,3,6,9,12,15→group0; 13,14,16-21→group1; 1,2,4,5,7,8,10,11→group2; joints 2-9 range details: 2→group2(legs), 3→group0(torso)... verified consistent.

2. `self.register_buffer("joint_group_ids", joint_group_ids)` — non-trainable buffer, shape (70,), correct

3. `self.group_emb = nn.Embedding(4, hidden_dim)` — shape (4, 384) with `head_hidden=384`, correct
   - `nn.init.zeros_(self.group_emb.weight)` — zero-init so training starts identical to baseline, correct

4. In `Pose3DHead.forward`:
   - After `queries2 = out1 + R`:
     ```python
     group_delta = self.group_emb(self.joint_group_ids)   # (70, hidden_dim)
     queries2 = queries2 + group_delta.unsqueeze(0)        # (B, 70, hidden_dim)
     ```
   - `unsqueeze(0)` makes (70, hidden_dim) → (1, 70, hidden_dim), broadcasts over batch — correct
   - Applied before pass 2 only — correct per design
   - `out2 = self.decoder(queries2, memory)` — uses full decoder call (no manual loop), correct for design004

5. Optimizer group: `group_emb` is a submodule of `model.head`, so its parameters are automatically included in the `head_params` group (LR=1e-4, weight_decay=0.3) — correct, no manual changes to train.py needed

---

## train.py

Loss is unchanged from baseline (0.5*L(J1) + 1.0*L(J2)):
```
l_pose = 0.5 * l_pose1 + 1.0 * l_pose2
```
Correct — no new loss terms for design004. LLRD, freeze/unfreeze, cosine schedule all preserved unchanged.

---

## Issues

**Minor (non-fatal, inherited from design):** `_TORSO` includes indices 23 and 24 which are "eyes" in the SMPL-X remapped space. Since these are within the 22-69 non-body range and not part of the training loss (BODY_IDX only covers 0-21), their group assignment (group 0 vs group 3) has no practical impact on training quality. Joint 22 defaults to group 3 (extremities) as intended. This is the same cosmetic issue flagged in the design review and is acceptable.

No other issues. The implementation is correct and matches the design intent precisely.
