# design004 — Joint-Group Query Initialization in Refinement Pass (Axis B2)

## Starting Point

`runs/idea015/design001/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE, two-pass shared-decoder refinement with query injection MLP.

## Problem

The current SOTA (idea015/design001) initializes all 70 joint queries with the same trunc_normal distribution — no structural prior distinguishes torso from limb queries, or arm from leg queries. Before the refinement pass (pass 2), the updated queries `queries2 = out1 + R` carry no explicit signal about which anatomical group each joint belongs to. Adding a learnable group embedding that is added to each query before the refinement pass provides a coarse anatomical grouping prior, encouraging the decoder to specialize differently for torso vs arm vs leg vs extremity tokens.

## Proposed Solution

**Learnable group embeddings added to joint queries before the second decoder pass, zero-initialized so training starts identical to baseline.**

Four anatomical groups:
- Group 0 — Torso: pelvis (0), spine (3, 6, 9, 12), neck (12), head (15), eyes (23, 24)
- Group 1 — Arms: shoulders (13, 14), elbows (16, 17), wrists (18, 19), hands (20, 21)
- Group 2 — Legs: hips (1, 2), knees (4, 5), ankles (7, 8), feet (10, 11)
- Group 3 — Extremities: all remaining joints (22+ in the 70-joint space: hands detail, face landmarks, toes, heels, fingertips)

A learnable embedding table `group_emb: (4, 384)` (zero-initialized) is used. A fixed integer buffer `joint_group_ids: (70,)` maps each joint index to its group (0-3). Before pass 2, `queries2 += group_emb[joint_group_ids]`.

### Architecture Changes (model.py — Pose3DHead)

```python
# In __init__, after existing setup:

# Anatomical group assignments for 70 joints
# Group 0: torso (pelvis + spine + neck + head + eyes)
# Group 1: arms (shoulders, elbows, wrists, hands 0-21 range)
# Group 2: legs (hips, knees, ankles, feet 0-21 range)
# Group 3: extremities (index 22+)
_TORSO = [0, 3, 6, 9, 12, 15, 23, 24]
_ARMS  = [13, 14, 16, 17, 18, 19, 20, 21]
_LEGS  = [1, 2, 4, 5, 7, 8, 10, 11]
# All remaining joints 22-69 (indices 22-24 minus eyes, plus 25-69) → group 3
# Eyes (23, 24) are already in torso group above; indices 22, 25-69 → group 3

joint_group_ids = torch.zeros(num_joints, dtype=torch.long)
for j in _TORSO:
    joint_group_ids[j] = 0
for j in _ARMS:
    joint_group_ids[j] = 1
for j in _LEGS:
    joint_group_ids[j] = 2
# Joints not assigned above default to 0 (torso); then override remaining body joints
# Explicitly set all indices 22-69 not already assigned to group 3
for j in range(22, num_joints):
    if j not in _TORSO:
        joint_group_ids[j] = 3

self.register_buffer("joint_group_ids", joint_group_ids)  # (70,) fixed buffer

# Learnable group embeddings, zero-initialized
self.group_emb = nn.Embedding(4, hidden_dim)
nn.init.zeros_(self.group_emb.weight)
```

```python
# In forward(), after computing queries2 = out1 + R:

# Add group embedding before second decoder pass
group_delta = self.group_emb(self.joint_group_ids)   # (70, hidden_dim)
queries2 = queries2 + group_delta.unsqueeze(0)        # (B, 70, hidden_dim)

# Pass 2: refined prediction (shared decoder weights, unchanged)
out2 = self.decoder(queries2, memory)
J2   = self.joints_out2(out2)
```

### Loss (train.py)

Unchanged from idea015/design001:
```python
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
l_dep   = pose_loss(out["pelvis_depth"], gt_pd)
l_uv    = pose_loss(out["pelvis_uv"],    gt_uv)
loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

### Optimizer Group

`group_emb` is a submodule of `model.head`, so its parameters are automatically included in the `head_params` optimizer group (LR=1e-4, weight_decay=0.3). No changes to train.py optimizer setup.

### config.py Changes

```python
group_emb_init = 0.0    # zero-init so training starts identical to baseline (informational)
num_joint_groups = 4    # torso, arms, legs, extremities (informational)
```

## Memory Estimate

- `group_emb`: 4 × 384 = 1,536 parameters (~6 KB).
- `joint_group_ids` buffer: 70 integers (~280 bytes).
- Total new learnable params: 1,536. No OOM risk.

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea015/design001/code/ |
| New module | group_emb Embedding(4, 384), zero-init; joint_group_ids buffer (70,) |
| Application | Added to queries2 before decoder pass 2 |
| Groups | 0=torso, 1=arms, 2=legs, 3=extremities (index 22+) |
| Init | Zero (training starts identical to baseline) |
| Loss formula | 0.5*L(J1) + 1.0*L(J2) (unchanged) |
| Extra params | 1,536 (~1.5K) |
| All other HPs | Identical to idea015/design001 |
