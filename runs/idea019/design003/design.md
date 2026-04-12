# design003 — Left-Right Symmetry Loss (Axis B1)

## Starting Point

`runs/idea015/design001/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE, two-pass shared-decoder refinement with query injection MLP.

## Problem

The current SOTA (idea015/design001) makes no use of the bilateral symmetry of the human body. Left and right limb segments should have approximately equal lengths within a subject. During refinement, the model has coarse predictions for both sides but no training signal enforcing symmetry. This design adds a symmetry regularization loss that penalizes asymmetric bone-length predictions between corresponding left-right joint pairs, applied only to the refined output J2.

## Proposed Solution

**Left-right symmetry auxiliary loss on limb bone lengths from the refined prediction J2.**

For 6 matching limb segments (upper arm L/R, forearm L/R, upper leg L/R), compute the absolute difference between left and right predicted bone lengths and minimize it. The loss is applied to J2 only, with weight `lambda_sym=0.05`. Zero new model parameters.

### Symmetric Limb Pairs

Using the SMPL-X body joint ordering in `ACTIVE_JOINT_INDICES` (indices 0-21):
- Pelvis=0, L_hip=1, R_hip=2, L_knee=4, R_knee=5, L_ankle=7, R_ankle=8
- L_shoulder=13, R_shoulder=14, L_elbow=16, R_elbow=17, L_wrist=18, R_wrist=19
  (from `_SMPLX_BONES_RAW`: (9,13),(13,16),(16,18),(18,20) left arm; (9,14),(14,17),(17,19),(19,21) right arm)
  (hips: (0,1),(1,4),(4,7) left leg; (0,2),(2,5),(5,8) right leg)

The 6 symmetric segment pairs are:
```python
# (left_joint_a, left_joint_b, right_joint_a, right_joint_b)
SYM_PAIRS = [
    (13, 16, 14, 17),   # upper arm:    L_shoulder-L_elbow  vs R_shoulder-R_elbow
    (16, 18, 17, 19),   # forearm:      L_elbow-L_wrist     vs R_elbow-R_wrist
    (18, 20, 19, 21),   # hand root:    L_wrist-L_hand      vs R_wrist-R_hand
    (1,  4,  2,  5 ),   # thigh:        L_hip-L_knee        vs R_hip-R_knee
    (4,  7,  5,  8 ),   # shin:         L_knee-L_ankle      vs R_knee-R_ankle
    (7,  10, 8,  11),   # foot:         L_ankle-L_foot      vs R_ankle-R_foot
]
```

Note: joint indices 10, 11, 20, 21 (feet and hands in SMPL-X body range) are within `BODY_IDX = slice(0,22)`. These correspond to left_foot=10, right_foot=11, left_hand=20, right_hand=21 in the remapped ordering (from `_SMPLX_BONES_RAW` edges (7,10),(8,11),(18,20),(19,21)).

The Builder should verify these indices by inspecting `_ORIG_TO_NEW` and `ACTIVE_JOINT_INDICES` in `infra.py`. If any index is out of body range (>=22), that pair should be dropped.

### Symmetry Loss (train.py)

```python
# Define at module level (or inside train_one_epoch)
SYM_PAIRS = [
    (13, 16, 14, 17),
    (16, 18, 17, 19),
    (18, 20, 19, 21),
    (1,  4,  2,  5),
    (4,  7,  5,  8),
    (7,  10, 8,  11),
]

def symmetry_loss(pred_joints, sym_pairs):
    """
    pred_joints: (B, 70, 3)
    sym_pairs: list of (la, lb, ra, rb) index tuples
    Returns scalar mean absolute left-right bone-length asymmetry.
    """
    total = 0.0
    for (la, lb, ra, rb) in sym_pairs:
        left_len  = (pred_joints[:, la, :] - pred_joints[:, lb, :]).norm(dim=-1)  # (B,)
        right_len = (pred_joints[:, ra, :] - pred_joints[:, rb, :]).norm(dim=-1)  # (B,)
        total += (left_len - right_len).abs().mean()
    return total / max(len(sym_pairs), 1)
```

### Loss Composition (train.py)

```python
l_pose1   = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2   = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_sym     = symmetry_loss(out["joints"], SYM_PAIRS)
l_pose    = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_sym * l_sym
l_dep     = pose_loss(out["pelvis_depth"], gt_pd)
l_uv      = pose_loss(out["pelvis_uv"],    gt_uv)
loss      = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

No changes to `model.py`.

### config.py Changes

```python
lambda_sym = 0.05   # weight for left-right symmetry auxiliary loss on J2
```

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea015/design001/code/ |
| model.py changes | None |
| New loss | symmetry_loss(J2) with lambda_sym=0.05 |
| Symmetric pairs | 6 left-right limb segment pairs (upper arm, forearm, hand, thigh, shin, foot) |
| Loss formula | 0.5*L(J1) + 1.0*L(J2) + 0.05*sym_loss(J2) + lambda_depth*L_dep + lambda_uv*L_uv |
| Extra params | 0 |
| All other HPs | Identical to idea015/design001 |
