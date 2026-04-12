# design001 — Bone-Length Auxiliary Loss on Refinement Output (Axis A1)

## Starting Point

`runs/idea015/design001/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE, two-pass shared-decoder refinement with query injection MLP.

## Problem

The current SOTA (idea015/design001, val_body=107.85 mm) performs two-pass iterative refinement but the loss provides no explicit constraint on anatomical bone-length consistency. The body joints (wrists, ankles) have the largest remaining errors, and these are precisely the joints where implausible limb proportions manifest. Adding a bone-length auxiliary loss on the refined output should regularize the model to produce anatomically consistent skeleton configurations, reducing overfitting to idiosyncratic training poses.

## Proposed Solution

**Bone-length consistency auxiliary loss applied to the refined prediction J2.**

At each training step, compute the mean absolute deviation between predicted bone lengths and ground-truth bone lengths, using the `SMPLX_SKELETON` edges restricted to the `BODY_IDX = slice(0, 22)` joints. The loss is applied only to J2 (not J1), with weight `lambda_bone=0.1`. This adds zero new model parameters — it is purely a training signal.

### Bone-Length Loss (train.py)

```python
from infra import SMPLX_SKELETON, BODY_IDX

# Filter skeleton edges to body joints only (indices 0-21)
BODY_EDGES = [(a, b) for (a, b) in SMPLX_SKELETON if a < 22 and b < 22]

def bone_length_loss(pred_joints, gt_joints, edges):
    """
    pred_joints, gt_joints: (B, 70, 3) — world/camera-space joints
    edges: list of (a, b) index pairs for body skeleton
    Returns scalar mean absolute bone-length deviation.
    """
    total = 0.0
    for (a, b) in edges:
        pred_len = (pred_joints[:, a, :] - pred_joints[:, b, :]).norm(dim=-1)  # (B,)
        gt_len   = (gt_joints[:, a, :]   - gt_joints[:, b, :]  ).norm(dim=-1)  # (B,)
        total += (pred_len - gt_len).abs().mean()
    return total / max(len(edges), 1)
```

### Loss Composition (train.py)

```python
l_pose1   = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2   = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_bone    = bone_length_loss(out["joints"], joints, BODY_EDGES)
l_pose    = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_bone * l_bone
l_dep     = pose_loss(out["pelvis_depth"], gt_pd)
l_uv      = pose_loss(out["pelvis_uv"],    gt_uv)
loss      = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

No changes to `model.py`. Metrics and validation remain identical to baseline.

### config.py Changes

Add the following field (no other config changes):
```python
lambda_bone = 0.1   # weight for bone-length auxiliary loss on J2
```

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea015/design001/code/ |
| model.py changes | None |
| New loss | bone_length_loss(J2) with lambda_bone=0.1 |
| Loss formula | 0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2) + lambda_depth*L_dep + lambda_uv*L_uv |
| Extra params | 0 |
| Skeleton edges | SMPLX_SKELETON restricted to BODY_IDX (a<22 and b<22) |
| All other HPs | Identical to idea015/design001 |
