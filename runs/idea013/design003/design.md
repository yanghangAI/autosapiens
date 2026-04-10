# Design 003 — Bone-Length Auxiliary Loss

## Starting Point

`runs/idea004/design002/` (best body MPJPE = 112.3 mm, LLRD gamma=0.90, unfreeze_epoch=5).

## Overview

Add a soft bone-length consistency penalty alongside the standard Smooth L1 pose loss. For each anatomical bone (edge in the body skeleton), compute the L2 distance between connected joints for both prediction and ground truth, then penalize the absolute difference in bone lengths. This encourages anatomically plausible poses without modifying the model architecture.

## Problem

The baseline loss treats each joint independently. Two predictions with the same per-joint MPJPE can have very different structural plausibility: one may predict a skeleton with realistic bone lengths, while another distorts limb proportions. A bone-length penalty provides an explicit structural prior that encourages the model to learn correlated joint positions.

## Changes

**Only `train.py` is modified.** The pose Smooth L1 loss (beta=0.05) is unchanged. An auxiliary bone-length L1 loss is added with weight `lambda_bone=0.1`.

### Body skeleton edges

The body skeleton edges (both endpoints in joints 0-21) are derived from `SMPLX_SKELETON` in `infra.py`. After filtering for body-only edges, the 21 edges are:

```
BODY_BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),    # spine: pelvis->spine1->spine2->spine3->neck->head
    (0, 1), (1, 4), (4, 7), (7, 10),                # left leg: pelvis->l_hip->l_knee->l_ankle->l_foot
    (0, 2), (2, 5), (5, 8), (8, 11),                # right leg: pelvis->r_hip->r_knee->r_ankle->r_foot
    (9, 13), (13, 16), (16, 18), (18, 20),           # left arm: spine3->l_collar->l_shoulder->l_elbow->l_wrist
    (9, 14), (14, 17), (17, 19), (19, 21),           # right arm: spine3->r_collar->r_shoulder->r_elbow->r_wrist
]
```

Note: these are the NEW (remapped) indices, which happen to be the same as the original indices for body joints 0-21 since body joints are the first 22 entries in `ACTIVE_JOINT_INDICES`.

### Bone-length loss computation

```python
def bone_length_loss(pred, target, bone_edges):
    """
    pred:   (B, J, 3) predicted body joint positions
    target: (B, J, 3) ground truth body joint positions
    bone_edges: list of (i, j) tuples, indices into the joint dimension
    """
    loss = 0.0
    for i, j in bone_edges:
        pred_len = (pred[:, i] - pred[:, j]).norm(dim=-1)    # (B,)
        gt_len   = (target[:, i] - target[:, j]).norm(dim=-1) # (B,)
        loss = loss + (pred_len - gt_len).abs().mean()
    return loss / len(bone_edges)
```

### Code changes in `train.py`

1. Import `SMPLX_SKELETON` from infra:
```python
from infra import (..., SMPLX_SKELETON)
```

2. Define the body bone list and loss function at module level (after imports):
```python
BODY_BONES = [(a, b) for a, b in SMPLX_SKELETON if a < 22 and b < 22]

def bone_length_loss(pred, target, bone_edges):
    loss = 0.0
    for i, j in bone_edges:
        pred_len = (pred[:, i] - pred[:, j]).norm(dim=-1)
        gt_len   = (target[:, i] - target[:, j]).norm(dim=-1)
        loss = loss + (pred_len - gt_len).abs().mean()
    return loss / len(bone_edges)
```

3. In `train_one_epoch`, after computing `l_pose`, add the bone loss:
```python
l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_bone = bone_length_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX], BODY_BONES)
l_dep  = pose_loss(out["pelvis_depth"], gt_pd)
l_uv   = pose_loss(out["pelvis_uv"],    gt_uv)
loss   = (l_pose + 0.1 * l_bone + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

4. Add `l_bone` to logging and `del` statement:
```python
iter_logger.log({..., "loss_bone": l_bone.item(), ...})
```
```python
del x, out, l_pose, l_bone, l_dep, l_uv, loss
```

### Config value

`lambda_bone = 0.1` is hardcoded in the loss line (not a config field) to keep the change minimal. If this were to become a config field: `lambda_bone = 0.1`.

## Config Summary

| Parameter | Value | Changed? |
|-----------|-------|----------|
| smooth_l1_beta (pose) | 0.05 | no |
| lambda_bone | 0.1 | NEW |
| gamma | 0.90 | no |
| base_lr_backbone | 1e-4 | no |
| lr_head | 1e-4 | no |
| unfreeze_epoch | 5 | no |
| weight_decay | 0.03 | no |
| epochs | 20 | no |
| warmup_epochs | 3 | no |
| BATCH_SIZE | 4 | no |
| ACCUM_STEPS | 8 | no |
| grad_clip | 1.0 | no |
| lambda_depth | 0.1 | no |
| lambda_uv | 0.2 | no |

## Architecture

Unchanged. No new model parameters. The bone-length loss is computed purely from the existing predicted and ground-truth joint positions.

## Evaluation

Standard MPJPE (unweighted) for fair comparison. The bone-length loss is a training-only auxiliary signal.

## Rationale

- The bone-length penalty acts as a soft structural prior. Even though the per-joint Smooth L1 does not explicitly model inter-joint relationships, the bone-length gradient propagates correlated updates to pairs of connected joints.
- lambda_bone=0.1 is conservative: at convergence, if per-bone length errors average ~5 mm, the bone loss contribution is ~0.5 mm (in Smooth L1 units), while the pose loss is ~0.01-0.05 (in meters). The 0.1 weight keeps the bone loss as a secondary regularizer without dominating the primary pose loss.
- 21 body bones cover the full kinematic chain (spine, both legs, both arms). No hand or face bones are included.
- Computational overhead is negligible: 21 vector subtractions and norms per batch, no extra forward passes.
