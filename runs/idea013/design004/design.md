# Design 004 — Hard-Joint-Weighted Loss

## Starting Point

`runs/idea004/design002/` (best body MPJPE = 112.3 mm, LLRD gamma=0.90, unfreeze_epoch=5).

## Overview

After epoch 0, compute per-joint mean training L1 error across all body joints (0-21). Derive fixed per-joint weights inversely proportional to accuracy: joints with higher error get more loss weight. Apply these fixed weights element-wise to the per-joint Smooth L1 loss for epochs 1-19. This is a one-shot reweighting that allocates more gradient signal to the hardest joints (typically extremities like wrists and ankles) without the instability of dynamic per-epoch scheduling.

## Problem

The baseline Smooth L1 treats all 22 body joints equally. In practice, proximal joints (pelvis, spine, neck) are predicted with much lower error than distal joints (wrists, ankles, head). Equal weighting means most of the gradient signal is driven by the already-accurate torso joints, while the high-error extremities receive proportionally less training focus. A fixed reweighting computed from epoch 0 statistics addresses this imbalance.

## Changes

**Only `train.py` is modified.** The loss function remains Smooth L1 (beta=0.05), but after epoch 0, per-joint weights are applied element-wise. No config.py or model.py changes.

### Weight computation

After epoch 0 completes:

1. During epoch 0, accumulate per-joint L1 errors: maintain a running sum tensor `joint_err_sum` of shape (22,) and a counter `joint_err_count`.
2. For each batch during epoch 0, compute `per_joint_err = (out["joints"][:, BODY_IDX] - joints[:, BODY_IDX]).abs().mean(dim=(0, 2))` which gives mean absolute error per joint averaged over batch and xyz dimensions. Shape: (22,). Add to `joint_err_sum`, increment `joint_err_count`.
3. At the end of epoch 0, compute `mean_err = joint_err_sum / joint_err_count` (shape (22,)).
4. Compute raw weights: `w = mean_err / mean_err.mean()` (normalize so mean weight = 1.0).
5. Clamp: `w = w.clamp(0.5, 2.0)`.
6. Re-normalize: `w = w * 22.0 / w.sum()` so weights sum to 22 (number of body joints).
7. Store as a non-gradient buffer: `joint_weights = w.detach()` on the training device.

### Weighted loss computation (epochs 1-19)

Replace the pose loss line. During epoch 0:
```python
l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
```

During epochs 1-19:
```python
# per-joint smooth L1: (B, 22, 3) -> per-joint mean over batch and xyz -> (22,)
per_joint_loss = F.smooth_l1_loss(
    out["joints"][:, BODY_IDX], joints[:, BODY_IDX],
    beta=0.05, reduction='none'
).mean(dim=(0, 2))  # (22,)
l_pose = (per_joint_loss * joint_weights).mean()
```

This applies the fixed weights element-wise across joints, then averages to produce a scalar loss.

### Code changes in `train.py`

1. Add at the top:
```python
import torch.nn.functional as F
```

2. Before the epoch loop, initialize accumulators:
```python
joint_err_sum = torch.zeros(22, device=device)
joint_err_count = 0
joint_weights = None  # computed after epoch 0
```

3. Inside `train_one_epoch`, add accumulation logic (only during epoch 0):
   - Pass `epoch` and the accumulator references to `train_one_epoch`, or handle at the call site.
   - During epoch 0, after forward pass (inside the no_grad metric block):
```python
if epoch == 0:
    with torch.no_grad():
        per_joint_err = (out["joints"][:, BODY_IDX].float() - joints[:, BODY_IDX].float()).abs().mean(dim=(0, 2))
        joint_err_sum += per_joint_err
        joint_err_count += 1
```

4. After epoch 0 returns, compute weights:
```python
if epoch == 0:
    mean_err = joint_err_sum / joint_err_count
    w = mean_err / mean_err.mean()
    w = w.clamp(0.5, 2.0)
    w = w * 22.0 / w.sum()
    joint_weights = w.detach()
    print(f"[Joint weights] min={joint_weights.min():.3f} max={joint_weights.max():.3f} "
          f"mean={joint_weights.mean():.3f}")
```

5. For epochs 1-19, pass `joint_weights` into `train_one_epoch` and use the weighted loss:
```python
if joint_weights is not None:
    per_joint_loss = F.smooth_l1_loss(
        out["joints"][:, BODY_IDX], joints[:, BODY_IDX],
        beta=0.05, reduction='none'
    ).mean(dim=(0, 2))
    l_pose = (per_joint_loss * joint_weights).mean()
else:
    l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
```

6. The rest of the loss computation remains unchanged:
```python
loss = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

### Implementation details

- `joint_err_sum` and `joint_err_count` can be passed to `train_one_epoch` as mutable containers (e.g., a dict), or `train_one_epoch` can return them alongside the metrics dict.
- The weight vector is computed once and frozen for the rest of training. No per-epoch recomputation.
- The accumulation uses the no_grad block that already exists for metrics, so it adds zero backward-pass cost.

## Config Summary

| Parameter | Value | Changed? |
|-----------|-------|----------|
| smooth_l1_beta (pose) | 0.05 | no |
| joint_weight_clamp | [0.5, 2.0] | NEW (hardcoded) |
| joint_weight_epoch | 0 | NEW (epoch after which weights are computed) |
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

Unchanged. No new model parameters. The weight computation is a one-shot statistics collection step in the training loop.

## Evaluation

Standard MPJPE (unweighted) for fair comparison. The per-joint weights affect only the training loss, not the evaluation metric.

## Rationale

- Extremity joints (wrists, ankles, feet) typically have 2-3x higher MPJPE than torso joints. Equal weighting means gradient signal is dominated by the high-count, low-error torso joints. Reweighting by error magnitude shifts gradient focus toward the harder joints.
- The [0.5, 2.0] clamp prevents extreme weights: no joint gets less than half the baseline weight, and no joint gets more than double. This avoids destabilizing torso prediction while boosting extremity learning.
- One-shot weighting after epoch 0 is more stable than per-epoch dynamic reweighting (which idea003's curriculum loss weighting showed can be unstable with only 20 epochs).
- The weight normalization ensures the total loss magnitude is unchanged (weights sum to 22, same as the number of joints), so the learning rate does not need adjustment.
- Epoch 0 provides a representative error distribution because the model's relative joint difficulty ordering is established early (proximal joints are always easier than distal ones, even before significant training).
