# Design 002 — Large-Beta Smooth L1 (beta=0.1)

## Starting Point

`runs/idea004/design002/` (best body MPJPE = 112.3 mm, LLRD gamma=0.90, unfreeze_epoch=5).

## Overview

Increase the Smooth L1 loss beta from 0.05 m (50 mm) to 0.1 m (100 mm). This makes the loss more L2-like for a wider range of errors, giving stronger gradients for large errors (50-100 mm range) which are now in the quadratic regime rather than the linear one. The quadratic regime also naturally down-weights outlier frames where a joint has an unusually large error, reducing gradient variance.

## Problem

With beta=0.05, errors above 50 mm are in the L1 regime with a constant gradient magnitude. Early in training, when many joint errors exceed 100 mm, the loss provides the same gradient magnitude for a 60 mm error as for a 200 mm error. A larger beta=0.1 extends the quadratic regime to 100 mm, making the gradient proportional to error magnitude up to 100 mm. This gives the optimizer stronger signal to fix the worst joints first during early training.

## Changes

**Only `train.py` is modified.** Replace the call to `pose_loss()` from `infra.py` with a local loss computation using `F.smooth_l1_loss(pred, target, beta=0.1)`. All other loss calls (depth, uv) remain unchanged.

### Code changes in `train.py`

1. Add at the top of the file:
```python
import torch.nn.functional as F
```

2. In `train_one_epoch`, replace:
```python
l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
```
with:
```python
l_pose = F.smooth_l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX], beta=0.1)
```

That is the only change.

## Config Summary

| Parameter | Value | Changed? |
|-----------|-------|----------|
| smooth_l1_beta (pose) | 0.1 | YES (was 0.05) |
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

Unchanged from idea004/design002. Sapiens ViT-B backbone + 4-layer transformer decoder head.

## Evaluation

Standard MPJPE (unweighted) for fair comparison.

## Rationale

- With beta=0.1 (100 mm), errors up to 100 mm are in the quadratic regime, giving gradients proportional to error magnitude. This prioritizes large errors during early training.
- The quadratic regime also means the loss for a single outlier joint at 200 mm contributes gradient proportional to its error, while a joint at 50 mm contributes proportionally less. This is a natural form of error-adaptive weighting without explicit per-joint weights.
- Together with design001 (beta=0.01), this characterizes the sensitivity of convergence to the beta hyperparameter along both directions. If both perform similarly to the baseline, beta is not a critical factor; if one clearly wins, it identifies a free performance gain.
- Note: the loss magnitude will differ from baseline (larger beta means the quadratic part is flatter), but AdamW's per-parameter adaptive learning rate largely compensates for loss scale differences.
