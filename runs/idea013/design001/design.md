# Design 001 — Small-Beta Smooth L1 (beta=0.01)

## Starting Point

`runs/idea004/design002/` (best body MPJPE = 112.3 mm, LLRD gamma=0.90, unfreeze_epoch=5).

## Overview

Reduce the Smooth L1 loss beta from the baseline 0.05 m (50 mm) to 0.01 m (10 mm). This makes the loss behave more like L1 for errors in the 10-50 mm range, which are currently treated quadratically. The stronger constant gradient for medium-sized errors may help the model push past plateaus where per-joint errors cluster around 30-50 mm.

## Problem

The baseline Smooth L1 with beta=0.05 uses a quadratic (L2-like) gradient for all errors below 50 mm. Once most joint errors are in the 20-50 mm range, the gradient magnitude scales linearly with error size, giving very weak signal for joints that are "almost right." Reducing beta to 0.01 switches these medium-error joints into the linear gradient regime, providing a constant-magnitude gradient that does not diminish as error decreases toward 10 mm.

## Changes

**Only `train.py` is modified.** Replace the call to `pose_loss()` from `infra.py` with a local loss computation using `F.smooth_l1_loss(pred, target, beta=0.01)`. All other loss calls (depth, uv) remain unchanged, still using the original `pose_loss()` with beta=0.05.

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
l_pose = F.smooth_l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX], beta=0.01)
```

That is the only change. No config.py changes needed since beta is not currently a config field.

## Config Summary

| Parameter | Value | Changed? |
|-----------|-------|----------|
| smooth_l1_beta (pose) | 0.01 | YES (was 0.05) |
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

Standard MPJPE (unweighted) for fair comparison. The loss change affects only training gradients.

## Rationale

- With beta=0.01 (10 mm), the quadratic-to-linear transition happens at 10 mm instead of 50 mm. For typical body joint errors of 30-100 mm, the gradient is a constant sign(error), which is stronger than the proportional gradient from the quadratic regime.
- Risk: the constant gradient near convergence could cause oscillation around the optimum for joints that are already very accurate. However, the 10 mm threshold is small enough that this is unlikely to matter for body joints with ~100 mm average error.
- This is the simplest possible change -- a single numeric constant -- making attribution of any performance difference unambiguous.
