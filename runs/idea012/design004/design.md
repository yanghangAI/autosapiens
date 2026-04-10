# Design 004 — R-Drop Consistency Regularization

## Starting Point

`runs/idea004/design002/` (best val_mpjpe_body = 112.3 mm)

## Overview

Add an R-Drop-style consistency regularization loss that penalizes the difference between two stochastic forward passes of the same input. For regression, this is implemented as MSE between two sets of body joint predictions obtained with different dropout/drop-path masks. The total training loss becomes `L_task + alpha * MSE(pred1_body, pred2_body)` with `alpha=1.0`. All other hyperparameters remain identical to idea004/design002.

## Problem

Dropout and stochastic depth introduce stochasticity during training, but the model is not explicitly encouraged to produce consistent outputs across different random masks. R-Drop (Liang et al., 2021) addresses this by adding a consistency term that penalizes prediction variance under dropout noise. This encourages smoother, more generalizable representations.

## Architecture

Identical to idea004/design002:
- **Backbone:** Sapiens ViT-B (`sapiens_0.3b`), 24 transformer blocks
- **Head:** Transformer decoder, 4 layers, hidden=256, 8 heads
- **LLRD:** gamma=0.90, unfreeze_epoch=5

No architectural changes. The only change is in the training loop.

## Config Changes

| Parameter | Baseline (idea004/design002) | This Design |
|-----------|------------------------------|-------------|
| `rdrop_alpha` | N/A (new field) | **1.0** |

All other config values unchanged:

| Parameter | Value |
|-----------|-------|
| gamma | 0.90 |
| base_lr_backbone | 1e-4 |
| lr_head | 1e-4 |
| unfreeze_epoch | 5 |
| head_dropout | 0.1 |
| drop_path | 0.1 |
| weight_decay | 0.03 |
| epochs | 20 |
| warmup_epochs | 3 |
| grad_clip | 1.0 |
| lambda_depth | 0.1 |
| lambda_uv | 0.2 |
| BATCH_SIZE | 4 |
| ACCUM_STEPS | 8 |

## Implementation Details

### New config field

Add to `config.py`:
```python
rdrop_alpha = 1.0
```

### Training loop change in `train.py`

Inside the training step, after the normal forward pass and loss computation:

```python
# Normal forward pass (pass 1) — already computed
pred1 = model(images)  # dict with 'joints_3d', 'depth', 'uv'
loss_task = compute_loss(pred1, targets)  # existing loss

# R-Drop: second forward pass with different dropout masks
with torch.no_grad():
    pred2 = model(images)  # different dropout/drop_path masks

# Consistency loss on body joints only
# BODY_IDX is the index tensor for body joints (excluding pelvis)
pred1_body = pred1['joints_3d'][:, BODY_IDX, :]  # (B, num_body, 3)
pred2_body = pred2['joints_3d'][:, BODY_IDX, :].detach()  # (B, num_body, 3)
consistency = F.mse_loss(pred1_body, pred2_body)

# Total loss
loss = loss_task + cfg.rdrop_alpha * consistency
```

Key implementation requirements:
1. The second forward pass uses `torch.no_grad()` to avoid building a second backward graph. This saves ~50% GPU memory compared to a naive two-pass approach.
2. `pred2_body` is `.detach()`-ed so gradients only flow through `pred1_body`.
3. The consistency loss is computed only on body joint 3D predictions (`joints_3d[:, BODY_IDX, :]`), not on pelvis depth or UV outputs.
4. The model must be in `train()` mode for both passes so that dropout and drop_path are active.
5. The consistency loss is added to the task loss before `loss.backward()` and gradient accumulation.

### Logging

Log `consistency_loss` as an additional metric each step for monitoring.

## Rationale

R-Drop encourages the model to produce stable predictions regardless of which neurons are dropped. With `alpha=1.0`, the consistency penalty is weighted equally with the task loss. Since the second pass is `no_grad`, the wall-time cost is roughly 1.5x per step (forward-only is cheaper than forward+backward), not 2x. The 20-epoch budget is wall-time flexible so this is acceptable.
