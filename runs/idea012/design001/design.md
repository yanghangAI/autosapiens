# Design 001 — Head Dropout 0.2

## Starting Point

`runs/idea004/design002/` (best val_mpjpe_body = 112.3 mm)

## Overview

Increase the transformer decoder head dropout from 0.1 to 0.2. This is the simplest single-knob regularization change targeting the decoder head where overfitting is most likely concentrated. All other hyperparameters remain identical to idea004/design002.

## Problem

idea004/design002 exhibits a train-val gap of ~29 mm (train_mpjpe_body=83.7 vs val_mpjpe_body=112.3). The decoder head has 4 transformer layers with dropout=0.1. Increasing dropout to 0.2 forces the head to rely on more robust feature combinations, potentially reducing overfitting.

## Architecture

Identical to idea004/design002:
- **Backbone:** Sapiens ViT-B (`sapiens_0.3b`), 24 transformer blocks
- **Head:** Transformer decoder, 4 layers, hidden=256, 8 heads
- **LLRD:** gamma=0.90, unfreeze_epoch=5

Only the dropout rate in the head changes.

## Config Changes

| Parameter | Baseline (idea004/design002) | This Design |
|-----------|------------------------------|-------------|
| `head_dropout` | 0.1 | **0.2** |

All other config values unchanged:

| Parameter | Value |
|-----------|-------|
| gamma | 0.90 |
| base_lr_backbone | 1e-4 |
| lr_head | 1e-4 |
| unfreeze_epoch | 5 |
| weight_decay | 0.03 |
| drop_path | 0.1 |
| epochs | 20 |
| warmup_epochs | 3 |
| grad_clip | 1.0 |
| lambda_depth | 0.1 |
| lambda_uv | 0.2 |
| BATCH_SIZE | 4 |
| ACCUM_STEPS | 8 |

## Implementation Notes

1. In `config.py`, change `head_dropout = 0.1` to `head_dropout = 0.2`.
2. No changes to `model.py`, `train.py`, or `infra.py`.
3. The dropout rate is passed as a constructor argument to the transformer decoder head; no architectural code change is needed.

## Rationale

Dropout is the most direct regularizer for the decoder head. Doubling it from 0.1 to 0.2 is a conservative increase that should reduce head overfitting without destabilizing training. This isolates the effect of head dropout from other regularization knobs.
