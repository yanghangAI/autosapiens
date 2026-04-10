# Design 002 — Weight Decay 0.3

## Starting Point

`runs/idea004/design002/` (best val_mpjpe_body = 112.3 mm)

## Overview

Increase weight decay from 0.03 (idea004/design002 value) to 0.3 for all optimizer parameter groups. This is a 10x increase in L2 regularization strength, testing whether stronger weight penalty reduces the train-val gap. All other hyperparameters remain identical.

## Problem

The 29 mm train-val gap may be partly caused by large parameter magnitudes in both the backbone and head. Weight decay penalizes large weights across all layers, providing a global regularization signal complementary to dropout.

## Architecture

Identical to idea004/design002:
- **Backbone:** Sapiens ViT-B (`sapiens_0.3b`), 24 transformer blocks
- **Head:** Transformer decoder, 4 layers, hidden=256, 8 heads
- **LLRD:** gamma=0.90, unfreeze_epoch=5

No architectural changes.

## Config Changes

| Parameter | Baseline (idea004/design002) | This Design |
|-----------|------------------------------|-------------|
| `weight_decay` | 0.03 | **0.3** |

All other config values unchanged:

| Parameter | Value |
|-----------|-------|
| gamma | 0.90 |
| base_lr_backbone | 1e-4 |
| lr_head | 1e-4 |
| unfreeze_epoch | 5 |
| head_dropout | 0.1 |
| drop_path | 0.1 |
| epochs | 20 |
| warmup_epochs | 3 |
| grad_clip | 1.0 |
| lambda_depth | 0.1 |
| lambda_uv | 0.2 |
| BATCH_SIZE | 4 |
| ACCUM_STEPS | 8 |

## Implementation Notes

1. In `config.py`, change `weight_decay = 0.03` to `weight_decay = 0.3`.
2. The LLRD optimizer builder in `train.py` already reads `weight_decay` from config and applies it to all param groups. No code change needed beyond config.
3. No changes to `model.py` or `infra.py`.

## Rationale

Weight decay 0.3 is aggressive but within the range commonly used for ViT fine-tuning (e.g., MAE uses 0.05, DeiT uses 0.05-0.3). Since idea004/design002 uses a very low 0.03, there is substantial headroom. The 10x increase tests whether the model is significantly over-parameterized relative to the training set size.
