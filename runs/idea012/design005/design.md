# Design 005 — Combined Regularization (Dropout + Weight Decay + Drop Path)

## Starting Point

`runs/idea004/design002/` (best val_mpjpe_body = 112.3 mm)

## Overview

Apply three orthogonal regularization knobs simultaneously: head dropout 0.2, weight decay 0.2, and stochastic depth (drop path) 0.2. This tests whether combining multiple moderate regularizers produces a better result than any single strong regularizer. No R-Drop is included, to isolate the effect of these three simpler knobs.

## Problem

Designs 001-003 each vary a single regularization knob. It is possible that individual knobs are insufficient to close the 29 mm train-val gap but their combination provides enough regularization pressure. This design tests that hypothesis with moderate (not extreme) values for each knob.

## Architecture

Identical to idea004/design002:
- **Backbone:** Sapiens ViT-B (`sapiens_0.3b`), 24 transformer blocks
- **Head:** Transformer decoder, 4 layers, hidden=256, 8 heads
- **LLRD:** gamma=0.90, unfreeze_epoch=5

No architectural changes.

## Config Changes

| Parameter | Baseline (idea004/design002) | This Design |
|-----------|------------------------------|-------------|
| `head_dropout` | 0.1 | **0.2** |
| `weight_decay` | 0.03 | **0.2** |
| `drop_path` | 0.1 | **0.2** |

All other config values unchanged:

| Parameter | Value |
|-----------|-------|
| gamma | 0.90 |
| base_lr_backbone | 1e-4 |
| lr_head | 1e-4 |
| unfreeze_epoch | 5 |
| epochs | 20 |
| warmup_epochs | 3 |
| grad_clip | 1.0 |
| lambda_depth | 0.1 |
| lambda_uv | 0.2 |
| BATCH_SIZE | 4 |
| ACCUM_STEPS | 8 |

## Implementation Notes

1. In `config.py`, change three values:
   - `head_dropout = 0.1` to `head_dropout = 0.2`
   - `weight_decay = 0.03` to `weight_decay = 0.2`
   - `drop_path = 0.1` to `drop_path = 0.2`
2. No changes to `model.py`, `train.py`, or `infra.py`.
3. All three parameters are already read from config by existing code. No new code paths needed.

## Rationale

Each regularizer targets a different aspect of the model:
- **Head dropout 0.2:** Regularizes the decoder head activations.
- **Weight decay 0.2:** Penalizes large parameter magnitudes globally (backbone + head).
- **Drop path 0.2:** Regularizes the backbone by randomly dropping transformer blocks.

The chosen values (all 0.2) are moderate — not as aggressive as design002's weight_decay=0.3 — to avoid over-regularizing when all three are active simultaneously. Weight decay is set to 0.2 (not 0.3) to provide headroom since it compounds with the other two regularizers.
