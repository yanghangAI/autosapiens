# Design 003 — Stochastic Depth 0.2

## Starting Point

`runs/idea004/design002/` (best val_mpjpe_body = 112.3 mm)

## Overview

Increase the stochastic depth (drop path) rate in the ViT backbone from 0.1 to 0.2. This randomly drops entire transformer blocks during training, forcing the network to be robust to missing intermediate representations. All other hyperparameters remain identical.

## Problem

Stochastic depth regularizes the backbone itself rather than just the head. With 24 transformer blocks and drop_path=0.1, the current setup already skips some blocks probabilistically. Increasing to 0.2 doubles the drop probability, which may reduce backbone overfitting — particularly relevant since the backbone has ~90M parameters vs ~2M in the head.

## Architecture

Identical to idea004/design002:
- **Backbone:** Sapiens ViT-B (`sapiens_0.3b`), 24 transformer blocks
- **Head:** Transformer decoder, 4 layers, hidden=256, 8 heads
- **LLRD:** gamma=0.90, unfreeze_epoch=5

The `drop_path_rate` is a constructor argument to the ViT backbone; the linearly-increasing per-block drop rates are computed internally. Block 0 gets drop_rate=0, block 23 gets drop_rate=0.2.

## Config Changes

| Parameter | Baseline (idea004/design002) | This Design |
|-----------|------------------------------|-------------|
| `drop_path` | 0.1 | **0.2** |

All other config values unchanged:

| Parameter | Value |
|-----------|-------|
| gamma | 0.90 |
| base_lr_backbone | 1e-4 |
| lr_head | 1e-4 |
| unfreeze_epoch | 5 |
| head_dropout | 0.1 |
| weight_decay | 0.03 |
| epochs | 20 |
| warmup_epochs | 3 |
| grad_clip | 1.0 |
| lambda_depth | 0.1 |
| lambda_uv | 0.2 |
| BATCH_SIZE | 4 |
| ACCUM_STEPS | 8 |

## Implementation Notes

1. In `config.py`, change `drop_path = 0.1` to `drop_path = 0.2`.
2. The backbone constructor already accepts `drop_path_rate` and distributes it linearly across blocks. No code change needed beyond config.
3. No changes to `model.py`, `train.py`, or `infra.py`.

## Rationale

Stochastic depth (Huang et al., 2016) is a standard ViT regularizer. The Sapiens backbone already implements DropPath; we are simply increasing its rate. At 0.2, the deepest block has a 20% chance of being skipped per forward pass, creating an implicit ensemble effect. This targets backbone overfitting specifically, which is complementary to head dropout.
