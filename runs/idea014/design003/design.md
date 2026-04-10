# Design 003 — LLRD + Depth PE + Wide Head + Weight Decay 0.3

## Starting Point

`runs/idea008/design003/`

This design starts from idea008/design003 (val_mpjpe_weighted = 112.0 mm, pelvis = 93.7 mm), same as design002. It applies the full triple combination (LLRD + continuous depth PE + wide head) with increased weight decay (0.3) as a regularization counterbalance.

## Problem

Design002 combines three independently strong improvements, but the wide head adds ~4.4M parameters to the decoder. Combined with unfreezing the full 293M backbone at epoch 5, overfitting risk increases. idea012/design002 showed early promise with weight_decay=0.3 (up from 0.03), suggesting that stronger L2 regularization can help generalization. This design tests whether higher weight decay improves the triple combination.

## Proposed Solution

Identical to design002 (LLRD + depth PE + wide head) with a single change: `weight_decay=0.3` (10x the baseline 0.03) applied to all optimizer param groups.

### Components (All Same as Design002)

1. **Wide head**: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `dim_feedforward=1536`.
2. **LLRD**: `gamma=0.90`, `base_lr_backbone=1e-4`, `unfreeze_epoch=5`. Per-layer LR = `1e-4 * 0.90^(23-i)`.
3. **Continuous depth PE**: sqrt-spaced anchors, 16 depth bins -- unchanged from idea008/design003.
4. **Weight decay**: `0.3` (the only difference from design002).

### LLRD Schedule Detail

Identical to design002:

**Phase 1 (epochs 0-4): Frozen lower backbone**
- Blocks 0-11 + patch/pos embeddings: frozen.
- Blocks 12-23: 12 param groups with LLRD.
- Depth PE params: 1 group, `lr = 1e-4`.
- Head params: 1 group, `lr = 1e-4`.
- Total param groups: 14.

**Phase 2 (epoch 5+): All unfrozen**
- Patch+pos embedding: 1 group, `lr ~ 8.014e-6`.
- Blocks 0-23: 24 groups with LLRD.
- Depth PE params: 1 group, `lr = 1e-4`.
- Head params: 1 group, `lr = 1e-4`.
- Total param groups: 27.

All groups use `weight_decay=0.3`.

### LR for Depth PE Params

Same as design002: `row_emb`, `col_emb`, `depth_emb` get head-level LR (`1e-4`), not subject to LLRD decay.

## LR Schedule

Linear warmup for 3 epochs, cosine decay thereafter.

## Exact File-Level Edit Plan

### `code/config.py`

- `head_hidden = 384` (from 256).
- `lr_backbone = 1e-4` (LLRD base rate).
- Add `gamma = 0.90`.
- Add `unfreeze_epoch = 5`.
- `weight_decay = 0.3` (from 0.03 -- the key difference from design002).
- `output_dir` updated.

### `code/model.py`

No changes.

### `code/train.py`

Same LLRD logic as design002: per-block param groups, freezing, optimizer rebuild at epoch 5. The only difference is that `weight_decay=0.3` is used in all param groups (read from config).

### `code/transforms.py`

No changes.

### `code/infra.py`

No changes.

## `config.py` Fields

Set these fields explicitly:

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea014/design003"
arch            = "sapiens_0.3b"
img_h           = IMG_H
img_w           = IMG_W
head_hidden     = 384      # widened from 256
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
epochs          = 20
lr_backbone     = 1e-4     # LLRD base rate for deepest block
lr_head         = 1e-4
lr_depth_pe     = 1e-4
gamma           = 0.90     # LLRD decay factor
unfreeze_epoch  = 5        # progressive unfreezing epoch
weight_decay    = 0.3      # increased from 0.03 for regularization
warmup_epochs   = 3
num_depth_bins  = 16
grad_clip       = 1.0
lambda_depth    = 0.1
lambda_uv       = 0.2
```

## Builder Implementation Notes

1. Start from `runs/idea008/design003/` using `setup-design`.
2. In `config.py`: set `head_hidden=384`, `lr_backbone=1e-4`, add `gamma=0.90` and `unfreeze_epoch=5`, set `weight_decay=0.3`, update `output_dir`.
3. In `train.py`: port LLRD optimizer logic from `runs/idea004/design002/code/train.py` (identical to design002). Ensure `weight_decay` is read from config and applied to all param groups.
4. In `model.py`: no changes needed.
5. Do not modify the continuous depth PE logic or `infra.py`.
6. The implementation is identical to design002 except for the `weight_decay` value in config.py.

## Parameter Count

Same as design001/design002: ~308M total. Well within VRAM at batch=4.

## Expected Outcome

If the triple combination (design002) shows signs of overfitting (train loss much lower than val loss, or val metrics degrading in later epochs), design003 should outperform it by regularizing the larger parameter space. If overfitting is not an issue, the stronger weight decay may slightly underperform design002 by over-constraining the model. Either way, this design provides a useful diagnostic signal about the regularization needs of the combined architecture.
