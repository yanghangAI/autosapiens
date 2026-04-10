# Design 001 — Depth PE + Wide Head (No LLRD)

## Starting Point

`runs/idea008/design003/`

This design starts from idea008/design003 (val_mpjpe_weighted = 112.0 mm, pelvis = 93.7 mm), which implements continuous depth PE with sqrt-spaced anchors on top of the row+col+depth decomposition from idea005/design001. The optimizer uses a flat schedule (no LLRD, no progressive unfreezing).

## Problem

idea008/design003 achieves the best pelvis/weighted MPJPE via continuous depth PE, but its decoder head uses `hidden_dim=256`, which compresses the 1024-dim backbone features aggressively. idea009/design002 showed that widening the head to 384 matched the best body MPJPE (112.3 mm) when combined with LLRD. This design tests whether the wide head also improves results when combined with continuous depth PE and a flat (non-LLRD) optimizer, isolating the depth-PE + wide-head interaction.

## Proposed Solution

Replace the standard head (hidden_dim=256) in idea008/design003 with the wide head from idea009/design002 (hidden_dim=384, num_heads=8, num_layers=4). Keep the continuous depth PE with sqrt spacing and the flat optimizer (lr_backbone=1e-5, lr_head=1e-4, lr_depth_pe=1e-4) unchanged.

### Head Changes

1. **`input_proj`**: `Linear(1024 -> 384)` (was 1024 -> 256).
2. **`joint_queries`**: `Embedding(70, 384)` (was 70 x 256).
3. **Decoder d_model**: 384, `dim_feedforward = 384 * 4 = 1536` (was 256, ffn=1024).
4. **Output heads**: `joints_out Linear(384 -> 3)`, `depth_out Linear(384 -> 1)`, `uv_out Linear(384 -> 2)` (in_features updated).
5. **`head_num_heads`**: 8 (unchanged). 384 / 8 = 48 per head -- valid.
6. **`head_num_layers`**: 4 (unchanged).

### What Stays Unchanged

- Continuous depth PE: row_emb (40, 1024), col_emb (24, 1024), depth_emb (16, 1024), sqrt anchor spacing, continuous interpolation.
- `vit.pos_embed` zeroed and frozen.
- Flat optimizer with three groups (backbone, depth_pe, head).
- Loss formulation: Smooth L1 (beta=0.05), lambda_depth=0.1, lambda_uv=0.2.
- All data pipeline, augmentation, and infrastructure settings.

## Parameter Count

Wide head (hidden_dim=384, 4 layers) adds ~4.4M params over the 256 baseline head (~9.88M vs ~5.48M). Combined with depth PE overhead from idea008/design003, total model is ~308M params -- well within 11GB VRAM at batch=4.

## Optimizer Groups

Flat optimizer, same structure as idea008/design003:

| Group | Parameters | LR | Weight Decay |
|---|---|---|---|
| `backbone` | all ViT params except depth-PE module | `1e-5` | `0.03` |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb` | `1e-4` | `0.03` |
| `head` | all pose head params (wider now) | `1e-4` | `0.03` |

No LLRD, no progressive unfreezing.

## LR Schedule

Linear warmup for 3 epochs, cosine decay thereafter (same as idea008/design003).

## Exact File-Level Edit Plan

### `code/config.py`

Change `head_hidden` from 256 to 384 and update `output_dir`.

### `code/model.py`

No structural changes needed. `Pose3DHead` already parameterizes all layer sizes through `hidden_dim`, which reads from `head_hidden`. The wider dimensions propagate automatically.

### `code/train.py`

No changes. The flat optimizer grouping collects all head parameters under a single `lr_head` group regardless of head width.

### `code/transforms.py`

No changes.

### `code/infra.py`

No changes.

## `config.py` Fields

Set these fields explicitly:

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea014/design001"
arch            = "sapiens_0.3b"
img_h           = IMG_H
img_w           = IMG_W
head_hidden     = 384      # widened from 256
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
epochs          = 20
lr_backbone     = 1e-5
lr_head         = 1e-4
lr_depth_pe     = 1e-4
weight_decay    = 0.03
warmup_epochs   = 3
num_depth_bins  = 16
grad_clip       = 1.0
lambda_depth    = 0.1
lambda_uv       = 0.2
```

## Builder Implementation Notes

1. Start from `runs/idea008/design003/` using `setup-design`.
2. The only meaningful change is `head_hidden = 384` in `config.py` and updating `output_dir`.
3. Verify that `Pose3DHead` correctly uses `hidden_dim` from config for `input_proj`, `joint_queries`, decoder `d_model`, and all output linears. No code edits should be needed in model.py.
4. Do not modify the continuous depth PE (sqrt spacing, 16 anchors, interpolation logic).
5. Do not introduce LLRD or progressive unfreezing.
6. Keep `lr_backbone = 1e-5` (not 1e-4) -- this is the flat optimizer rate, not the LLRD base rate.

## Expected Outcome

Expecting val_mpjpe_weighted to improve over idea008/design003's 112.0 mm. The wider head should capture richer joint representations from the depth-enhanced backbone features, particularly improving body joint accuracy while maintaining the pelvis improvements from depth PE.
