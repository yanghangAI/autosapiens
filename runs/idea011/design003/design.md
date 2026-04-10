# Design 003 — LLRD (gamma=0.90, unfreeze=10) + Sqrt-Spaced Continuous Depth PE

## Starting Point

`runs/idea008/design003/`

## Problem

idea004/design004 showed that later unfreezing (epoch 10) with gamma=0.95 achieved
competitive 112.8 mm body MPJPE. With continuous depth PE providing richer spatial
information to the head from the start, the head may benefit from even longer exclusive
training before backbone fine-tuning begins. This design tests whether combining the
stronger gamma=0.90 decay with later unfreezing at epoch 10 yields better results than
the epoch-5 unfreezing in design001.

## Proposed Solution

Take the architecture from idea008/design003 (row + column + continuous interpolated
depth PE with sqrt anchor spacing) and add an LLRD optimization schedule with gamma=0.90
and progressive unfreezing delayed to epoch 10. During the longer frozen phase, only the
upper 12 backbone blocks, depth PE parameters, and head are trained; the lower 12 blocks
and embeddings remain frozen for half the total training duration.

## LLRD Schedule

Block index `i` runs from 0 (shallowest) to 23 (deepest).

```
lr_i = base_lr_backbone * gamma^(num_blocks - 1 - i)
```

Where:
- `base_lr_backbone = 1e-4` (applied to the deepest block, block 23)
- `gamma = 0.90`
- `num_blocks = 24`

Key computed values (identical to design001):
- Block 23 (deepest):   `lr_23 = 1e-4 * 0.90^0  = 1.000e-4`
- Block 11 (mid):       `lr_11 = 1e-4 * 0.90^12 ~ 2.824e-5`
- Block 0 (shallowest): `lr_0  = 1e-4 * 0.90^23 ~ 8.904e-6`
- Patch+pos embedding:  `lr_embed = 1e-4 * 0.90^24 ~ 8.014e-6`

Head LR: `1e-4` (unchanged).
Depth PE LR: `1e-4` (unchanged -- `row_emb`, `col_emb`, `depth_emb` stay at high LR).

## Progressive Unfreezing

- **Epochs 0-9 (frozen phase):** Blocks 0-11 (12 shallowest) and patch/pos embeddings
  have `requires_grad=False` and are omitted from optimizer param groups. Only blocks
  12-23, depth PE parameters, and head are trained.
- **Epoch 10 (unfreeze):** All backbone params set to `requires_grad=True`. Optimizer is
  rebuilt from scratch with all groups. Current LR scale applied immediately.

Note: The warmup (3 epochs) completes well before the unfreeze point. After rebuild at
epoch 10, the cosine schedule is already in its decay phase; the newly unfrozen layers
start at a reduced effective LR, which provides gentle adaptation.

## Optimizer Param Groups

### Epochs 0-9 (frozen lower half)

| Group | Parameters | LR |
|---|---|---|
| Blocks 12-23 | 12 groups, `lr_i = 1e-4 * 0.90^(23-i)` | varies |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb` | `1e-4` |
| `head` | all pose head params | `1e-4` |

Total param groups: 14

### Epochs 10-19 (all unfrozen)

| Group | Parameters | LR |
|---|---|---|
| Patch+pos embed | `vit.patch_embed`, `vit.pos_embed` | `~8.014e-6` |
| Blocks 0-23 | 24 groups, `lr_i = 1e-4 * 0.90^(23-i)` | varies |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb` | `1e-4` |
| `head` | all pose head params | `1e-4` |

Total param groups: 27

## LR Schedule

Linear warmup for 3 epochs, cosine decay thereafter:

```python
scale = get_lr_scale(epoch, total_epochs=20, warmup_epochs=3)
for g in optimizer.param_groups:
    g["lr"] = g["initial_lr"] * scale
```

After optimizer rebuild at epoch 10, set `initial_lr` per-group and apply current scale.

## Architecture Summary

| Component | idea008/design003 | Design 003 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Depth anchor spacing | sqrt near-emphasized | unchanged |
| Depth embedding | continuous interpolation, 16 anchors | unchanged |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | transformer decoder, 4 layers | unchanged |
| Optimizer | flat LR groups | LLRD with per-block LR (gamma=0.90) |
| Unfreezing | none | progressive at epoch 10 |

## Exact File-Level Edit Plan

### `code/model.py`

- **No changes.** Depth PE architecture kept exactly as in idea008/design003.

### `code/train.py`

- Replace the flat backbone optimizer group with LLRD per-block groups (gamma=0.90).
- Access ViT blocks via `model.backbone.vit.layers` (ModuleList of length 24).
- Patch embedding params: `model.backbone.vit.patch_embed.parameters()`.
- Positional embedding: `model.backbone.vit.pos_embed` (nn.Parameter).
- Implement progressive unfreezing: freeze blocks 0-11 + embeddings at init, unfreeze
  and rebuild optimizer at epoch 10.
- Depth PE params (`row_emb`, `col_emb`, `depth_emb`) stay in high-LR group (`1e-4`),
  never frozen.
- After optimizer rebuild, set `initial_lr` for each group and apply `get_lr_scale`.

### `code/config.py`

- Add LLRD-specific fields: `llrd_gamma`, `base_lr_backbone`, `unfreeze_epoch`.
- Update `output_dir`.

### `code/transforms.py`

- No change.

## Builder Implementation Notes

1. Start from `runs/idea008/design003/` using `setup-design`.
2. Do NOT modify `code/model.py` -- keep the depth PE architecture untouched.
3. All LLRD changes go in `code/train.py`: per-block param groups, freeze/unfreeze logic,
   optimizer rebuild.
4. The depth PE parameters must be identified by name and placed in their own high-LR
   group, separate from backbone blocks.
5. Do not modify `infra.py`, transforms, or loss computation.
6. This design is identical to design001 except `unfreeze_epoch = 10` instead of `5`.
   The Builder can reuse the same LLRD implementation pattern with only the unfreeze
   epoch changed.

## `config.py` Fields

Set these fields explicitly:

```python
output_dir       = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea011/design003"
arch             = "sapiens_0.3b"
img_h            = IMG_H
img_w            = IMG_W
head_hidden      = 256
head_num_heads   = 8
head_num_layers  = 4
head_dropout     = 0.1
drop_path        = 0.1
epochs           = 20
warmup_epochs    = 3
base_lr_backbone = 1e-4
llrd_gamma       = 0.90
unfreeze_epoch   = 10
lr_head          = 1e-4
lr_depth_pe      = 1e-4
weight_decay     = 0.03
num_depth_bins   = 16
grad_clip        = 1.0
lambda_depth     = 0.1
lambda_uv        = 0.2
```

All other infrastructure values remain identical to the starting point.
