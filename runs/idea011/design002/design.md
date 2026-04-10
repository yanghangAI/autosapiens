# Design 002 — LLRD (gamma=0.85, unfreeze=5) + Sqrt-Spaced Continuous Depth PE

## Starting Point

`runs/idea008/design003/`

## Problem

idea004/design003 showed that a more aggressive LLRD decay (gamma=0.85) performed slightly
worse than gamma=0.90 on body MPJPE (113.9 vs 112.3 mm) when applied alone. However, the
continuous depth PE from idea008/design003 provides much richer spatial information to the
head. With stronger positional encoding, the model may tolerate more aggressive suppression
of shallow backbone layers because the depth PE compensates by supplying spatial structure
that would otherwise depend on lower backbone features.

## Proposed Solution

Take the architecture from idea008/design003 (row + column + continuous interpolated
depth PE with sqrt anchor spacing) and add an LLRD optimization schedule with gamma=0.85
and progressive unfreezing at epoch 5. This is identical to design001 except for the
steeper decay factor.

## LLRD Schedule

Block index `i` runs from 0 (shallowest) to 23 (deepest).

```
lr_i = base_lr_backbone * gamma^(num_blocks - 1 - i)
```

Where:
- `base_lr_backbone = 1e-4` (applied to the deepest block, block 23)
- `gamma = 0.85`
- `num_blocks = 24`

Key computed values:
- Block 23 (deepest):   `lr_23 = 1e-4 * 0.85^0  = 1.000e-4`
- Block 11 (mid):       `lr_11 = 1e-4 * 0.85^12 ~ 1.422e-5`
- Block 0 (shallowest): `lr_0  = 1e-4 * 0.85^23 ~ 2.024e-6`
- Patch+pos embedding:  `lr_embed = 1e-4 * 0.85^24 ~ 1.720e-6`

Head LR: `1e-4` (unchanged).
Depth PE LR: `1e-4` (unchanged -- `row_emb`, `col_emb`, `depth_emb` stay at high LR).

## Progressive Unfreezing

- **Epochs 0-4 (frozen phase):** Blocks 0-11 (12 shallowest) and patch/pos embeddings
  have `requires_grad=False` and are omitted from optimizer param groups. Only blocks
  12-23, depth PE parameters, and head are trained.
- **Epoch 5 (unfreeze):** All backbone params set to `requires_grad=True`. Optimizer is
  rebuilt from scratch with all groups. Current LR scale applied immediately.

## Optimizer Param Groups

### Epochs 0-4 (frozen lower half)

| Group | Parameters | LR |
|---|---|---|
| Blocks 12-23 | 12 groups, `lr_i = 1e-4 * 0.85^(23-i)` | varies |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb` | `1e-4` |
| `head` | all pose head params | `1e-4` |

Total param groups: 14

### Epochs 5-19 (all unfrozen)

| Group | Parameters | LR |
|---|---|---|
| Patch+pos embed | `vit.patch_embed`, `vit.pos_embed` | `~1.720e-6` |
| Blocks 0-23 | 24 groups, `lr_i = 1e-4 * 0.85^(23-i)` | varies |
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

After optimizer rebuild at epoch 5, set `initial_lr` per-group and apply current scale.

## Architecture Summary

| Component | idea008/design003 | Design 002 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Depth anchor spacing | sqrt near-emphasized | unchanged |
| Depth embedding | continuous interpolation, 16 anchors | unchanged |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | transformer decoder, 4 layers | unchanged |
| Optimizer | flat LR groups | LLRD with per-block LR (gamma=0.85) |
| Unfreezing | none | progressive at epoch 5 |

## Exact File-Level Edit Plan

### `code/model.py`

- **No changes.** Depth PE architecture kept exactly as in idea008/design003.

### `code/train.py`

- Replace the flat backbone optimizer group with LLRD per-block groups (gamma=0.85).
- Access ViT blocks via `model.backbone.vit.layers` (ModuleList of length 24).
- Patch embedding params: `model.backbone.vit.patch_embed.parameters()`.
- Positional embedding: `model.backbone.vit.pos_embed` (nn.Parameter).
- Implement progressive unfreezing: freeze blocks 0-11 + embeddings at init, unfreeze
  and rebuild optimizer at epoch 5.
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
6. This design is identical to design001 except `llrd_gamma = 0.85` instead of `0.90`.
   The Builder can reuse the same LLRD implementation pattern with only the gamma value
   changed.

## `config.py` Fields

Set these fields explicitly:

```python
output_dir       = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea011/design002"
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
llrd_gamma       = 0.85
unfreeze_epoch   = 5
lr_head          = 1e-4
lr_depth_pe      = 1e-4
weight_decay     = 0.03
num_depth_bins   = 16
grad_clip        = 1.0
lambda_depth     = 0.1
lambda_uv        = 0.2
```

All other infrastructure values remain identical to the starting point.
