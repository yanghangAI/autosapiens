# Design 004 — LLRD (gamma=0.90, unfreeze=5) + Gated Continuous Depth PE

## Starting Point

`runs/idea008/design002/`

## Problem

idea008/design002 introduced a learned scalar residual gate on the depth PE contribution,
achieving val_mpjpe_weighted = 112.1 mm -- nearly matching the best design003 (112.0 mm).
The gate gives the model explicit control over depth PE influence strength. When combined
with LLRD, the interplay between the per-block learning rate decay and the gated depth PE
may be beneficial: the gate can learn to compensate for the reduced adaptation capacity of
shallow backbone layers by increasing or decreasing the depth PE signal accordingly.

## Proposed Solution

Take the architecture from idea008/design002 (row + column + gated continuous interpolated
depth PE with linear anchor spacing and scalar sigmoid gate) and add the LLRD optimization
schedule from idea004/design002 (gamma=0.90, progressive unfreezing at epoch 5). The depth
PE architecture including the learned gate is not modified -- only the optimizer is changed.

## LLRD Schedule

Block index `i` runs from 0 (shallowest) to 23 (deepest).

```
lr_i = base_lr_backbone * gamma^(num_blocks - 1 - i)
```

Where:
- `base_lr_backbone = 1e-4` (applied to the deepest block, block 23)
- `gamma = 0.90`
- `num_blocks = 24`

Key computed values:
- Block 23 (deepest):   `lr_23 = 1e-4 * 0.90^0  = 1.000e-4`
- Block 11 (mid):       `lr_11 = 1e-4 * 0.90^12 ~ 2.824e-5`
- Block 0 (shallowest): `lr_0  = 1e-4 * 0.90^23 ~ 8.904e-6`
- Patch+pos embedding:  `lr_embed = 1e-4 * 0.90^24 ~ 8.014e-6`

Head LR: `1e-4` (unchanged).
Depth PE LR: `1e-4` (unchanged -- `row_emb`, `col_emb`, `depth_emb`, `depth_gate` stay
at high LR).

## Progressive Unfreezing

- **Epochs 0-4 (frozen phase):** Blocks 0-11 (12 shallowest) and patch/pos embeddings
  have `requires_grad=False` and are omitted from optimizer param groups. Only blocks
  12-23, depth PE parameters (including gate), and head are trained.
- **Epoch 5 (unfreeze):** All backbone params set to `requires_grad=True`. Optimizer is
  rebuilt from scratch with all groups. Current LR scale applied immediately.

## Optimizer Param Groups

### Epochs 0-4 (frozen lower half)

| Group | Parameters | LR |
|---|---|---|
| Blocks 12-23 | 12 groups, `lr_i = 1e-4 * 0.90^(23-i)` | varies |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb`, `depth_gate` | `1e-4` |
| `head` | all pose head params | `1e-4` |

Total param groups: 14

### Epochs 5-19 (all unfrozen)

| Group | Parameters | LR |
|---|---|---|
| Patch+pos embed | `vit.patch_embed`, `vit.pos_embed` | `~8.014e-6` |
| Blocks 0-23 | 24 groups, `lr_i = 1e-4 * 0.90^(23-i)` | varies |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb`, `depth_gate` | `1e-4` |
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

| Component | idea008/design002 | Design 004 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Depth anchor spacing | uniform linear | unchanged |
| Depth embedding | continuous interpolation, 16 anchors | unchanged |
| Depth gate | scalar sigmoid gate (zero-init) | unchanged |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | transformer decoder, 4 layers | unchanged |
| Optimizer | flat LR groups | LLRD with per-block LR (gamma=0.90) |
| Unfreezing | none | progressive at epoch 5 |

## Gated Depth PE Recap (from idea008/design002)

The depth PE uses continuous interpolation with uniform (linear) anchor spacing:

```python
depth_pos = depth_patches.clamp(0.0, 1.0) * (num_depth_bins - 1)
```

The interpolated depth term is then gated:

```python
gate = torch.sigmoid(self.depth_gate)  # scalar in (0, 1), init at sigmoid(0)=0.5
depth_pe = gate * depth_pe
```

This gate is NOT modified by this design. It remains a learned scalar initialized at zero
(sigmoid output = 0.5).

## Exact File-Level Edit Plan

### `code/model.py`

- **No changes.** The gated depth PE architecture is kept exactly as in idea008/design002.

### `code/train.py`

- Replace the flat backbone optimizer group with LLRD per-block groups (gamma=0.90).
- Access ViT blocks via `model.backbone.vit.layers` (ModuleList of length 24).
- Patch embedding params: `model.backbone.vit.patch_embed.parameters()`.
- Positional embedding: `model.backbone.vit.pos_embed` (nn.Parameter).
- Implement progressive unfreezing: freeze blocks 0-11 + embeddings at init, unfreeze
  and rebuild optimizer at epoch 5.
- Depth PE params (`row_emb`, `col_emb`, `depth_emb`, `depth_gate`) stay in high-LR
  group (`1e-4`), never frozen.
- After optimizer rebuild, set `initial_lr` for each group and apply `get_lr_scale`.

### `code/config.py`

- Add LLRD-specific fields: `llrd_gamma`, `base_lr_backbone`, `unfreeze_epoch`.
- Update `output_dir`.

### `code/transforms.py`

- No change.

## Builder Implementation Notes

1. Start from `runs/idea008/design002/` using `setup-design`.
2. Do NOT modify `code/model.py` -- keep the gated depth PE architecture untouched.
3. All LLRD changes go in `code/train.py`: per-block param groups, freeze/unfreeze logic,
   optimizer rebuild.
4. The depth PE parameters must be identified by name (including `depth_gate`) and placed
   in their own high-LR group, separate from backbone blocks.
5. Do not modify `infra.py`, transforms, or loss computation.
6. The LLRD implementation pattern is the same as design001; only the starting point
   differs (idea008/design002 instead of idea008/design003) because this design uses
   the gated variant of depth PE.

## `config.py` Fields

Set these fields explicitly:

```python
output_dir       = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea011/design004"
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
