# Design 003 — Interpolated Depth PE with Near-Emphasized Spacing

## Starting Point

`runs/idea005/design001/`

## Problem

`runs/idea005/design001` uses uniformly spaced hard depth buckets. `idea008/design001`
already addresses the hard-boundary problem with continuous interpolation, but it still
places the 16 learned depth anchors at uniformly spaced depth positions. That may be
suboptimal because small depth differences in near-body regions are often more informative
for pose estimation than equally sized differences farther away.

## Proposed Solution

Keep the successful row + column decomposition from `runs/idea005/design001` and the
continuous interpolation strategy from `idea008/design001`, but change the **anchor
spacing** so more effective resolution is allocated to nearer depths.

Use a simple square-root remapping of normalized depth before the interpolation step:

```python
depth_pos = torch.sqrt(depth_patches.clamp(0.0, 1.0)) * (num_depth_bins - 1)
```

Because `sqrt(x)` expands smaller values, this creates denser effective anchor coverage
near depth `0` while compressing farther depths. The rest of the model and training
recipe remain unchanged.

## Module Change

### `ContinuousDepthPE` with Near-Emphasized Spacing in `code/model.py`

This design modifies the depth positional encoding logic in `code/model.py`. The row and
column embeddings stay unchanged from `runs/idea005/design001`, but the mapping from
normalized patch depth to anchor position is changed from linear spacing to a near-
emphasized square-root spacing before interpolation.

#### Parameters

| Parameter | Shape | Init |
|---|---|---|
| `row_emb` | `(40, 1024)` | copied from pretrained 2D PE row means |
| `col_emb` | `(24, 1024)` | copied from pretrained 2D PE column means |
| `depth_emb` | `(16, 1024)` | zero-initialized |

`vit.pos_embed` is still zeroed and frozen exactly as in `runs/idea005/design001`.

## Near-Emphasized Continuous Depth Encoding

Compute the pooled depth map on the `40 x 24` patch grid as usual:

```python
depth_patches = F.avg_pool2d(depth_ch, kernel_size=16, stride=16).squeeze(1)  # (B, 40, 24)
depth_norm = depth_patches.clamp(0.0, 1.0)
```

Then remap depth nonlinearly before interpolation:

```python
depth_pos = torch.sqrt(depth_norm) * (num_depth_bins - 1)  # [0, 15], denser near 0
```

Interpolate between neighboring anchors exactly as in `idea008/design001`:

```python
idx_lo = torch.floor(depth_pos).long()
idx_hi = torch.clamp(idx_lo + 1, max=num_depth_bins - 1)
alpha = (depth_pos - idx_lo.float()).unsqueeze(-1)

depth_pe = (
    (1.0 - alpha) * self.depth_emb[idx_lo]
    + alpha * self.depth_emb[idx_hi]
)  # (B, 40, 24, 1024)
```

Flatten to `(B, 960, 1024)` and add to row/column embeddings in the custom backbone
forward path in `code/model.py`:

```python
pe = row_pe + col_pe + depth_pe_flat
patch_tokens = patch_tokens + pe
```

### Why Square-Root Spacing

- It is cheap and deterministic.
- It preserves the same 16 learned anchors and interpolation machinery as `design001`.
- It allocates more representational resolution to smaller normalized depth values.
- It avoids introducing extra parameters or heavy pairwise computations.

## Architecture Summary

| Component | `runs/idea005/design001` | Design 003 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Depth anchor spacing | uniform linear buckets | near-emphasized square-root spacing |
| Depth embedding usage | hard 16-bin lookup | continuous interpolation over same 16 anchors |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | baseline transformer decoder head | unchanged |

## Optimizer Groups

Use the same optimizer grouping strategy as `runs/idea005/design001`.

| Group | Parameters | LR | Weight Decay |
|---|---|---|---|
| `backbone` | all ViT params except continuous depth-PE module | `1e-5` | `0.03` |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb` | `1e-4` | `0.03` |
| `head` | all pose head params | `1e-4` | `0.03` |

The continuous depth-PE parameters must stay in the high-LR new-module group.

## Exact File-Level Edit Plan

### `code/model.py`

- Replace the hard depth bucket lookup with continuous interpolation.
- Change the depth-to-anchor-position mapping from linear spacing to the square-root
  near-emphasized spacing shown above.
- Keep row/column embedding definitions, pretrained initialization, and the zeroed frozen
  `vit.pos_embed` behavior from `runs/idea005/design001`.
- Keep the custom backbone forward path in `code/model.py`, updating only the depth
  positional computation.

### `code/train.py`

- No architectural edits are required here.
- Only update optimizer grouping if the module name changes and the training code needs an
  explicit reference to the continuous depth-PE module.
- Training loop, loss computation, and dataloading should otherwise stay unchanged.

## Builder Implementation Notes

1. Start from `runs/idea005/design001/` using the repository `setup-design` flow.
2. Make the spacing and interpolation changes in `code/model.py`, not `code/train.py`.
3. Reuse the same `40 x 24` token grid and the same `16` learned depth anchors.
4. Keep the nonlinearity fixed and simple: use `torch.sqrt` on clamped normalized depth.
   Do not introduce additional learned remapping parameters in this design.
5. Keep `row_emb` and `col_emb` initialization identical to `runs/idea005/design001`.
6. Keep `depth_emb` zero-initialized.
7. `train.py`, `transforms.py`, and the decoder head should otherwise remain as close as
   possible to the starting point to isolate the effect of near-emphasized anchor spacing.

## Files to Modify

| File | Change |
|---|---|
| `code/model.py` | replace hard bucket lookup with continuous interpolation using square-root near-emphasized depth spacing |
| `code/train.py` | no architectural change; only update optimizer wiring if needed for the continuous depth-PE module |
| `code/config.py` | update `output_dir` to `runs/idea008/design003`; keep explicit fields below |
| `code/transforms.py` | no change |

## `config.py` Fields

Set these fields explicitly:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea008/design003"
arch        = "sapiens_0.3b"
img_h       = IMG_H
img_w       = IMG_W
head_hidden     = 256
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
epochs       = 20
lr_backbone  = 1e-5
lr_head      = 1e-4
lr_depth_pe  = 1e-4
weight_decay = 0.03
warmup_epochs= 3
num_depth_bins = 16
grad_clip    = 1.0
lambda_depth = 0.1
lambda_uv    = 0.2
```

All other infrastructure values should remain identical to the starting point.
