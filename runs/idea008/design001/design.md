# Design 001 — Continuous Interpolated Depth Positional Encoding

## Starting Point

`runs/idea005/design001/`

## Problem

`runs/idea005/design001` improved performance by adding a row + column + depth-bucket
positional embedding, but its depth term uses a hard bucket lookup. Two nearby depth
values on opposite sides of a bucket boundary receive entirely different embeddings, while
two values inside the same bucket receive exactly the same depth positional signal. That
quantization may throw away useful geometry.

## Proposed Solution

Keep the successful row + column decomposition from `runs/idea005/design001`, but replace
the hard 16-bin depth lookup with **continuous linear interpolation** between neighboring
depth embeddings. The model still learns 16 depth-anchor embeddings, but each patch
receives a weighted mixture of the two nearest anchors instead of a one-hot bucket.

The ViT backbone, decoder head, image size, and overall training recipe remain unchanged.

## Module Change

### `ContinuousDepthPE` in `code/model.py`

This design replaces the depth lookup inside `DepthBucketPE` with an interpolated lookup.
The row and column embeddings are unchanged from `runs/idea005/design001`.

The new interpolation module belongs in `code/model.py`, because the starting point keeps
the depth positional embedding module and the custom backbone forward path there. The
Builder should modify the existing depth-PE module in `code/model.py` or replace it with
`ContinuousDepthPE` in that file; this is not a `train.py` change.

#### Parameters

| Parameter | Shape | Init |
|---|---|---|
| `row_emb` | `(40, 1024)` | copied from pretrained 2D PE row means |
| `col_emb` | `(24, 1024)` | copied from pretrained 2D PE column means |
| `depth_emb` | `(16, 1024)` | zero-initialized |

`vit.pos_embed` is still zeroed and frozen exactly as in `runs/idea005/design001`.

## Continuous Depth Encoding

Let `depth_patches` be the pooled normalized depth map on the `40 x 24` patch grid:

```python
depth_patches = F.avg_pool2d(depth_ch, kernel_size=16, stride=16).squeeze(1)  # (B, 40, 24)
depth_pos = depth_patches.clamp(0.0, 1.0) * (num_depth_bins - 1)              # [0, 15]
```

For each patch:

```python
idx_lo = torch.floor(depth_pos).long()
idx_hi = torch.clamp(idx_lo + 1, max=num_depth_bins - 1)
alpha = (depth_pos - idx_lo.float()).unsqueeze(-1)  # interpolation weight
```

Then interpolate between neighboring depth embeddings:

```python
depth_pe = (
    (1.0 - alpha) * self.depth_emb[idx_lo]
    + alpha * self.depth_emb[idx_hi]
)  # (B, 40, 24, 1024)
```

Flatten to `(B, 960, 1024)` and add to the row/column embeddings exactly where
`runs/idea005/design001` adds its decomposed PE inside the backbone forward path in
`code/model.py`:

```python
pe = row_pe + col_pe + depth_pe_flat
patch_tokens = patch_tokens + pe
```

### Boundary Behavior

- Depth `0.0` uses anchor `0` exactly.
- Depth `1.0` uses anchor `15` exactly because both `idx_lo` and `idx_hi` clamp to `15`.
- Intermediate values linearly mix the two nearest anchors.

## Architecture Summary

| Component | `runs/idea005/design001` | Design 001 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Depth embedding usage | hard 16-bin lookup | continuous interpolation over same 16 anchors |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | baseline transformer decoder head | unchanged |

## Optimizer Groups

Use the same optimizer grouping strategy as `runs/idea005/design001`.

| Group | Parameters | LR | Weight Decay |
|---|---|---|---|
| `backbone` | all ViT params except continuous depth PE module | `1e-5` | `0.03` |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb` | `1e-4` | `0.03` |
| `head` | all pose head params | `1e-4` | `0.03` |

The continuous depth PE parameters must stay in the high-LR new-module group, not the
backbone group.

## Exact File-Level Edit Plan

### `code/model.py`

- Replace the hard depth bucket lookup logic from the starting-point depth-PE module with
  the continuous interpolation math above.
- Keep the row/column embedding definitions, pretrained row/column initialization, and
  zeroed frozen `vit.pos_embed` behavior from `runs/idea005/design001`.
- Keep the custom backbone forward path in `code/model.py`, but update the point where it
  builds the depth positional term so it uses interpolated depth embeddings instead of
  integer bucket indexing.

### `code/train.py`

- No architectural edits are required here.
- Only keep or adjust optimizer parameter grouping if the starting-point training code
  needs an explicit reference to the renamed continuous depth-PE module.
- Training loop, loss computation, and dataloading should otherwise stay unchanged.

## Builder Implementation Notes

1. Start from `runs/idea005/design001/` by running the repository's `setup-design` flow.
2. Make the interpolation-module change in `code/model.py`, not `code/train.py`, because
   the starting point keeps the depth positional encoding module and backbone forward hook
   in `model.py`.
3. Keep the custom backbone structure from `runs/idea005/design001` intact; only change
   the depth positional lookup math from hard indexing to continuous interpolation.
4. If the module name changes in `code/model.py`, update `code/train.py` only as needed
   so the optimizer still places the continuous depth-PE parameters into the `lr_depth_pe`
   group.
5. Preserve the same `40 x 24` token grid and `16` learned depth anchors.
6. Do not add new heavy tensor interactions, attention biases, or pairwise patch terms.
7. Keep `row_emb` and `col_emb` initialization identical to `runs/idea005/design001`.
8. Keep `depth_emb` zero-initialized so the model starts close to the pretrained behavior.
9. `train.py` should otherwise stay as close as possible to the starting point to isolate
   the effect of continuous depth interpolation.

## Files to Modify

| File | Change |
|---|---|
| `code/model.py` | replace hard depth bucket lookup with continuous interpolated lookup and update the custom backbone forward path to use it |
| `code/train.py` | no architectural change; only update optimizer wiring if needed to point the `lr_depth_pe` group at the continuous depth-PE module |
| `code/config.py` | update `output_dir` to `runs/idea008/design001`; keep explicit fields below |
| `code/transforms.py` | no change |

## `config.py` Fields

Set these fields explicitly:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea008/design001"
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
