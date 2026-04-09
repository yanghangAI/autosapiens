# Design 004 — Hybrid Two-Resolution Depth PE

## Starting Point

`runs/idea005/design001/`

## Problem

`runs/idea005/design001` gives each patch a single local depth-bucket positional signal.
That helps, but the depth term encodes only the patch's own local depth and does not
separately communicate the broader scene-depth context. A patch-level token may benefit
from both:

- a **fine local** depth encoding that distinguishes nearby patch depths, and
- a **coarse global** depth code that summarizes the overall scene-depth regime.

The challenge is to add this extra context without introducing expensive pairwise token
operations or large attention-bias tensors.

## Proposed Solution

Keep the successful row + column positional decomposition from `runs/idea005/design001`,
replace the hard local bucket lookup with the continuous interpolation scheme from
`idea008/design001`, and add a second lightweight **coarse global depth positional term**
shared across all tokens in an image.

Each patch token receives:

```python
row_pe + col_pe + fine_local_depth_pe + coarse_global_depth_pe
```

where:

- `fine_local_depth_pe` is patch-specific and built from interpolated depth anchors on the
  `40 x 24` grid
- `coarse_global_depth_pe` is a single image-level depth code broadcast to all tokens

This creates a two-resolution depth positional signal while staying lightweight and fully
compatible with the 20-epoch proxy budget.

## Module Change

### `HybridDepthPE` in `code/model.py`

This design modifies the depth positional encoding module in `code/model.py`. The row and
column embeddings remain unchanged from `runs/idea005/design001`. The depth term is split
into two parts:

1. a **fine local interpolated** depth PE on the patch grid
2. a **coarse global** depth PE from a small image-level depth code

#### Parameters

| Parameter | Shape | Init |
|---|---|---|
| `row_emb` | `(40, 1024)` | copied from pretrained 2D PE row means |
| `col_emb` | `(24, 1024)` | copied from pretrained 2D PE column means |
| `fine_depth_emb` | `(16, 1024)` | zero-initialized |
| `coarse_depth_emb` | `(4, 1024)` | zero-initialized |

Use `16` fine anchors to match the successful starting point and only `4` coarse anchors
to keep the extra parameter count tiny.

`vit.pos_embed` is still zeroed and frozen exactly as in `runs/idea005/design001`.

## Fine Local Depth Encoding

Compute the local patch-grid depth map exactly as in `idea008/design001`:

```python
depth_patches = F.avg_pool2d(depth_ch, kernel_size=16, stride=16).squeeze(1)  # (B, 40, 24)
depth_pos = depth_patches.clamp(0.0, 1.0) * (num_fine_depth_bins - 1)

idx_lo = torch.floor(depth_pos).long()
idx_hi = torch.clamp(idx_lo + 1, max=num_fine_depth_bins - 1)
alpha = (depth_pos - idx_lo.float()).unsqueeze(-1)

fine_depth_pe = (
    (1.0 - alpha) * self.fine_depth_emb[idx_lo]
    + alpha * self.fine_depth_emb[idx_hi]
)  # (B, 40, 24, 1024)
```

Flatten to `(B, 960, 1024)` as usual.

## Coarse Global Depth Encoding

Compute a single image-level average depth:

```python
global_depth = depth_patches.mean(dim=(1, 2)).clamp(0.0, 1.0)  # (B,)
```

Map it to a small 4-anchor coarse code using the same interpolation pattern:

```python
global_pos = global_depth * (num_coarse_depth_bins - 1)  # [0, 3]
g_lo = torch.floor(global_pos).long()
g_hi = torch.clamp(g_lo + 1, max=num_coarse_depth_bins - 1)
g_alpha = (global_pos - g_lo.float()).unsqueeze(-1)  # (B, 1)

coarse_depth_pe = (
    (1.0 - g_alpha) * self.coarse_depth_emb[g_lo]
    + g_alpha * self.coarse_depth_emb[g_hi]
)  # (B, 1024)
```

Broadcast the coarse code to all tokens:

```python
coarse_depth_pe = coarse_depth_pe.unsqueeze(1).expand(B, H_tok * W_tok, embed_dim)
```

## Full Positional Injection

Inside the custom backbone forward path in `code/model.py`, add both depth terms:

```python
pe = row_pe + col_pe + fine_depth_pe_flat + coarse_depth_pe
patch_tokens = patch_tokens + pe
```

## Architecture Summary

| Component | `runs/idea005/design001` | Design 004 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Local depth encoding | hard 16-bin lookup | continuous interpolation with 16 fine anchors |
| Global depth encoding | none | interpolated broadcast code with 4 coarse anchors |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | baseline transformer decoder head | unchanged |

## Optimizer Groups

Use the same optimizer grouping strategy as `runs/idea005/design001`, with both depth
embedding tables placed in the high-LR new-module group.

| Group | Parameters | LR | Weight Decay |
|---|---|---|---|
| `backbone` | all ViT params except hybrid depth-PE module | `1e-5` | `0.03` |
| `depth_pe` | `row_emb`, `col_emb`, `fine_depth_emb`, `coarse_depth_emb` | `1e-4` | `0.03` |
| `head` | all pose head params | `1e-4` | `0.03` |

## Exact File-Level Edit Plan

### `code/model.py`

- Replace the hard local depth bucket lookup with the continuous interpolated fine local
  depth encoding above.
- Add the `coarse_depth_emb` table and the image-level coarse-depth interpolation path.
- Keep row/column embedding definitions, pretrained initialization, and the zeroed frozen
  `vit.pos_embed` behavior from `runs/idea005/design001`.
- Keep the custom backbone forward path in `code/model.py`, updating only the positional
  encoding construction so it adds both fine and coarse depth terms.

### `code/train.py`

- No architectural edits are required here.
- Only update optimizer grouping if needed so `fine_depth_emb` and `coarse_depth_emb` are
  included in the `lr_depth_pe` parameter group.
- Training loop, loss computation, and dataloading should otherwise stay unchanged.

## Builder Implementation Notes

1. Start from `runs/idea005/design001/` using the repository `setup-design` flow.
2. Make all hybrid depth-PE changes in `code/model.py`, not `code/train.py`.
3. Keep the extra global code lightweight: use only `4` coarse anchors and broadcast the
   resulting embedding to all tokens.
4. Reuse the `40 x 24` patch grid and `16` fine local depth anchors.
5. Keep `row_emb` and `col_emb` initialization identical to `runs/idea005/design001`.
6. Keep `fine_depth_emb` and `coarse_depth_emb` zero-initialized.
7. Do not introduce pairwise token interactions, extra attention layers, or heavy scene
   encoders.
8. `train.py`, `transforms.py`, and the decoder head should otherwise remain as close as
   possible to the starting point to isolate the effect of hybrid two-resolution depth PE.

## Files to Modify

| File | Change |
|---|---|
| `code/model.py` | add hybrid two-resolution depth PE with fine local interpolation and coarse global broadcast depth code |
| `code/train.py` | no architectural change; only update optimizer wiring if needed so both depth embedding tables are in the `lr_depth_pe` group |
| `code/config.py` | update `output_dir` to `runs/idea008/design004`; keep explicit fields below |
| `code/transforms.py` | no change |

## `config.py` Fields

Set these fields explicitly:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea008/design004"
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
num_fine_depth_bins   = 16
num_coarse_depth_bins = 4
grad_clip    = 1.0
lambda_depth = 0.1
lambda_uv    = 0.2
```

All other infrastructure values should remain identical to the starting point.
