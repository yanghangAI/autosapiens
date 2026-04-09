# Design 002 — Interpolated Depth PE with Residual Gate

## Starting Point

`runs/idea005/design001/`

## Problem

`runs/idea005/design001` showed that explicit depth-aware positional structure helps, but
it uses a hard bucket lookup and always injects the full depth positional term with fixed
strength. Even if continuous interpolation reduces quantization error, the model may still
benefit from learning how strongly the depth positional signal should influence the
pretrained row/column positional structure.

## Proposed Solution

Start from the same successful row + column + depth positional decomposition as
`runs/idea005/design001`, replace the hard depth bucket lookup with the same continuous
interpolation scheme used in `idea008/design001`, and add a lightweight **learned residual
gate** on the depth term.

The gate lets the model scale the interpolated depth positional contribution rather than
always adding it at full strength. This keeps the architecture close to the winning
starting point while giving the model a simple way to calibrate depth-PE influence.

The ViT backbone, decoder head, image size, and overall training recipe remain unchanged.

## Module Change

### `GatedContinuousDepthPE` in `code/model.py`

This design modifies the depth positional encoding module in `code/model.py`. The row and
column embeddings remain unchanged from `runs/idea005/design001`, but the depth term is
computed with continuous interpolation and then scaled by a learned residual gate before
being added to the tokens.

#### Parameters

| Parameter | Shape | Init |
|---|---|---|
| `row_emb` | `(40, 1024)` | copied from pretrained 2D PE row means |
| `col_emb` | `(24, 1024)` | copied from pretrained 2D PE column means |
| `depth_emb` | `(16, 1024)` | zero-initialized |
| `depth_gate` | `(1,)` or `(1024,)` | zero-initialized |

Use a single scalar gate for simplicity and low risk:

```python
self.depth_gate = nn.Parameter(torch.zeros(1))
```

Apply it through `torch.sigmoid` so the effective depth scale stays bounded:

```python
gate = torch.sigmoid(self.depth_gate)  # scalar in (0, 1)
```

`vit.pos_embed` is still zeroed and frozen exactly as in `runs/idea005/design001`.

## Continuous Depth Encoding

Compute interpolated depth embeddings exactly as in `idea008/design001`:

```python
depth_patches = F.avg_pool2d(depth_ch, kernel_size=16, stride=16).squeeze(1)  # (B, 40, 24)
depth_pos = depth_patches.clamp(0.0, 1.0) * (num_depth_bins - 1)

idx_lo = torch.floor(depth_pos).long()
idx_hi = torch.clamp(idx_lo + 1, max=num_depth_bins - 1)
alpha = (depth_pos - idx_lo.float()).unsqueeze(-1)

depth_pe = (
    (1.0 - alpha) * self.depth_emb[idx_lo]
    + alpha * self.depth_emb[idx_hi]
)  # (B, 40, 24, 1024)
```

Then gate the interpolated depth term before combining it with row/column embeddings:

```python
gate = torch.sigmoid(self.depth_gate)
depth_pe = gate * depth_pe
depth_pe_flat = depth_pe.reshape(B, -1, embed_dim)

pe = row_pe + col_pe + depth_pe_flat
patch_tokens = patch_tokens + pe
```

### Gate Initialization Rationale

- Zero-initializing `depth_gate` makes `sigmoid(0) = 0.5`, which starts from a moderate
  non-saturated scale.
- `depth_emb` remains zero-initialized, so the model still begins close to the pretrained
  row/column-only behavior.
- The gate can then learn whether the interpolated depth term should stay weak or become
  stronger during fine-tuning.

## Architecture Summary

| Component | `runs/idea005/design001` | Design 002 |
|---|---|---|
| Row embedding | learned `(40, 1024)` | unchanged |
| Column embedding | learned `(24, 1024)` | unchanged |
| Depth embedding usage | hard 16-bin lookup | continuous interpolation |
| Depth-term scale | fixed implicit scale 1.0 | learned sigmoid gate |
| `vit.pos_embed` | zeroed frozen buffer | unchanged |
| Head | baseline transformer decoder head | unchanged |

## Optimizer Groups

Use the same optimizer grouping structure as `runs/idea005/design001`, but include
`depth_gate` with the other new depth-PE parameters.

| Group | Parameters | LR | Weight Decay |
|---|---|---|---|
| `backbone` | all ViT params except gated depth-PE module | `1e-5` | `0.03` |
| `depth_pe` | `row_emb`, `col_emb`, `depth_emb`, `depth_gate` | `1e-4` | `0.03` |
| `head` | all pose head params | `1e-4` | `0.03` |

## Exact File-Level Edit Plan

### `code/model.py`

- Replace the hard depth bucket lookup with continuous interpolation.
- Add the scalar `depth_gate` parameter to the depth-PE module in `code/model.py`.
- Keep pretrained row/column initialization and zeroed frozen `vit.pos_embed` behavior
  identical to `runs/idea005/design001`.
- Update the custom backbone forward path so it multiplies the interpolated depth term by
  `sigmoid(depth_gate)` before adding it to the tokens.

### `code/train.py`

- No architectural edits are required here.
- Only update optimizer grouping if needed so `depth_gate` is included in the
  `lr_depth_pe` parameter group with the other depth-PE parameters.
- Training loop, loss computation, and dataloading should otherwise stay unchanged.

## Builder Implementation Notes

1. Start from `runs/idea005/design001/` using the repository `setup-design` flow.
2. Make the interpolation and gating changes in `code/model.py`, not `code/train.py`.
3. Reuse the same token grid (`40 x 24`) and `16` learned depth anchors as the starting
   point.
4. Keep the gate lightweight: a single scalar parameter is required; do not introduce a
   heavy per-token or pairwise gating mechanism.
5. Keep `row_emb` and `col_emb` initialization identical to `runs/idea005/design001`.
6. Keep `depth_emb` zero-initialized.
7. Ensure `depth_gate` is included in the high-LR `depth_pe` optimizer group.
8. `train.py`, `transforms.py`, and the decoder head should otherwise remain as close as
   possible to the starting point to isolate the effect of the gated depth term.

## Files to Modify

| File | Change |
|---|---|
| `code/model.py` | add continuous interpolated depth PE plus scalar residual gate and update backbone forward path to use it |
| `code/train.py` | no architectural change; only update optimizer wiring if needed to include `depth_gate` in the `lr_depth_pe` group |
| `code/config.py` | update `output_dir` to `runs/idea008/design002`; keep explicit fields below |
| `code/transforms.py` | no change |

## `config.py` Fields

Set these fields explicitly:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea008/design002"
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
