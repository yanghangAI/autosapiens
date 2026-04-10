# idea010 / design001 -- Last-4-Layer Concatenation + Linear Projection

## Starting Point

`runs/idea004/design002/`

## Overview

Extract the outputs of the last 4 ViT transformer blocks (0-indexed layers 20, 21, 22, 23 out of the 24-block Sapiens 0.3B), concatenate along the channel dimension to form a `(B, 4096, 40, 24)` tensor, then project back to `(B, 1024, 40, 24)` via a single linear layer. This is the simplest multi-scale aggregation baseline: it gives the decoder head access to features from multiple depths of the backbone without changing the head architecture at all.

## Config Changes (`config.py`)

All values below are **unchanged** from the starting point unless explicitly listed:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea010/design001` | Updated for this run |
| `lr_backbone` | `1e-4` | Unchanged |
| `lr_head` | `1e-4` | Unchanged |
| `gamma` | `0.90` | Unchanged |
| `unfreeze_epoch` | `5` | Unchanged |
| `epochs` | `20` | Unchanged |
| `warmup_epochs` | `3` | Unchanged |
| `head_hidden` | `256` | Unchanged |
| `head_num_heads` | `8` | Unchanged |
| `head_num_layers` | `4` | Unchanged |
| `head_dropout` | `0.1` | Unchanged |
| `drop_path` | `0.1` | Unchanged |
| `lambda_depth` | `0.1` | Unchanged |
| `lambda_uv` | `0.2` | Unchanged |

**New config fields:**

| Field | Value | Note |
|---|---|---|
| `multiscale_mode` | `"concat4"` | Signals which aggregation strategy to use |
| `multiscale_layers` | `[20, 21, 22, 23]` | 0-indexed ViT block indices to extract |

## Architecture Changes

### 1. Backbone (`model.py` -- `SapiensBackboneRGBD`)

Replace the simple `forward()` with a custom forward that iterates through `self.vit.layers` manually and saves intermediate outputs:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Run patch embed + pos embed (same as VisionTransformer.forward preamble)
    x = self.vit.patch_embed(x)  # (B, num_patches, embed_dim)
    x = x + self.vit.pos_embed   # add positional embedding

    # Iterate through transformer blocks, saving selected outputs
    extract_indices = {20, 21, 22, 23}  # last 4 blocks
    intermediates = []
    for i, layer in enumerate(self.vit.layers):
        x = layer(x)
        if i in extract_indices:
            intermediates.append(x)

    # Final norm on the last layer output
    x_normed = self.vit.ln1(x)  # final LayerNorm
    # Also norm each intermediate (use same final LN for consistency)
    normed = [self.vit.ln1(feat) for feat in intermediates[:-1]] + [x_normed]

    # Concatenate along channel dim: (B, 960, 1024) * 4 -> (B, 960, 4096)
    cat = torch.cat(normed, dim=-1)  # (B, 960, 4096)

    # Reshape to (B, C, H, W) for the head
    B, N, C = cat.shape
    H, W = 40, 24  # patch grid for 640x384 with patch_size=16
    return cat.transpose(1, 2).reshape(B, C, H, W)  # (B, 4096, 40, 24)
```

**Important**: The VisionTransformer's `forward()` applies `self.ln1` (final LayerNorm) only to the last layer output. For multi-scale, apply `self.ln1` to each extracted intermediate for consistent normalization before concatenation.

### 2. Feature Aggregation Module (`model.py` -- new `MultiScaleConcat` module)

Add a lightweight aggregation module to `SapiensPose3D`:

```python
class MultiScaleConcat(nn.Module):
    """Concatenate multi-layer features and project back to embed_dim."""
    def __init__(self, embed_dim=1024, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(embed_dim * num_layers, embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, embed_dim * num_layers, H, W)
        B, C, H, W = feat.shape
        # Project channel dim: (B, H*W, C) -> (B, H*W, embed_dim) -> (B, embed_dim, H, W)
        out = self.proj(feat.flatten(2).transpose(1, 2))  # (B, H*W, embed_dim)
        return out.transpose(1, 2).reshape(B, -1, H, W)   # (B, embed_dim, H, W)
```

### 3. Full Model (`model.py` -- `SapiensPose3D`)

Insert the aggregation module between backbone and head:

```python
class SapiensPose3D(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(...)
        self.aggregator = MultiScaleConcat(embed_dim=1024, num_layers=4)
        self.head = Pose3DHead(in_channels=1024, ...)  # in_channels stays 1024

    def forward(self, x):
        feat = self.backbone(x)        # (B, 4096, 40, 24)
        feat = self.aggregator(feat)   # (B, 1024, 40, 24)
        return self.head(feat)
```

### 4. Optimizer Wiring (`train.py`)

The `MultiScaleConcat` parameters must be included in the optimizer. Add them as a separate param group with `lr=1e-4` (same as head LR):

In `_build_optimizer_frozen()` and `_build_optimizer_full()`, after the head param group, add:

```python
# Aggregator params (if model has an aggregator)
if hasattr(model, 'aggregator'):
    param_groups.append({
        "params": list(model.aggregator.parameters()),
        "lr": lr_head,
        "initial_lr": lr_head,
    })
```

Also ensure the aggregator params have `requires_grad = True` in both optimizer builders.

Update the LR reporting indices in the main loop to account for the extra param group (the aggregator group is appended after the head group, so the head group index stays the same).

### 5. Parameter Count Estimate

- `nn.Linear(4096, 1024)`: 4096 * 1024 + 1024 = ~4.2M parameters
- Well within the 11GB memory budget at batch_size=4

## What NOT to Change

- `infra.py` -- no changes
- Loss computation -- no changes
- LLRD schedule -- unchanged (gamma=0.90, unfreeze_epoch=5)
- Head architecture -- unchanged (in_channels=1024 since aggregator projects back)
- Pretrained weight loading -- no changes needed (only loads backbone weights)
