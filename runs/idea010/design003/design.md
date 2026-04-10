# idea010 / design003 -- Feature Pyramid with 3 Scales

## Starting Point

`runs/idea004/design002/`

## Overview

Extract features from 3 evenly-spaced ViT layers spanning the full backbone depth: layer 7 (early), layer 15 (middle), and layer 23 (final) (0-indexed, out of 24 blocks). Project each from 1024 to 256 channels via separate linear layers, concatenate to get `(B, 768, 40, 24)`, then project to `(B, 1024, 40, 24)` via a final linear. This gives the head a true multi-scale feature pyramid combining low-level spatial detail, mid-level part structure, and high-level semantics.

Note: All ViT layers output the same spatial shape `(B, 960, 1024)` (flattened 40x24 grid), so no spatial resampling is needed.

## Config Changes (`config.py`)

All values below are **unchanged** from the starting point unless explicitly listed:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea010/design003` | Updated for this run |
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
| `multiscale_mode` | `"pyramid3"` | Signals which aggregation strategy to use |
| `multiscale_layers` | `[7, 15, 23]` | 0-indexed ViT block indices (early, mid, final) |

## Architecture Changes

### 1. Backbone (`model.py` -- `SapiensBackboneRGBD`)

Modify `forward()` to extract 3 intermediate outputs:

```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    x = self.vit.patch_embed(x)
    x = x + self.vit.pos_embed

    extract_indices = {7, 15, 23}
    intermediates = []
    for i, layer in enumerate(self.vit.layers):
        x = layer(x)
        if i in extract_indices:
            intermediates.append(x)

    # Apply final LayerNorm to each
    normed = [self.vit.ln1(feat) for feat in intermediates]

    # Reshape each to (B, C, H, W)
    results = []
    for feat in normed:
        B, N, C = feat.shape
        results.append(feat.transpose(1, 2).reshape(B, C, 40, 24))

    return results  # list of 3 tensors, each (B, 1024, 40, 24)
```

### 2. Feature Aggregation Module (`model.py` -- new `FeaturePyramid` module)

```python
class FeaturePyramid(nn.Module):
    """Project each scale to 256 channels, concatenate, project to embed_dim."""
    def __init__(self, embed_dim: int = 1024, proj_dim: int = 256, num_scales: int = 3):
        super().__init__()
        self.scale_projs = nn.ModuleList([
            nn.Linear(embed_dim, proj_dim) for _ in range(num_scales)
        ])
        self.fuse_proj = nn.Linear(proj_dim * num_scales, embed_dim)
        # Xavier init for all projections
        for proj in self.scale_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.fuse_proj.weight)
        nn.init.zeros_(self.fuse_proj.bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # features: list of num_scales x (B, embed_dim, H, W)
        projected = []
        for proj, feat in zip(self.scale_projs, features):
            B, C, H, W = feat.shape
            # (B, C, H, W) -> (B, H*W, C) -> project -> (B, H*W, proj_dim)
            f = proj(feat.flatten(2).transpose(1, 2))
            projected.append(f)

        # Concatenate: (B, H*W, proj_dim * num_scales) = (B, 960, 768)
        cat = torch.cat(projected, dim=-1)

        # Final projection: (B, 960, 768) -> (B, 960, 1024)
        out = self.fuse_proj(cat)

        # Reshape to (B, embed_dim, H, W)
        B = out.shape[0]
        return out.transpose(1, 2).reshape(B, -1, 40, 24)  # (B, 1024, 40, 24)
```

**Mathematical detail**:
- 3 per-scale projections: `Linear(1024, 256)` each = 1024*256 + 256 = 262,400 params each, total 787,200
- Fusion projection: `Linear(768, 1024)` = 768*1024 + 1024 = 787,456
- **Total new parameters: ~1.57M** -- well within memory budget

### 3. Full Model (`model.py` -- `SapiensPose3D`)

```python
class SapiensPose3D(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(...)
        self.aggregator = FeaturePyramid(embed_dim=1024, proj_dim=256, num_scales=3)
        self.head = Pose3DHead(in_channels=1024, ...)  # unchanged

    def forward(self, x):
        features = self.backbone(x)       # list of 3 x (B, 1024, 40, 24)
        feat = self.aggregator(features)  # (B, 1024, 40, 24)
        return self.head(feat)
```

### 4. Optimizer Wiring (`train.py`)

Add aggregator parameters as a separate param group with `lr=1e-4`:

```python
if hasattr(model, 'aggregator'):
    param_groups.append({
        "params": list(model.aggregator.parameters()),
        "lr": lr_head,
        "initial_lr": lr_head,
    })
```

Add in both `_build_optimizer_frozen()` and `_build_optimizer_full()`, after the head param group. Ensure `requires_grad = True`.

### 5. Memory Note

Extracting features from layers 7 and 15 requires storing those activations during the forward pass. Since the backbone already computes all layers sequentially, the only additional memory cost is holding 2 extra `(B, 960, 1024)` tensors (~7.5 MB each at fp32, batch=4). This is negligible.

## What NOT to Change

- `infra.py` -- no changes
- Loss computation -- no changes
- LLRD schedule -- unchanged (gamma=0.90, unfreeze_epoch=5)
- Head architecture -- unchanged (in_channels=1024)
- Pretrained weight loading -- no changes needed
