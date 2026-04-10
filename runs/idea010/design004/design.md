# idea010 / design004 -- Cross-Scale Attention Gate

## Starting Point

`runs/idea004/design002/`

## Overview

Extract features from two ViT layers: layer 11 (mid-level, roughly halfway through the 24-block backbone) and layer 23 (final). Compute a spatial attention gate from the mid-layer features using a single linear projection to a scalar, apply sigmoid, and use it to modulate the final-layer features via a residual multiplicative gate: `output = layer_23 * (1 + gate)`. The gate learns to up-weight spatial locations where mid-level features indicate important local structure (e.g., limb boundaries, joint neighborhoods).

The key design choice is **zero-initialization of the gate bias** so that at initialization `gate = sigmoid(Linear(layer_11)) ~ sigmoid(small_values) ~ 0.5`. To make the initial behavior closer to the baseline, we use a bias initialization of `-5.0` so that `sigmoid(-5) ~ 0.0067 ~ 0`, giving `output ~ layer_23 * (1 + 0) = layer_23` -- essentially identical to the baseline at the start of training.

## Config Changes (`config.py`)

All values below are **unchanged** from the starting point unless explicitly listed:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea010/design004` | Updated for this run |
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
| `multiscale_mode` | `"cross_gate"` | Signals which aggregation strategy to use |
| `multiscale_layers` | `[11, 23]` | 0-indexed: mid-layer (gate source) and final layer |
| `gate_bias_init` | `-5.0` | Initial bias for gate linear, so sigmoid ~ 0 at start |

## Architecture Changes

### 1. Backbone (`model.py` -- `SapiensBackboneRGBD`)

Modify `forward()` to return 2 intermediate outputs:

```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    x = self.vit.patch_embed(x)
    x = x + self.vit.pos_embed

    extract_indices = {11, 23}
    intermediates = {}
    for i, layer in enumerate(self.vit.layers):
        x = layer(x)
        if i in extract_indices:
            intermediates[i] = x

    # Apply final LayerNorm to both
    mid_feat = self.vit.ln1(intermediates[11])
    final_feat = self.vit.ln1(intermediates[23])

    # Reshape to (B, C, H, W)
    def reshape(feat):
        B, N, C = feat.shape
        return feat.transpose(1, 2).reshape(B, C, 40, 24)

    return [reshape(mid_feat), reshape(final_feat)]
    # returns [layer_11: (B, 1024, 40, 24), layer_23: (B, 1024, 40, 24)]
```

### 2. Feature Aggregation Module (`model.py` -- new `CrossScaleGate` module)

```python
class CrossScaleGate(nn.Module):
    """Spatial attention gate: mid-layer features modulate final-layer features.

    gate = sigmoid(Linear(mid_features))   -- shape (B, 1, H, W)
    output = final_features * (1 + gate)   -- residual gating
    """
    def __init__(self, embed_dim: int = 1024, bias_init: float = -5.0):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, 1)
        # Zero-init weight, negative bias so sigmoid(output) ~ 0 at init
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        mid_feat, final_feat = features  # each (B, C, H, W)
        B, C, H, W = mid_feat.shape

        # Compute gate from mid-layer features
        # (B, C, H, W) -> (B, H*W, C) -> Linear -> (B, H*W, 1)
        gate = self.gate_proj(mid_feat.flatten(2).transpose(1, 2))  # (B, H*W, 1)
        gate = torch.sigmoid(gate)  # (B, H*W, 1)

        # Reshape gate to (B, 1, H, W) for broadcasting
        gate = gate.transpose(1, 2).reshape(B, 1, H, W)  # (B, 1, H, W)

        # Residual multiplicative gating
        output = final_feat * (1.0 + gate)  # (B, C, H, W)
        return output
```

**Mathematical detail**:
- At initialization: `gate_proj.weight = 0`, `gate_proj.bias = -5.0`
- So `gate_proj(mid_feat) = -5.0` for all spatial positions
- `sigmoid(-5.0) = 0.0067 ~ 0`
- `output = final_feat * (1 + 0.0067) ~ final_feat` -- nearly identical to baseline
- During training, the gate learns spatial attention from mid-level features
- **Total new parameters**: 1024 * 1 + 1 = **1,025 parameters** (negligible)

### 3. Full Model (`model.py` -- `SapiensPose3D`)

```python
class SapiensPose3D(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(...)
        self.aggregator = CrossScaleGate(embed_dim=1024, bias_init=-5.0)
        self.head = Pose3DHead(in_channels=1024, ...)  # unchanged

    def forward(self, x):
        features = self.backbone(x)       # [mid (B,1024,40,24), final (B,1024,40,24)]
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

Add in both `_build_optimizer_frozen()` and `_build_optimizer_full()`, after the head param group.

## What NOT to Change

- `infra.py` -- no changes
- Loss computation -- no changes
- LLRD schedule -- unchanged (gamma=0.90, unfreeze_epoch=5)
- Head architecture -- unchanged (in_channels=1024)
- Pretrained weight loading -- no changes needed
