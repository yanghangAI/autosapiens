# idea010 / design002 -- Learned Layer Weights (Softmax-Weighted Sum)

## Starting Point

`runs/idea004/design002/`

## Overview

Extract the outputs of the last 4 ViT transformer blocks (0-indexed layers 20, 21, 22, 23) and compute a learned weighted average: `output = sum(softmax(w)[i] * layer_i)` where `w` is a vector of 4 learnable scalars initialized to equal values (so initial weights are uniform 0.25 each). This preserves the `(B, 1024, 40, 24)` shape identically to the baseline -- no projection layer needed, only 4 new scalar parameters. Inspired by ELMo-style layer mixing.

## Config Changes (`config.py`)

All values below are **unchanged** from the starting point unless explicitly listed:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea010/design002` | Updated for this run |
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
| `multiscale_mode` | `"learned_weights"` | Signals which aggregation strategy to use |
| `multiscale_layers` | `[20, 21, 22, 23]` | 0-indexed ViT block indices to extract |

## Architecture Changes

### 1. Backbone (`model.py` -- `SapiensBackboneRGBD`)

Modify `forward()` to iterate through `self.vit.layers` and return multiple intermediate outputs. The backbone should return a **list** of tensors rather than a single tensor:

```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    x = self.vit.patch_embed(x)
    x = x + self.vit.pos_embed

    extract_indices = {20, 21, 22, 23}
    intermediates = []
    for i, layer in enumerate(self.vit.layers):
        x = layer(x)
        if i in extract_indices:
            intermediates.append(x)

    # Apply final LayerNorm to each extracted feature
    normed = [self.vit.ln1(feat) for feat in intermediates]

    # Reshape each to (B, C, H, W)
    results = []
    for feat in normed:
        B, N, C = feat.shape
        results.append(feat.transpose(1, 2).reshape(B, C, 40, 24))

    return results  # list of 4 tensors, each (B, 1024, 40, 24)
```

### 2. Feature Aggregation Module (`model.py` -- new `LearnedLayerWeights` module)

```python
class LearnedLayerWeights(nn.Module):
    """Softmax-normalized learned weighted sum of multi-layer features."""
    def __init__(self, num_layers: int = 4):
        super().__init__()
        # Initialize all weights to 0.0 so softmax gives uniform 1/num_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # features: list of (B, C, H, W) tensors
        weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers,)
        out = torch.zeros_like(features[0])
        for w, feat in zip(weights, features):
            out = out + w * feat
        return out  # (B, 1024, 40, 24)
```

**Mathematical detail**: At initialization, `self.layer_weights = [0, 0, 0, 0]`, so `softmax([0,0,0,0]) = [0.25, 0.25, 0.25, 0.25]`. The output is a uniform average of the 4 layer outputs, which is a reasonable starting point. During training, the weights will specialize to upweight the most informative layers.

### 3. Full Model (`model.py` -- `SapiensPose3D`)

```python
class SapiensPose3D(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(...)
        self.aggregator = LearnedLayerWeights(num_layers=4)
        self.head = Pose3DHead(in_channels=1024, ...)  # unchanged

    def forward(self, x):
        features = self.backbone(x)      # list of 4 x (B, 1024, 40, 24)
        feat = self.aggregator(features)  # (B, 1024, 40, 24)
        return self.head(feat)
```

### 4. Optimizer Wiring (`train.py`)

Add the aggregator's 4 scalar parameters as a separate param group with `lr=1e-4`:

```python
if hasattr(model, 'aggregator'):
    param_groups.append({
        "params": list(model.aggregator.parameters()),
        "lr": lr_head,
        "initial_lr": lr_head,
    })
```

Add this block in both `_build_optimizer_frozen()` and `_build_optimizer_full()`, after the head param group. Ensure `requires_grad = True` for aggregator parameters.

### 5. Parameter Count Estimate

- Only 4 new scalar parameters (the `layer_weights` vector)
- Negligible memory and compute overhead

## What NOT to Change

- `infra.py` -- no changes
- Loss computation -- no changes
- LLRD schedule -- unchanged (gamma=0.90, unfreeze_epoch=5)
- Head architecture -- completely unchanged (in_channels=1024)
- Pretrained weight loading -- no changes needed
