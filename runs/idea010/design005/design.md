# idea010 / design005 -- Alternating Layer Average

## Starting Point

`runs/idea004/design002/`

## Overview

Extract features from all even-indexed ViT blocks (0-indexed: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 -- i.e., every other block, 12 total) and compute a simple channel-wise mean. This uniformly samples the full depth of the backbone with no additional parameters. The head receives a smoothed representation that combines low-level spatial detail with high-level semantics.

**Correction from idea.md**: The idea.md specifies "even-indexed layers (2, 4, 6, 8, 10, 12)" assuming a 12-layer model. Since Sapiens 0.3B has 24 layers, we use every other layer (odd 0-indexed: 1, 3, 5, ..., 23) to get 12 evenly-spaced samples across the full depth. This provides the same uniform-sampling spirit with correct layer indices.

## Config Changes (`config.py`)

All values below are **unchanged** from the starting point unless explicitly listed:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea010/design005` | Updated for this run |
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
| `multiscale_mode` | `"alt_avg"` | Signals which aggregation strategy to use |
| `multiscale_layers` | `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]` | Every other block (0-indexed) |

## Architecture Changes

### 1. Backbone (`model.py` -- `SapiensBackboneRGBD`)

Modify `forward()` to extract 12 intermediate outputs and average them:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.vit.patch_embed(x)
    x = x + self.vit.pos_embed

    extract_indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
    intermediates = []
    for i, layer in enumerate(self.vit.layers):
        x = layer(x)
        if i in extract_indices:
            intermediates.append(x)

    # Apply final LayerNorm to each extracted feature
    normed = [self.vit.ln1(feat) for feat in intermediates]

    # Stack and average: list of (B, 960, 1024) -> (B, 960, 1024)
    stacked = torch.stack(normed, dim=0)  # (12, B, 960, 1024)
    averaged = stacked.mean(dim=0)         # (B, 960, 1024)

    # Reshape to (B, C, H, W)
    B, N, C = averaged.shape
    return averaged.transpose(1, 2).reshape(B, C, 40, 24)  # (B, 1024, 40, 24)
```

### 2. No Aggregation Module Needed

Since the averaging is done directly in the backbone's forward pass, no separate aggregation module is needed. The output shape `(B, 1024, 40, 24)` is identical to the baseline.

### 3. Full Model (`model.py` -- `SapiensPose3D`)

```python
class SapiensPose3D(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(...)
        self.head = Pose3DHead(in_channels=1024, ...)  # unchanged

    def forward(self, x):
        feat = self.backbone(x)   # (B, 1024, 40, 24) -- already averaged
        return self.head(feat)
```

### 4. Optimizer Wiring (`train.py`)

**No changes needed.** There are zero new parameters. The optimizer setup remains identical to the starting point.

### 5. Mathematical Detail

The output is:

```
output = (1/12) * sum_{k in {1,3,5,7,9,11,13,15,17,19,21,23}} LayerNorm(block_k(x))
```

This is a uniform average over 12 evenly-spaced transformer block outputs spanning the full depth of the network. Early blocks contribute local texture/edge features; middle blocks contribute part-level structure; late blocks contribute global semantic context.

**Total new parameters: 0** -- this is the simplest possible multi-scale baseline.

### 6. Memory Note

Storing 12 intermediate `(B, 960, 1024)` tensors requires ~12 * 3.75 MB = ~45 MB at fp32 with batch=4. This is well within the 11GB budget. However, the `torch.stack` + `mean` operation is done after collecting all intermediates, so peak memory includes holding all 12 tensors simultaneously. An alternative memory-efficient implementation uses a running sum:

```python
# Memory-efficient alternative (preferred):
running_sum = None
count = 0
for i, layer in enumerate(self.vit.layers):
    x = layer(x)
    if i in extract_indices:
        normed = self.vit.ln1(x)
        if running_sum is None:
            running_sum = normed
        else:
            running_sum = running_sum + normed
        count += 1
averaged = running_sum / count
```

The Builder should use the **running-sum** approach to minimize peak memory.

## What NOT to Change

- `infra.py` -- no changes
- Loss computation -- no changes
- LLRD schedule -- unchanged (gamma=0.90, unfreeze_epoch=5)
- Head architecture -- unchanged (in_channels=1024)
- Pretrained weight loading -- no changes needed
- Optimizer wiring -- no changes needed (zero new parameters)
