# Code Review — idea010/design003 (Feature Pyramid with 3 Scales)

**Reviewer:** Reviewer Agent
**Date:** 2026-04-10
**Verdict:** APPROVED

## Config Check

| Field | Design Spec | config.py | Match |
|---|---|---|---|
| output_dir | runs/idea010/design003 | runs/idea010/design003 | Yes |
| lr_backbone | 1e-4 | 1e-4 | Yes |
| lr_head | 1e-4 | 1e-4 | Yes |
| gamma | 0.90 | 0.90 | Yes |
| unfreeze_epoch | 5 | 5 | Yes |
| epochs | 20 | 20 | Yes |
| warmup_epochs | 3 | 3 | Yes |
| head_hidden | 256 | 256 | Yes |
| head_num_heads | 8 | 8 | Yes |
| head_num_layers | 4 | 4 | Yes |
| head_dropout | 0.1 | 0.1 | Yes |
| drop_path | 0.1 | 0.1 | Yes |
| lambda_depth | 0.1 | 0.1 | Yes |
| lambda_uv | 0.2 | 0.2 | Yes |
| multiscale_mode | "pyramid3" | "pyramid3" | Yes |
| multiscale_layers | [7,15,23] | [7,15,23] | Yes |

All 16 config fields match the design specification.

## Architecture Review

### Backbone (model.py)
- Returns `list[torch.Tensor]` of 3 feature maps -- matches design.
- Extracts indices {7, 15, 23} -- matches design (early, mid, final).
- Each intermediate normed with `self.vit.ln1` -- matches design.
- Each reshaped to (B, C, H, W) using dynamic `patch_resolution` -- correct.

### FeaturePyramid (model.py)
- `nn.ModuleList` of 3 `nn.Linear(1024, 256)` -- matches design (3 per-scale projections).
- `fuse_proj = nn.Linear(768, 1024)` -- 256*3=768 input, 1024 output -- matches design.
- Xavier uniform init on all projection weights, zeros init on biases -- matches design.
- Forward: for each scale, flatten(2).transpose(1,2) -> project to 256d -> collect; concatenate along last dim -> fuse to 1024d -> reshape to (B, C, H, W) -- correct tensor flow.
- H, W recovered from `features[0].shape[2], features[0].shape[3]` -- correct.

### Parameter count verification
- 3 x Linear(1024, 256): 3 x (1024*256 + 256) = 3 x 262,400 = 787,200
- 1 x Linear(768, 1024): 768*1024 + 1024 = 787,456 + 1024 = 788,480
- Total: ~1.57M -- matches design estimate.

### SapiensPose3D (model.py)
- Backbone -> aggregator -> head pipeline correct.
- `in_channels=embed_dim=1024` (after fuse_proj) -- correct.

### Optimizer Wiring (train.py)
- Both optimizer builders include aggregator guard with `lr=lr_head` -- matches design.
- LR reporting indices unchanged (aggregator appended after head) -- correct.

### Weight Loading
- Unchanged, loads only backbone keys -- correct. Aggregator params randomly initialized (Xavier).

## Potential Issues

- None found.

## Summary

Implementation faithfully matches the design. The backbone extracts 3 scales from the correct layers, the FeaturePyramid projects each to 256d, concatenates to 768d, and fuses to 1024d with Xavier initialization throughout. Optimizer wiring correctly includes the aggregator. No bugs detected. 2-epoch sanity test passed.
