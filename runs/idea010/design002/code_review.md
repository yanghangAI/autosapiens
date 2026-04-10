# Code Review — idea010/design002 (Learned Layer Weights)

**Reviewer:** Reviewer Agent
**Date:** 2026-04-10
**Verdict:** APPROVED

## Config Check

| Field | Design Spec | config.py | Match |
|---|---|---|---|
| output_dir | runs/idea010/design002 | runs/idea010/design002 | Yes |
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
| multiscale_mode | "learned_weights" | "learned_weights" | Yes |
| multiscale_layers | [20,21,22,23] | [20,21,22,23] | Yes |

All 16 config fields match the design specification.

## Architecture Review

### Backbone (model.py)
- Returns `list[torch.Tensor]` of 4 feature maps -- matches design.
- `_run_vit_preamble` correctly handles VisionTransformer preamble with resize_pos_embed.
- Extracts indices {20, 21, 22, 23} -- matches design.
- Each intermediate normed with `self.vit.ln1` -- matches design.
- Each reshaped to (B, C, H, W) using dynamic `patch_resolution` -- correct.

### LearnedLayerWeights (model.py)
- `nn.Parameter(torch.zeros(4))` -- initialized to zeros, so `softmax([0,0,0,0]) = [0.25,0.25,0.25,0.25]` -- matches design exactly.
- Forward: `torch.softmax(self.layer_weights, dim=0)` then weighted sum via loop -- correct.
- `torch.zeros_like(features[0])` as accumulator, then `out = out + w * feat` -- numerically correct.

### SapiensPose3D (model.py)
- Backbone returns list -> aggregator combines -> head receives (B, 1024, H, W) -- correct.
- `in_channels=embed_dim=1024` -- unchanged from baseline, correct.

### Optimizer Wiring (train.py)
- Both optimizer builders include `hasattr(model, 'aggregator')` guard with `lr=lr_head` -- matches design.
- Only 4 scalar params in aggregator -- negligible.
- LR reporting indices unchanged (aggregator appended after head) -- correct.

### Weight Loading
- Unchanged from baseline, loads only backbone keys -- correct.

## Potential Issues

- None found. The implementation is minimal and clean.

## Summary

Implementation faithfully matches the design. The 4 learnable scalar weights are initialized to 0 (uniform softmax 0.25), the backbone returns a list of 4 normed feature maps from the correct layers, and the aggregator computes the softmax-weighted sum. Optimizer wiring correctly includes the aggregator group. No bugs detected. 2-epoch sanity test passed.
