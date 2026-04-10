# Code Review — idea010/design001 (Last-4-Layer Concatenation)

**Reviewer:** Reviewer Agent
**Date:** 2026-04-10
**Verdict:** APPROVED

## Config Check

| Field | Design Spec | config.py | Match |
|---|---|---|---|
| output_dir | runs/idea010/design001 | runs/idea010/design001 | Yes |
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
| multiscale_mode | "concat4" | "concat4" | Yes |
| multiscale_layers | [20,21,22,23] | [20,21,22,23] | Yes |

All 16 config fields match the design specification.

## Architecture Review

### Backbone (model.py)
- `_run_vit_preamble` correctly replicates the VisionTransformer preamble: patch_embed, resize_pos_embed, drop_after_pos, pre_norm. This is more robust than the simplified design pseudocode (which used raw `self.vit.patch_embed(x)` + `self.vit.pos_embed`).
- `forward()` extracts indices {20, 21, 22, 23} -- matches design.
- Each intermediate is normed with `self.vit.ln1` -- matches design.
- Concatenation along channel dim: `torch.cat(normed, dim=-1)` produces (B, N, 4096) -- correct.
- Reshape to (B, 4096, H, W) using dynamic `patch_resolution` -- correct and more robust than hardcoded 40x24.

### MultiScaleConcat (model.py)
- `nn.Linear(4096, 1024)` via `embed_dim * num_layers = 1024*4 = 4096` input, `embed_dim=1024` output -- matches design.
- Xavier uniform init on weight, zeros init on bias -- matches design.
- Forward: flatten(2).transpose(1,2) -> proj -> transpose back -> reshape -- correct tensor flow.

### SapiensPose3D (model.py)
- Backbone -> aggregator -> head pipeline correct.
- `in_channels=embed_dim=1024` for head (after projection) -- correct.

### Optimizer Wiring (train.py)
- Both `_build_optimizer_frozen` and `_build_optimizer_full` include `hasattr(model, 'aggregator')` guard.
- Aggregator params get `lr=lr_head=1e-4` and `initial_lr=lr_head` -- matches design.
- `requires_grad = True` explicitly set for aggregator -- correct.
- LR reporting indices: block23 at index 11 (frozen) / 24 (full), head at 12 / 25 -- these remain correct because the aggregator group is appended **after** the head group.

### Weight Loading
- `load_sapiens_pretrained` unchanged, only loads `backbone.*` keys -- correct. Aggregator params will be randomly initialized (Xavier).

## Potential Issues

- None found. Implementation is clean and matches design precisely.

## Summary

Implementation faithfully matches the design specification. Config values are exact. Backbone extracts the correct layer indices, applies LN, concatenates, and the aggregator projects back to 1024. Optimizer wiring correctly includes the aggregator as a separate group. No bugs detected. 2-epoch sanity test passed.
