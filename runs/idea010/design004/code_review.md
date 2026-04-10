# Code Review — idea010/design004 (Cross-Scale Attention Gate)

**Reviewer:** Reviewer Agent
**Date:** 2026-04-10
**Verdict:** APPROVED

## Config Check

| Field | Design Spec | config.py | Match |
|---|---|---|---|
| output_dir | runs/idea010/design004 | runs/idea010/design004 | Yes |
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
| multiscale_mode | "cross_gate" | "cross_gate" | Yes |
| multiscale_layers | [11,23] | [11,23] | Yes |
| gate_bias_init | -5.0 | -5.0 | Yes |

All 17 config fields match the design specification.

## Architecture Review

### Backbone (model.py)
- Returns `list[torch.Tensor]` of 2 feature maps: [mid_feat, final_feat].
- Extracts indices {11, 23} using a dict -- matches design.
- Both intermediates normed with `self.vit.ln1` -- matches design.
- Each reshaped to (B, C, H, W) via dynamic `patch_resolution` -- correct.
- Return order is [layer_11, layer_23] -- matches design (mid first, final second).

### CrossScaleGate (model.py)
- `nn.Linear(1024, 1)` -- matches design (embed_dim -> 1 scalar gate per spatial location).
- `nn.init.zeros_(self.gate_proj.weight)` -- zero-init weight -- matches design.
- `nn.init.constant_(self.gate_proj.bias, -5.0)` -- bias_init=-5.0 -- matches design.
- Forward: flatten(2).transpose(1,2) on mid_feat -> gate_proj -> sigmoid -> reshape to (B,1,H,W) -> `final_feat * (1.0 + gate)` -- matches design exactly.
- At init: gate_proj output = -5.0 for all positions, sigmoid(-5.0) = 0.0067, so output ~ final_feat * 1.0067 ~ final_feat -- near-identical to baseline. Correct.

### Parameter count verification
- Linear(1024, 1): 1024 + 1 = 1,025 parameters -- matches design.

### SapiensPose3D (model.py)
- Backbone -> aggregator -> head pipeline correct.
- `bias_init=-5.0` passed from constructor -- matches config.
- `in_channels=embed_dim=1024` -- correct (gate does not change channel dim).

### Optimizer Wiring (train.py)
- Both optimizer builders include aggregator guard with `lr=lr_head` -- matches design.
- LR reporting indices unchanged -- correct.

### Weight Loading
- Unchanged, loads only backbone keys -- correct.

## Potential Issues

- None found.

## Summary

Implementation faithfully matches the design. The backbone extracts layers 11 and 23, the CrossScaleGate applies a residual multiplicative gate with the correct initialization (zero weight, -5.0 bias), and the output form `final_feat * (1 + gate)` ensures near-baseline behavior at initialization. Optimizer wiring correctly includes the aggregator. No bugs detected. 2-epoch sanity test passed.
