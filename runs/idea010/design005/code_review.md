# Code Review — idea010/design005 (Alternating Layer Average)

**Reviewer:** Reviewer Agent
**Date:** 2026-04-10
**Verdict:** APPROVED

## Config Check

| Field | Design Spec | config.py | Match |
|---|---|---|---|
| output_dir | runs/idea010/design005 | runs/idea010/design005 | Yes |
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
| multiscale_mode | "alt_avg" | "alt_avg" | Yes |
| multiscale_layers | [1,3,5,7,9,11,13,15,17,19,21,23] | [1,3,5,7,9,11,13,15,17,19,21,23] | Yes |

All 16 config fields match the design specification.

## Architecture Review

### Backbone (model.py)
- Extracts indices {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23} -- matches design (12 odd-indexed layers).
- Uses **running-sum** approach as recommended by design: `running_sum = normed` on first hit, then `running_sum = running_sum + normed` for subsequent -- matches design's preferred memory-efficient implementation.
- Each extracted intermediate is normed with `self.vit.ln1` before accumulation -- matches design.
- Final average: `running_sum / count` where count=12 -- correct.
- Returns single tensor (B, C, H, W) -- matches design (no separate aggregator needed).
- Uses dynamic `patch_resolution` for reshape -- correct.

### SapiensPose3D (model.py)
- No aggregator module -- correct, design specifies zero new parameters.
- Direct `self.head(self.backbone(x))` -- matches baseline structure.

### Optimizer Wiring (train.py)
- No aggregator handling needed -- correct.
- Optimizer builders are identical to the starting point (idea004/design002): 13 groups frozen, 26 groups full -- correct.
- LR reporting indices: block23=11/24, head=12/25 -- correct.

### Weight Loading
- Unchanged -- correct.

### Parameter Count
- Zero new parameters -- matches design.

## Potential Issues

- None found. The running-sum implementation is memory-efficient as requested by the design.

## Summary

Implementation faithfully matches the design. The backbone extracts 12 odd-indexed layers using a running-sum approach (as recommended by design for memory efficiency), applies LayerNorm to each before accumulation, and divides by count=12. No aggregator module or extra parameters. Optimizer is unchanged from baseline. No bugs detected. 2-epoch sanity test passed.
