# Design Review: idea010 / design001

**Design:** Last-4-Layer Concatenation + Linear Projection
**Reviewer Verdict:** APPROVED
**Date:** 2026-04-10

## Summary

This design extracts the last 4 transformer block outputs (layers 20-23, correctly indexed for the 24-block Sapiens 0.3B), concatenates them along the channel dimension to `(B, 4096, 40, 24)`, and projects back to `(B, 1024, 40, 24)` via a single `Linear(4096, 1024)` with Xavier init. A new `MultiScaleConcat` module is placed between backbone and head.

## Checklist

- [x] **Layer indices correct**: Layers 20, 21, 22, 23 are the last 4 blocks of the 24-block model. Correct.
- [x] **Backbone forward rewrite**: Manual iteration through `self.vit.layers` with intermediate extraction. Applies `self.vit.ln1` to all intermediates for consistent normalization. Correct approach.
- [x] **Aggregation module**: `MultiScaleConcat` with `Linear(4096, 1024)`, Xavier uniform init on weight, zeros on bias. Clean and correct.
- [x] **Output shape preserved**: Projects back to `(B, 1024, 40, 24)` so head `in_channels=1024` is unchanged. Correct.
- [x] **Parameter count**: ~4.2M new params (4096*1024 + 1024). Reasonable for 11GB VRAM budget.
- [x] **Optimizer wiring**: Aggregator params added as separate group with `lr=lr_head` in both frozen and full optimizer builders. Correct.
- [x] **LLRD schedule**: Unchanged (gamma=0.90, unfreeze_epoch=5). Correct.
- [x] **Config fields**: All 14 base fields + 2 new fields specified with exact values. Complete.
- [x] **No infra.py changes**: Confirmed.
- [x] **No loss changes**: Confirmed.
- [x] **Pretrained weight loading**: No changes needed since only backbone weights are loaded. Correct.

## Issues Found

None. The design is complete, mathematically sound, and provides unambiguous instructions for the Builder.
