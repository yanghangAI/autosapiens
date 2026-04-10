# Design Review: idea010 / design003

**Design:** Feature Pyramid with 3 Scales
**Reviewer Verdict:** APPROVED
**Date:** 2026-04-10

## Summary

This design extracts features from 3 evenly-spaced ViT layers (7, 15, 23) spanning early, mid, and final backbone depth. Each is projected from 1024 to 256 channels via separate linear layers, concatenated to `(B, 768, 40, 24)`, then fused via `Linear(768, 1024)` to restore the standard head input shape. Xavier init on all new projections.

## Checklist

- [x] **Layer indices correct**: Layers 7, 15, 23 correctly adapted from idea.md's {4, 8, 12} for a 12-block model to the actual 24-block model. Evenly spaced at roughly 1/3, 2/3, and 3/3 depth. Correct.
- [x] **Backbone forward rewrite**: Returns list of 3 tensors, each `(B, 1024, 40, 24)`. Applies `self.vit.ln1` to all. Correct.
- [x] **Aggregation module**: `FeaturePyramid` with `nn.ModuleList` of 3 `Linear(1024, 256)` per-scale projections + `Linear(768, 1024)` fusion. Xavier uniform init on all weights, zeros on biases. Mathematically sound.
- [x] **Parameter count**: 3 * (1024*256 + 256) = 787,200 + (768*1024 + 1024) = 787,456. Total ~1.57M. Well within budget.
- [x] **Output shape preserved**: `(B, 1024, 40, 24)`. Head unchanged. Correct.
- [x] **Optimizer wiring**: Aggregator params in separate group with `lr=lr_head`. Correct.
- [x] **LLRD schedule**: Unchanged. Correct.
- [x] **Config fields**: All 14 base fields + 2 new fields specified. Complete.
- [x] **Memory note**: 2 extra `(B, 960, 1024)` tensors ~7.5 MB each. Negligible. Correct.
- [x] **No infra.py / loss changes**: Confirmed.

## Issues Found

None. The design is complete with correct parameter math, proper initialization, and clear Builder instructions.
