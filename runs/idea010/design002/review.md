# Design Review: idea010 / design002

**Design:** Learned Layer Weights (Softmax-Weighted Sum)
**Reviewer Verdict:** APPROVED
**Date:** 2026-04-10

## Summary

This design extracts the last 4 ViT block outputs (layers 20-23) and computes a learned softmax-weighted sum using 4 learnable scalar parameters initialized to 0.0 (giving uniform 0.25 weights via softmax). Output shape is `(B, 1024, 40, 24)` -- identical to baseline. A `LearnedLayerWeights` module is placed between backbone and head.

## Checklist

- [x] **Layer indices correct**: Layers 20, 21, 22, 23 for the 24-block model. Correct.
- [x] **Backbone forward rewrite**: Returns a list of 4 tensors, each `(B, 1024, 40, 24)`. Applies `self.vit.ln1` to all intermediates. Correct.
- [x] **Aggregation module**: `LearnedLayerWeights` with `nn.Parameter(torch.zeros(4))`. Softmax of [0,0,0,0] = [0.25, 0.25, 0.25, 0.25]. Mathematically correct uniform initialization.
- [x] **Output shape preserved**: `(B, 1024, 40, 24)` unchanged. Head needs no modification. Correct.
- [x] **Parameter count**: 4 scalar parameters. Negligible.
- [x] **Optimizer wiring**: Aggregator params in separate group with `lr=lr_head`. Correct.
- [x] **LLRD schedule**: Unchanged. Correct.
- [x] **Config fields**: All 14 base fields + 2 new fields specified. Complete.
- [x] **No infra.py / loss changes**: Confirmed.

## Issues Found

None. This is the minimal multi-scale design -- only 4 new scalars. Clean and unambiguous for the Builder.
