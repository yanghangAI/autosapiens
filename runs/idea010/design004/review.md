# Design Review: idea010 / design004

**Design:** Cross-Scale Attention Gate
**Reviewer Verdict:** APPROVED
**Date:** 2026-04-10

## Summary

This design extracts features from layers 11 (mid) and 23 (final), computes a spatial attention gate from the mid-layer features via `sigmoid(Linear(1024, 1))`, and applies it as a residual multiplicative gate: `output = layer_23 * (1 + gate)`. Zero-init weight and bias=-5.0 ensures gate starts near zero so initial behavior matches the baseline.

## Checklist

- [x] **Layer indices correct**: Layers 11 and 23 correctly adapted from idea.md's {6, 12} for the 12-block model to the actual 24-block model. Layer 11 is approximately mid-depth (index 11 of 0-23). Correct.
- [x] **Backbone forward rewrite**: Returns list of 2 tensors `[mid, final]`, each `(B, 1024, 40, 24)`. Applies `self.vit.ln1` to both. Uses dict-based intermediate storage for clarity. Correct.
- [x] **Aggregation module**: `CrossScaleGate` with `Linear(1024, 1)`. Zero-init weight, bias=-5.0. The `sigmoid(-5.0) = 0.0067` makes `output ~ layer_23 * 1.0067 ~ layer_23` at initialization. This is correct and an improvement over idea.md's stated "zero-init bias" which would give `sigmoid(0) = 0.5` and `output = layer_23 * 1.5` -- clearly NOT baseline behavior. The Designer correctly identified and fixed this discrepancy.
- [x] **Residual form**: `output = final_feat * (1.0 + gate)` matches idea.md's specification. Correct.
- [x] **Parameter count**: 1024 + 1 = 1,025 parameters. Negligible.
- [x] **Output shape preserved**: `(B, 1024, 40, 24)`. Head unchanged. Correct.
- [x] **Optimizer wiring**: Aggregator params in separate group with `lr=lr_head`. Correct.
- [x] **LLRD schedule**: Unchanged. Correct.
- [x] **Config fields**: All 14 base fields + 3 new fields (including `gate_bias_init`). Complete and well-specified.
- [x] **No infra.py / loss changes**: Confirmed.

## Issues Found

None. The bias initialization fix from idea.md is mathematically justified and correct. Design is complete and unambiguous.
