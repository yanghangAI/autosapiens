# Design Review: idea010 / design005

**Design:** Alternating Layer Average
**Reviewer Verdict:** APPROVED
**Date:** 2026-04-10

## Summary

This design extracts features from 12 evenly-spaced ViT blocks (odd indices: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23) and computes a simple channel-wise mean. Zero new parameters. The backbone forward returns a single `(B, 1024, 40, 24)` tensor, so no aggregation module or optimizer changes are needed.

## Checklist

- [x] **Layer indices correct**: Idea.md specified "even-indexed layers (2, 4, 6, 8, 10, 12)" for a 12-block model. The Designer correctly adapted this to 12 evenly-spaced layers across the 24-block model, using every other block. The choice of odd vs even indices is immaterial -- what matters is uniform depth sampling. Correct.
- [x] **Backbone forward rewrite**: Manual iteration through `self.vit.layers`, applies `self.vit.ln1` to each extracted intermediate before averaging. Both stack-and-mean and running-sum implementations provided; running-sum recommended for memory efficiency. Correct.
- [x] **No aggregation module needed**: Averaging done in backbone forward. Output shape `(B, 1024, 40, 24)` matches baseline exactly. Correct.
- [x] **Parameter count**: 0 new parameters. Simplest possible multi-scale design. Correct.
- [x] **No optimizer changes**: Confirmed -- no new params to optimize.
- [x] **LLRD schedule**: Unchanged. Correct.
- [x] **Config fields**: All 14 base fields + 2 new fields specified. Complete.
- [x] **Memory note**: 12 intermediate tensors at ~3.75 MB each = ~45 MB. Running-sum approach reduces this to holding only 1 extra tensor. Well within budget either way. Correct.
- [x] **No infra.py / loss / head changes**: Confirmed.

## Issues Found

None. The design is clean, zero-parameter, and provides both a straightforward implementation and a memory-efficient alternative for the Builder.
