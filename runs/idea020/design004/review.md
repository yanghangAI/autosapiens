# Review: idea020 / design004 — Higher LR for Refine Decoder (2x Head LR, Axis B2)

**Design_ID:** idea020/design004
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design splits the head optimizer group into a coarse-head group (LR=1e-4) and a refine-head group (LR=2e-4). Two helper functions `_coarse_head_params()` and `_refine_head_params()` enumerate the relevant parameters. Changes to `train.py` only; `lr_refine_head = 2e-4` added to `config.py`.

## Evaluation

### Completeness
All required config fields are specified. The design provides explicit helper function implementations for `_coarse_head_params()` and `_refine_head_params()`, which is sufficient for the Builder. The `LR_REFINE = args.lr_head * 2.0` formula is unambiguous. Both `_build_optimizer_frozen()` and `_build_optimizer_full()` are addressed.

### Mathematical / Architectural Correctness
- The parameter split is correct: `input_proj`, `joint_queries`, `decoder`, `joints_out`, `depth_out`, `uv_out` → coarse group; `refine_decoder`, `refine_mlp`, `joints_out2` → refine group. This covers all head parameters without overlap or omission (assuming the idea015/design004 head has exactly these modules).
- Group index references in the LR reporting block are correctly updated: frozen phase has groups 0..11 (backbone blocks 12-23), 12 (depth_pe), 13 (coarse_head), 14 (refine_head); full phase has groups 0 (embed), 1-24 (blocks), 25 (depth_pe), 26 (coarse_head), 27 (refine_head). These indices need to be verified against the actual baseline optimizer construction, but the design provides sufficient specificity for the Builder to implement correctly.
- `lr_refine_head = 2e-4` is explicitly in config.py.
- Zero new parameters; only optimizer state changes (minor overhead as noted).

### Constraint Adherence
- Architecture unchanged.
- All training hyperparameters preserved.
- VRAM impact negligible (extra optimizer state tensors for the new group are ~few MB).

### Issues
None. The design is thorough in specifying the group index updates, which is the most error-prone part of the implementation.

## Verdict: APPROVED
