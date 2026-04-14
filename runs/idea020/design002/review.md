# Review: idea020 / design002 — Reduced Coarse Supervision Weight 0.1 (Axis A2)

**Design_ID:** idea020/design002
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design reduces the deep supervision coarse weight from 0.5 to 0.1: `loss = 0.1 * L(J1) + 1.0 * L(J2)`. One-line change in `train.py`. `refine_loss_weight = 0.1` added to `config.py`.

## Evaluation

### Completeness
All required config fields are specified. The change is fully specified: one constant in `train.py`, one config field updated. Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- Reducing the coarse weight to 0.1 is a valid hyperparameter adjustment. The coarse decoder retains a minimal supervision signal (0.1 is non-zero, avoiding degenerate feature collapse as noted in the design).
- The refinement pass continues at full weight 1.0. This is architecturally sound.
- The design correctly notes that `refine_loss_weight` should be read from config (`args.refine_loss_weight`) or hardcoded as `0.1` in train.py. This is unambiguous.
- No gradient flow changes; this is a pure loss weight adjustment.

### Constraint Adherence
- Zero new parameters. Identical VRAM to idea015/design004.
- Architecture unchanged.
- All training hyperparameters preserved.
- The `refine_loss_weight = 0.1` config field is new and clearly specified.

### Issues
None.

## Verdict: APPROVED
