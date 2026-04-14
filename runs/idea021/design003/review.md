# Review: idea021 / design003 — Bone-Length Loss on J2 with lambda=0.05 (Axis B1)

**Design_ID:** idea021/design003
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design adds a bone-length auxiliary loss on J2 only: `0.5*L(J1) + 1.0*L(J2) + 0.05*bone_loss(J2)`. The `bone_length_loss()` helper function is fully specified. `lambda_bone = 0.05` added to config.py. Zero new parameters.

## Evaluation

### Completeness
All required config fields are specified. The `bone_length_loss()` helper is fully provided. The `BODY_EDGES` precomputation and loss integration steps are precisely described. No model.py changes. `lambda_bone = 0.05` correctly added to config.py. Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- `bone_length_loss()` computes `mean |pred_bone_len - gt_bone_len|` over body edges. The torch.norm / torch.abs / torch.stack formulation is mathematically correct.
- `BODY_EDGES = [(a, b) for (a, b) in SMPLX_SKELETON if a < 22 and b < 22]`: correctly filters to body-only edges. Since `SMPLX_SKELETON` is in remapped index space (0-21 are body joints), this filter is correct.
- Indexing: `out["joints"][:, BODY_IDX]` → (B, 22, 3). `BODY_EDGES` contains indices in range [0, 21] since both `a < 22` and `b < 22`. These directly index into the (B, 22, 3) tensor. Correct.
- `lambda_bone = 0.05` (half of idea019's 0.1 which was neutral) is an appropriate conservative weight for the tighter-error refinement regime.
- The design correctly notes that bone loss applies only during training, not validation. The validate() function in infra.py is unchanged.
- The loss is applied to J2 only (refine decoder output), not J1. This is consistent with the idea.md spec.

### Constraint Adherence
- Zero new parameters. Identical VRAM.
- Architecture and model.py unchanged.
- All training hyperparameters preserved.
- `lambda_bone = 0.05` specified in config.py.

### Issues
None. This is a straightforward reuse of the idea019/design001 pattern on the stronger baseline with halved weight.

## Verdict: APPROVED
