# Review: idea020 / design005 — Residual Refinement Formulation (Axis B3)

**Design_ID:** idea020/design005
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design changes the refine decoder output from absolute joint prediction to residual correction: `J2 = J1 + self.joints_out2(out2)`. Two-line change in `model.py` only.

## Evaluation

### Completeness
All required config fields are specified. The change is precisely specified: replace `J2 = self.joints_out2(out2)` with `delta = self.joints_out2(out2); J2 = J1 + delta`. Config changes limited to `output_dir`. Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- The residual formulation `J2 = J1 + delta` is correct. The loss is applied to J2 (absolute coordinates), which is unchanged from the baseline's loss computation.
- The zero-initialization warm-start analysis is correct: `joints_out2` uses `trunc_normal_(m.weight, std=0.02)` and `zeros_(m.bias)`, so `delta ≈ 0` at initialization, giving `J2 ≈ J1`. This is a genuine smooth warm-start.
- The clarification that `depth_out` and `uv_out` operate on `out2[:, 0, :]` features (not J2) is correct and important — these heads are feature-based, not coordinate-based, so the residual change does not affect them.
- `pelvis_token = out2[:, 0, :]` is unaffected by the J2 formula change.
- Gradients flow through J1 and J2 correctly: the coarse decoder still receives gradient from both `0.5*L(J1)` (through J1 directly) and `1.0*L(J2)` (through J1 in the residual path, since J2 = J1 + delta). This is the expected behavior and consistent with the idea.md specification.

### Constraint Adherence
- Zero new parameters. Identical VRAM.
- Architecture unchanged except for the single forward-pass modification.
- All training hyperparameters preserved.

### Issues
None.

## Verdict: APPROVED
