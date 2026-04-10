# Design Review — idea013/design002

**Design:** Large-Beta Smooth L1 (beta=0.1)
**Reviewer Verdict:** APPROVED

## Summary

This design increases the Smooth L1 beta from 0.05 to 0.1 in the pose loss computation. Mirror of design001 in the opposite direction. Single numeric constant change in train.py.

## Evaluation

### Completeness
- Explicit code change: `F.smooth_l1_loss(pred, target, beta=0.1)` replacing `pose_loss()`.
- All config fields listed with correct baseline values.
- Rationale for larger beta (more L2-like, stronger gradients for large errors) is well-explained.

### Mathematical Correctness
- With beta=0.1, the quadratic regime extends to 100 mm. Errors above 100 mm get constant gradient; errors below get gradient proportional to error magnitude. This analysis is correct.
- The design correctly notes that AdamW's per-parameter adaptive LR compensates for the loss scale difference (quadratic part is flatter with larger beta). This is a valid observation.
- Depth and uv losses remain unchanged.

### Architectural Feasibility
- No architecture changes. No memory impact. Trivially within constraints.

### Constraint Adherence
- All constraints satisfied: LLRD preserved, fixed hyperparameters unchanged, no modifications to infra.py/transforms/model.

### Concerns
- None. Symmetric counterpart to design001 for characterizing beta sensitivity.

## Verdict: APPROVED
