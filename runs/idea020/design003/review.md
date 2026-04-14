# Review: idea020 / design003 — L1 Loss on Refinement Pass Only (Axis B1)

**Design_ID:** idea020/design003
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design replaces the Smooth L1 loss for J2 with pure `F.l1_loss` while keeping Smooth L1 (beta=0.05) for J1. One-line change in `train.py`.

## Evaluation

### Completeness
All required config fields are specified. The change is precisely specified: replace `l_pose2 = pose_loss(out["joints"][...])` with `l_pose2 = F.l1_loss(out["joints"][...])`. The note to verify `import torch.nn.functional as F` is appropriate. Config changes are limited to `output_dir`.

### Mathematical / Architectural Correctness
- `F.l1_loss` computes mean absolute error; this is the correct function for pure L1 loss.
- The coarse pass retains `pose_loss()` (Smooth L1 beta=0.05). This asymmetry is intentional and well-justified (early training has large coarse errors; refinement operates at smaller error scales).
- The loss scale compatibility between Smooth L1 and L1 is noted: both compute mean absolute-value-like quantities, so the 0.5:1.0 weighting ratio is still reasonable in terms of magnitude.
- `BODY_IDX` indexing is consistent between `l_pose1` and `l_pose2`.
- Zero new parameters, no model changes.

### Constraint Adherence
- Identical VRAM and parameter count to idea015/design004.
- Architecture, optimizer, and all other config values unchanged.

### Issues
None.

## Verdict: APPROVED
