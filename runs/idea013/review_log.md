# Review Log — idea013

## design001 — Small-Beta Smooth L1 (beta=0.01)
**Date:** 2026-04-10
**Verdict:** APPROVED
Reduce Smooth L1 beta from 0.05 to 0.01. Single constant change in train.py. All config fields specified. No architecture or memory impact. Mathematically correct: quadratic-to-linear transition at 10 mm provides constant gradient for medium errors.

## design002 — Large-Beta Smooth L1 (beta=0.1)
**Date:** 2026-04-10
**Verdict:** APPROVED
Increase Smooth L1 beta from 0.05 to 0.1. Single constant change in train.py. All config fields specified. Symmetric counterpart to design001. Extends quadratic regime to 100 mm for stronger large-error gradients.

## design003 — Bone-Length Auxiliary Loss
**Date:** 2026-04-10
**Verdict:** APPROVED
Add soft bone-length consistency penalty (lambda_bone=0.1) over 21 body skeleton edges. Bone edge list correctly derived from SMPLX_SKELETON filtered to body joints (0-21). Loss function fully specified. Negligible compute overhead. No new model parameters.

## design004 — Hard-Joint-Weighted Loss
**Date:** 2026-04-10
**Verdict:** APPROVED
One-shot per-joint weight computation after epoch 0. Weights clamped to [0.5, 2.0], normalized to sum to 22. Applied element-wise to per-joint Smooth L1 loss for epochs 1-19. Weight computation and weighted loss formulas are mathematically correct. No new model parameters.
