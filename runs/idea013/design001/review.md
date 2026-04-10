# Design Review — idea013/design001

**Design:** Small-Beta Smooth L1 (beta=0.01)
**Reviewer Verdict:** APPROVED

## Summary

This design reduces the Smooth L1 beta from 0.05 to 0.01 in the pose loss computation. It is the simplest possible change (a single numeric constant) applied only to the body pose loss in train.py.

## Evaluation

### Completeness
- The design clearly specifies the single change: `F.smooth_l1_loss(pred, target, beta=0.01)` replacing the `pose_loss()` call from infra.py.
- All config fields are listed with correct baseline values.
- The code change is explicit: add `import torch.nn.functional as F` and replace one line.

### Mathematical Correctness
- Smooth L1 with beta=0.01 transitions from quadratic to linear at 10 mm. For typical body errors of 30-100 mm, this means a constant gradient magnitude. The analysis is correct.
- The design correctly notes that depth and uv losses remain unchanged (still using `pose_loss()` with beta=0.05 from infra.py).

### Architectural Feasibility
- No architecture changes. No memory impact. Trivially fits within 11GB VRAM and 20-epoch budget.

### Constraint Adherence
- LLRD schedule (gamma=0.90, unfreeze_epoch=5) preserved.
- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup_epochs=3, grad_clip=1.0 all fixed.
- infra.py, transforms, and model architecture unchanged.
- Evaluation uses standard unweighted MPJPE.

### Concerns
- None. The design is minimal, unambiguous, and correct.

## Verdict: APPROVED
