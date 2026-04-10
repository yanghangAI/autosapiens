# Design Review — idea012/design004

**Reviewer verdict: APPROVED**

## Summary

R-Drop consistency regularization: add MSE consistency loss between two stochastic forward passes with `alpha=1.0`. Second pass uses `torch.no_grad()` + `.detach()`. Consistency computed on body joints only.

## Evaluation

### Completeness
- Starting point: `runs/idea004/design002/`. Correct.
- New config field: `rdrop_alpha = 1.0`. Clearly specified.
- Full implementation pseudocode provided with all key requirements:
  1. Second forward pass under `torch.no_grad()` -- avoids doubling backward graph memory. Correct.
  2. `pred2_body.detach()` -- gradients flow only through pred1. Correct.
  3. Consistency on body joints only (`joints_3d[:, BODY_IDX, :]`), not pelvis depth/uv. Matches idea012 constraint.
  4. Model in `train()` mode for both passes (dropout/drop_path active). Correct.
  5. Consistency loss added before `loss.backward()` and gradient accumulation. Correct.
- Logging of `consistency_loss` metric requested. Good practice.

### Mathematical Correctness
- MSE between two stochastic forward pass outputs is a valid regression analog of KL-divergence R-Drop. The formulation `L_task + alpha * MSE(pred1_body, pred2_body)` is standard.
- With `alpha=1.0`, the consistency penalty is weighted equally with task loss. This is aggressive but the design provides clear rationale.

### Feasibility
- Two forward passes per step: ~1.5x wall time (forward-only is cheaper than forward+backward). Acceptable per idea012 constraints.
- Memory: second pass is `no_grad`, so only stores activations for first pass backward. Peak VRAM approximately same as baseline. Within 11GB.
- 20-epoch budget is wall-time flexible per constraints.

### Config Fields
- All unchanged parameters listed in table. `rdrop_alpha=1.0` is the only addition.

### Constraint Adherence
- LLRD schedule kept fixed. Correct.
- No augmentation or architecture changes. Correct.
- Second pass uses `torch.no_grad()` and `.detach()` per idea012 constraint. Correct.
- Consistency on body joints only, not pelvis. Correct per idea012 constraint.

## Issues Found
None. The implementation details are thorough and leave no ambiguity for the Builder.
