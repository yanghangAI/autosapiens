# Design Review — idea013/design004

**Design:** Hard-Joint-Weighted Loss
**Reviewer Verdict:** APPROVED

## Summary

This design computes per-joint mean training L1 error during epoch 0, derives fixed per-joint weights (clamped to [0.5, 2.0], normalized to sum to 22), and applies them element-wise to the per-joint Smooth L1 loss for epochs 1-19. Only train.py is modified.

## Evaluation

### Completeness
- The weight computation is fully specified: accumulate per-joint L1 errors during epoch 0, compute w = mean_err / mean(mean_err), clamp to [0.5, 2.0], normalize so sum = 22.
- The weighted loss computation is explicit: `F.smooth_l1_loss(..., reduction='none').mean(dim=(0, 2))` gives per-joint loss (22,), then `(per_joint_loss * joint_weights).mean()` produces scalar loss.
- The epoch 0 fallback to standard `pose_loss()` is correctly handled via `if joint_weights is not None` guard.
- Code change locations are clearly specified: accumulator initialization before epoch loop, accumulation inside epoch 0, weight computation between epochs, conditional weighted loss in subsequent epochs.
- All config fields listed.

### Mathematical Correctness
- `per_joint_err = (...).abs().mean(dim=(0, 2))` averages absolute error over batch (dim=0) and xyz (dim=2), producing shape (22,). Correct.
- Normalization `w = mean_err / mean_err.mean()` centers weights around 1.0. Correct.
- Clamping to [0.5, 2.0] prevents extreme weights. After clamping, re-normalization `w * 22.0 / w.sum()` ensures total loss magnitude is preserved. Correct.
- The weighted loss `(per_joint_loss * joint_weights).mean()` broadcasts correctly: both are (22,) tensors. The final `.mean()` averages over joints. Since weights sum to 22, this is equivalent to a weighted average with mean weight 1.0. Correct.

### Architectural Feasibility
- No new model parameters. The weight vector is a simple (22,) tensor on the device. Negligible memory.
- Accumulation during epoch 0 uses `torch.no_grad()` and adds zero backward-pass cost.
- No extra forward passes needed.

### Constraint Adherence
- LLRD schedule preserved. All fixed hyperparameters unchanged.
- infra.py, transforms, and model architecture unchanged.
- Evaluation uses standard unweighted MPJPE.
- The one-shot weighting (not per-epoch dynamic) avoids the instability pattern from idea003.

### Concerns
- The design mentions passing accumulators to `train_one_epoch` via "mutable containers (e.g., a dict)" or returning them. This leaves some implementation flexibility to the Builder, but the logic is unambiguous. The Builder should not have trouble implementing this.
- The accumulation happens inside the no_grad metric block that "already exists" -- the Builder should verify the exact location in the existing train.py to place the accumulation code correctly (it should be before the `del` statement).

## Verdict: APPROVED
