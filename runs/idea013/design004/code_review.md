# Code Review — idea013/design004

**Design:** Hard-Joint-Weighted Loss
**Reviewer verdict:** APPROVED

## Checklist

1. **train_one_epoch signature** — Lines 164-165: accepts `joint_weights=None` and `err_accum=None` parameters. Correct.

2. **Weighted loss (epochs >= 1)** — Lines 185-193: When `joint_weights is not None`, computes per-joint Smooth L1 with `reduction='none'`, takes `.mean(dim=(0, 2))` to get shape (22,), multiplies by `joint_weights`, then `.mean()` for scalar loss. Beta=0.05 matches design. When `joint_weights is None` (epoch 0), falls back to standard `pose_loss()`. Correct.

3. **Error accumulation during epoch 0** — Lines 214-217: Inside the no_grad block, when `err_accum is not None`, computes per-joint L1 error averaged over batch and xyz dimensions, shape (22,), accumulates into `err_accum["sum"]` and increments `err_accum["count"]`. Correct per design.

4. **Weight computation after epoch 0** — Lines 387-394 in main():
   - `mean_err = err_accum["sum"] / err_accum["count"]` — correct
   - `w = mean_err / mean_err.mean()` — normalize so mean=1.0, correct
   - `w = w.clamp(0.5, 2.0)` — matches design [0.5, 2.0] clamp
   - `w = w * 22.0 / w.sum()` — re-normalize so sum=22, correct
   - `joint_weights = w.detach()` — frozen, correct
   - Print statement for diagnostics. Correct.

5. **Passing accumulators** — Lines 381-384: `cur_err_accum = err_accum if (epoch == 0 and joint_weights is None) else None`. Only passes accumulator during epoch 0, before weights are computed. Correct.

6. **Config verification:**
   - `output_dir` = correct path (idea013/design004)
   - All config fields match design table (weight_decay=0.03, epochs=20, etc.)
   - No new config fields for weight clamp/epoch (hardcoded as specified).

7. **LLRD optimizer logic** — Present and correct.

8. **del statement** — Line 231: correctly includes all standard loss variables.

## Issues

None found. The implementation faithfully follows the design's one-shot weight computation protocol.
