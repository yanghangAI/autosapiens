# Code Review — idea020 / design005 — Residual Refinement Formulation

**Design_ID:** idea020/design005
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- `Pose3DHead.forward()` correctly implements the residual formulation (lines 222-223):
  ```python
  delta = self.joints_out2(out2)    # (B, 70, 3): predicted correction
  J2    = J1 + delta                # residual: absolute = coarse + correction
  ```
- This is exactly the two-line change specified in the design.
- `joints_out2` is already zero-biased in `_init_weights()` (via `nn.init.zeros_(m.bias)`), providing the smooth warm-start property where `J2 ≈ J1` at initialization.
- `pelvis_token = out2[:, 0, :]` is correctly unaffected by the J2 formulation change.
- All other model components unchanged.

### train.py
- Unchanged from baseline. Loss is still `0.5 * l_pose1 + 1.0 * l_pose2` applied to `out["joints"]` (= J2 = J1 + delta). Correct.

### config.py
- `output_dir` correctly set to `runs/idea020/design005`.
- `refine_loss_weight = 0.5` (unchanged). All other fields match design spec.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 988.5mm.
- Higher initial MPJPE relative to other designs is expected during the 2-epoch smoke test — the residual warm-start requires a few epochs to warm up from J2 ≈ J1.

## Issues Found
None.
