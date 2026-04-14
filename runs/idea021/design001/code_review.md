# Code Review — idea021 / design001 — Kinematic Soft-Attention Bias in Refine Decoder

**Design_ID:** idea021/design001
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- `SMPLX_SKELETON` correctly imported from `infra`.
- `_compute_kin_bias()` helper is added at module level, using BFS with `collections.deque`. Implementation matches the design spec:
  - 1-hop: 1.0, 2-hop: 0.5, 3-hop: 0.25, beyond 3 hops: 0.0.
  - Correctly skips edges where `a >= num_joints` or `b >= num_joints`.
  - `hop >= 3` early-exit prevents expansion beyond 3 hops.
- In `Pose3DHead.__init__()`:
  - `self.register_buffer("kin_bias", _compute_kin_bias(SMPLX_SKELETON, num_joints))` — correct.
  - `self.kin_bias_scale = nn.Parameter(torch.zeros(1))` — correct zero-init scalar.
- In `Pose3DHead.forward()`:
  - `bias_matrix = self.kin_bias_scale * self.kin_bias` — shape (70, 70), correct.
  - `out2 = self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)` — bias applied to refine decoder only.
  - `out1 = self.decoder(queries, memory)` — coarse decoder has no tgt_mask (unchanged).
- `kin_bias_scale` automatically included in `model.head.parameters()` → head optimizer group.

### train.py
- Unchanged from baseline. Correct.

### config.py
- `output_dir` correctly set to `runs/idea021/design001`.
- All other fields match design spec: head_hidden=384, refine_loss_weight=0.5, all training hyperparameters unchanged from idea015/design004.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 797.9mm.

## Issues Found
None.
