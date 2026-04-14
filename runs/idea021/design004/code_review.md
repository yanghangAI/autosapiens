# Code Review — idea021 / design004 — Kinematic Bias + Joint-Group Injection Combined

**Design_ID:** idea021/design004
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- `SMPLX_SKELETON` correctly imported from `infra`.
- `_compute_kin_bias()` helper added at module level (same implementation as design001).
- In `Pose3DHead.__init__()`:
  - `self.register_buffer("kin_bias", _compute_kin_bias(SMPLX_SKELETON, num_joints))` — correct.
  - `self.kin_bias_scale = nn.Parameter(torch.zeros(1))` — correct zero-init scalar.
  - `self.group_emb = nn.Embedding(4, hidden_dim)` with `nn.init.zeros_(self.group_emb.weight)` — correct zero-init.
  - `joint_group_ids` buffer correctly constructed (joints 4-9: group 1 arms, 10-15: group 2 legs, 22-69: group 3 extremities, rest: group 0).
- In `Pose3DHead.forward()`:
  - Correct ordering: group embedding added to `queries2` first, then `bias_matrix` computed, then `refine_decoder(queries2, memory, tgt_mask=bias_matrix)`.
  - This matches the design spec exactly.
  - Coarse decoder call `self.decoder(queries, memory)` unchanged (no tgt_mask).
- Total new trainable: 1 (`kin_bias_scale`) + 1536 (`group_emb`) = 1,537 parameters, all zero-initialized. Both auto-included in head group.

### train.py
- Unchanged from baseline. Correct. No bone loss, no symmetry loss — deliberately omitted as per design.

### config.py
- `output_dir` correctly set to `runs/idea021/design004`.
- All other fields match design spec. No new config fields required (no lambda hyperparameters for the two structural priors).

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 929.5mm.

## Issues Found
None. The combination of A1 (kinematic bias) and A2 (group injection) is correctly implemented as a clean union of design001 and design002, with no overlap or ordering issues.
