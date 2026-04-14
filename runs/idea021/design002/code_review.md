# Code Review — idea021 / design002 — Joint-Group Query Injection Before Refine Decoder

**Design_ID:** idea021/design002
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- In `Pose3DHead.__init__()`:
  - `self.group_emb = nn.Embedding(4, hidden_dim)` with `nn.init.zeros_(self.group_emb.weight)` — correct zero-init, 4 groups × 384 = 1,536 parameters.
  - `joint_group_ids` buffer correctly constructed:
    - Joints 0-3: group 0 (pelvis/spine — default zeros).
    - Joints 4-9: group 1 (arms).
    - Joints 10-15: group 2 (legs).
    - Joints 16-21: group 0 (head/neck — default zeros, correct).
    - Joints 22-69: group 3 (extremities).
  - `self.register_buffer("joint_group_ids", joint_group_ids)` — correct.
- In `Pose3DHead.forward()` (lines 231-232):
  ```python
  group_bias = self.group_emb(self.joint_group_ids)   # (70, hidden_dim)
  queries2   = queries2 + group_bias.unsqueeze(0)     # (B, 70, hidden_dim)
  ```
  - Applied after `queries2 = out1 + R`, before `refine_decoder` call. Correct order.
- `group_emb` automatically included in `model.head.parameters()` → head optimizer group.

### train.py
- Unchanged from baseline. Correct.

### config.py
- `output_dir` correctly set to `runs/idea021/design002`.
- All other fields match design spec. No new config fields required (group_emb has no config hyperparameters).

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 929.5mm.

## Issues Found
None. The group assignment correctly implements the design spec.
