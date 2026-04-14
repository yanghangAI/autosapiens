# Review: idea021 / design002 — Joint-Group Query Injection Before Refine Decoder (Axis A2)

**Design_ID:** idea021/design002
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design adds a learnable `nn.Embedding(4, 384)` zero-initialized group embedding to `queries2` before the refine decoder pass. Group assignments for all 70 joints are specified. 1,536 new trainable parameters.

## Evaluation

### Completeness
All required config fields are specified. Group assignment table is provided. Model changes are precisely listed. No train.py changes. Config changes limited to `output_dir`. Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- `group_emb = nn.Embedding(4, hidden_dim)` with `zeros_` initialization is correct. At training start, the embedding weights are all zero, so `group_bias = 0` and `queries2` is unchanged from baseline. Correct smooth warm-start.
- `group_bias = self.group_emb(self.joint_group_ids)` yields `(70, hidden_dim)`. The `.unsqueeze(0)` broadcast to `(1, 70, hidden_dim)` is correct for addition with `queries2` of shape `(B, 70, hidden_dim)`.
- Group assignment: joints 0-3 → group 0 (torso/spine), 4-9 → group 1 (arms), 10-15 → group 2 (legs), 16-21 → group 0 (head/neck), 22-69 → group 3 (extremities). This is a reasonable mapping and consistent with the idea.md spec.
- `joint_group_ids` is a `(70,)` long tensor registered as a buffer (not a parameter) — correct, it is fixed.
- `group_emb` belongs to `model.head.parameters()` automatically → head group (LR=1e-4, WD=0.3). Correct.
- `4 × 384 = 1,536` new trainable parameters. Negligible.

### Constraint Adherence
- 1,536 new parameters. Negligible VRAM overhead.
- Architecture unchanged except for the `queries2` modification before refine decoder.
- All training hyperparameters preserved.

### Issues
None. The design is consistent with the idea.md constraint that group assignments should cover all 70 joints and that the prior is zero-initialized.

## Verdict: APPROVED
