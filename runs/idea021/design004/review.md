# Review: idea021 / design004 — Kinematic Bias + Joint-Group Injection Combined (Axis B2)

**Design_ID:** idea021/design004
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design combines design001 (kinematic soft-attention bias) and design002 (joint-group query injection) targeting the refine decoder. 1,537 new trainable parameters (1 scalar + 1,536 embedding), all zero-initialized. Changes to model.py only.

## Evaluation

### Completeness
All required config fields are specified. All model.py changes are precisely enumerated (4 steps in __init__, 4 lines in forward). No train.py changes. Config changes limited to `output_dir`. Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- The combination is a strict superset of design001 and design002, correctly applied to the same location (refine decoder pass).
- Ordering in forward(): group embedding added to `queries2` first, then `bias_matrix` computed, then both applied to `self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)`. Correct.
- Both priors are zero-initialized, so at training start the refine decoder is identical to baseline. Correct.
- `kin_bias_scale = nn.Parameter(torch.zeros(1))`: scalar init 0.0 → bias matrix all zeros at start.
- `group_emb = nn.Embedding(4, hidden_dim)` with `zeros_`: group_bias = 0 at start.
- The `_compute_kin_bias()` function is the same as design001 — correct.
- `joint_group_ids` buffer assignments match design002 — correct.
- Both `kin_bias_scale` and `group_emb` belong to `model.head.parameters()` → head optimizer group (LR=1e-4, WD=0.3). Correct.
- Total: 1 + 1,536 = 1,537 new trainable parameters. Correct.

### Constraint Adherence
- 1,537 new parameters. Negligible VRAM overhead.
- Architecture unchanged except for refine decoder modifications.
- Coarse decoder unchanged: `out1 = self.decoder(queries, memory)` with `tgt_mask=None`.
- All training hyperparameters preserved.
- Deliberately omits bone loss and symmetry loss (in line with idea.md guidance to avoid the over-regularization failure of idea019/design005).

### Issues
None. The design is a clean, well-specified combination of the two individual designs.

## Verdict: APPROVED
