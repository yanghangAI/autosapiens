# Review: idea019/design003 — Left-Right Symmetry Loss (Axis B1)

**Design_ID:** idea019/design003
**Verdict:** APPROVED

## Evaluation

### Completeness
All baseline hyperparameters preserved. New field `lambda_sym=0.05` specified in `config.py`. `SYM_PAIRS` fully enumerated with exact index tuples. No ambiguity for the Builder.

### Mathematical Correctness
The `symmetry_loss` function computes:
```
mean over pairs of |left_bone_len - right_bone_len|
```
where each bone length is `||pred_a - pred_b||_2`. This is the correct L1 symmetry penalty. Division by `max(len(sym_pairs), 1)` is safe.

Loss composition: `0.5*L(J1) + 1.0*L(J2) + 0.05*sym_loss(J2) + lambda_depth*L_dep + lambda_uv*L_uv` matches the idea.md spec exactly.

### Index Correctness (Verified)
Since `ACTIVE_JOINT_INDICES[0:22] = range(0, 22)` (identity mapping for body joints), the remapped indices for body joints are identical to the original SMPL-X raw indices. The `SYM_PAIRS` tuples were cross-checked against `_SMPLX_BONES_RAW`:

| Pair | Segment | Left edge exists | Right edge exists |
|------|---------|----------------|-----------------|
| (13,16,14,17) | upper arm | ✓ (9,13),(13,16) | ✓ (9,14),(14,17) |
| (16,18,17,19) | forearm | ✓ | ✓ |
| (18,20,19,21) | hand root | ✓ | ✓ |
| (1,4,2,5) | thigh | ✓ | ✓ |
| (4,7,5,8) | shin | ✓ | ✓ |
| (7,10,8,11) | foot | ✓ | ✓ |

All 6 pairs use valid body joint indices (all < 22) and correspond to confirmed skeleton edges. The design's note "The Builder should verify these indices by inspecting `_ORIG_TO_NEW` and `ACTIVE_JOINT_INDICES` in `infra.py`" is appropriate caution; however, verification confirms all indices are correct.

### Architectural Feasibility
Zero new parameters. No VRAM impact. Clean loss-only change.

### Constraint Adherence
- Zero new model parameters ✓
- `lambda_sym=0.05` specified in config ✓
- Applied to J2 only ✓
- No changes to `model.py`, `infra.py`, or transforms ✓
- All baseline HPs preserved ✓

No issues found.
