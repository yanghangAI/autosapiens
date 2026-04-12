# Review: idea019/design005 — Combined Anatomical Priors: Bone-Length + Symmetry + Kinematic Bias (Axis B3)

**Design_ID:** idea019/design005
**Verdict:** APPROVED

## Evaluation

### Completeness
All baseline hyperparameters preserved. All four new config fields are specified: `lambda_bone=0.1`, `lambda_sym=0.05`, `kin_bias_max_hops=3`, `kin_bias_scale_init=0.0`. The combined forward pass and loss functions are fully written out. No ambiguity for the Builder.

### Mathematical Correctness
This design correctly combines the three approved components from designs 001, 002, and 003:

**Loss:**
`0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2) + 0.05*sym_loss(J2) + lambda_depth*L_dep + lambda_uv*L_uv`
This exactly matches the idea.md Axis B3 specification. ✓

**Kinematic bias:**
`_build_kin_bias` BFS is identical to design002. kin_bias_scale initialized at 0.0, bias applied only in pass 2 via manual layer loop. ✓

**bone_length_loss and symmetry_loss:** Identical to designs 001 and 003 respectively. ✓

The forward pass code correctly reconstructs the pass-2 output dict including `pelvis_depth` and `pelvis_uv` from `out2[:, 0, :]`. ✓

### Architectural Feasibility
- 1 scalar parameter (`kin_bias_scale`) + `(70,70)` buffer. ~20 KB extra memory.
- Zero parameters from loss functions.
- Total new learnable params: 1. No OOM risk at batch=4. ✓

### Code Quality Notes
The design has a minor style issue: `import collections as _col` inside `_build_kin_bias` body while `import collections` is also at the top level. This is inert — the Builder should use only the top-level import and remove the redundant alias. Non-fatal.

### Constraint Adherence
- kin_bias_scale in `head_params` group (auto via `model.head`) ✓
- Soft additive bias only, never -inf ✓
- Learned scalar init=0.0 ✓
- Deep supervision base weights 0.5/1.0 preserved ✓
- All baseline HPs preserved ✓
- No changes to `infra.py` or transforms ✓

No blocking issues. Combination of three individually-approved components into a single design is internally consistent and well-specified.
