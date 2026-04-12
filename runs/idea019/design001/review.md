# Review: idea019/design001 — Bone-Length Auxiliary Loss on Refinement Output (Axis A1)

**Design_ID:** idea019/design001
**Verdict:** APPROVED

## Evaluation

### Completeness
All hyperparameters from the baseline (idea015/design001) are explicitly preserved. The single new element — `lambda_bone=0.1` — is specified in `config.py`. No fields are left for the Builder to guess.

### Mathematical Correctness
The `bone_length_loss` function is mathematically sound:
- Iterates over body edges `(a, b)` from `SMPLX_SKELETON` filtered to `a<22 and b<22` (21 edges confirmed by inspection of `infra.py`).
- Computes `||pred_a - pred_b||_2 - ||gt_a - gt_b||_2|` per edge, averages over batch and edges.
- Division by `max(len(edges), 1)` is safe.

Loss composition: `0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2) + lambda_depth*L_dep + lambda_uv*L_uv` matches the idea.md spec exactly.

### Architectural Feasibility
Zero new model parameters. No VRAM impact. The function operates on `out["joints"]` (J2) which is already computed in the baseline forward pass.

### Constraint Adherence
- Starting point: `runs/idea015/design001/code/` ✓
- Deep supervision base weights `0.5/1.0` preserved ✓
- Bone loss additive on top: ✓
- `SMPLX_SKELETON` from `infra.py` used ✓
- `BODY_IDX = slice(0,22)` — the design uses `a<22 and b<22` for edge filtering, which is equivalent ✓
- No changes to `model.py` ✓
- No changes to `infra.py` or transforms ✓

### Minor Notes
The design correctly applies bone loss to `out["joints"]` (J2 in full 70-joint space) rather than indexing with `BODY_IDX` for the loss — this is intentional since bones are defined only between body joints, so non-body joints are simply never referenced in the edge list.

No issues found. This is a clean, minimal, well-specified design.
