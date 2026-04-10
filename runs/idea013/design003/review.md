# Design Review — idea013/design003

**Design:** Bone-Length Auxiliary Loss
**Reviewer Verdict:** APPROVED

## Summary

This design adds a soft bone-length consistency penalty (lambda_bone=0.1) to the training loss. It computes L2 bone lengths for 21 body skeleton edges from both predictions and ground truth, then penalizes the absolute difference in bone lengths. Only train.py is modified.

## Evaluation

### Completeness
- The bone edge list is explicitly enumerated (21 edges) and matches the body-only filtering of SMPLX_SKELETON from infra.py. Verified: `_SMPLX_BONES_RAW` entries where both endpoints are in 0-21 map to new indices 0-21 (since body joints are the first 22 entries in ACTIVE_JOINT_INDICES). The design's BODY_BONES list is correct.
- The `bone_length_loss` function is fully specified with clear pseudocode: L2 norm per bone, absolute difference, mean over bones.
- The code change instructions are explicit: import SMPLX_SKELETON, define BODY_BONES via filtering, define bone_length_loss, add l_bone to the loss line and logging/del statements.
- All config fields listed.

### Mathematical Correctness
- `pred_len = (pred[:, i] - pred[:, j]).norm(dim=-1)` computes L2 norm correctly. Shape (B,).
- `(pred_len - gt_len).abs().mean()` is L1 loss on bone lengths, averaged over batch. Correct.
- `loss / len(bone_edges)` averages over the 21 bones. The total bone loss is thus mean absolute bone-length error in meters.
- lambda_bone=0.1 is conservative. At convergence with ~5 mm bone errors, the bone loss contribution is ~0.0005 m, while pose loss is ~0.01-0.05 m. The 0.1 weight keeps it as a secondary regularizer. This analysis is sound.
- The design correctly uses `BODY_BONES = [(a, b) for a, b in SMPLX_SKELETON if a < 22 and b < 22]` which filters the already-remapped SMPLX_SKELETON. This is correct.

### Architectural Feasibility
- No new model parameters. Bone-length loss is computed from existing outputs. Negligible computational overhead (21 vector subtractions and norms per batch).
- No VRAM impact.

### Constraint Adherence
- LLRD schedule preserved. All fixed hyperparameters unchanged.
- infra.py not modified (SMPLX_SKELETON is imported, not redefined).
- Model architecture and transforms unchanged.
- Evaluation uses standard unweighted MPJPE.

### Minor Note
- lambda_bone is hardcoded in the loss line rather than being a config field. The design acknowledges this explicitly. The idea.md does not mandate it as a config field, so this is acceptable for a single-value experiment. However, the Builder should ensure the hardcoded 0.1 is clearly commented.

## Verdict: APPROVED
