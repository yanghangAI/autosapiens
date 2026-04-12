# Review: idea019/design004 — Joint-Group Query Initialization in Refinement Pass (Axis B2)

**Design_ID:** idea019/design004
**Verdict:** APPROVED (with flagged index inconsistency — non-fatal)

## Evaluation

### Completeness
All baseline hyperparameters preserved. New config fields `group_emb_init=0.0` and `num_joint_groups=4` specified (informational). The group assignment lists `_TORSO`, `_ARMS`, `_LEGS` are fully enumerated. Forward pass change is shown precisely. No ambiguity for core functionality.

### Mathematical Correctness
- `group_emb: Embedding(4, 384)` zero-initialized. Adding `group_emb[joint_group_ids]` to `queries2` before pass 2 is a valid query conditioning approach. ✓
- Zero initialization guarantees training starts identical to baseline. ✓
- `group_emb` is a submodule of `model.head`, automatically included in `head_params` (LR=1e-4, weight_decay=0.3). ✓
- 1,536 new parameters (4×384). No OOM risk. ✓

### Body Joint Assignments (Verified Correct)
Cross-checking body joints 0-21 against `_TORSO`, `_ARMS`, `_LEGS`:
- Group 0 (torso): 0, 3, 6, 9, 12, 15 — spine, neck, head ✓
- Group 1 (arms): 13, 14, 16, 17, 18, 19, 20, 21 — shoulders, elbows, wrists, hands ✓
- Group 2 (legs): 1, 2, 4, 5, 7, 8, 10, 11 — hips, knees, ankles, feet ✓
- No body joints are unassigned (all 22 body joints fall in exactly one group) ✓

### Flagged Index Inconsistency (Non-Fatal)
The design places `[23, 24]` in `_TORSO`, using these as **new index space** values. However, the design comments describe these as "eyes (left_eye_smplhf, right_eye_smplhf)". In the remapped index space: new index 22 = original joint 23 (left eye) and new index 23 = original joint 24 (right eye).

Consequence: `_TORSO` contains new index 23 (right eye — correct) and new index 24 (which is original joint 25, the first left-hand finger — misclassified as torso). New index 22 (left eye) is not in `_TORSO` and falls to group 3 (extremity) via the `range(22, num_joints)` loop.

**Impact assessment:** This affects only the group assignment of two non-body joints (new indices 22 and 24) out of 70. Body joints 0-21 are all correctly assigned. The loss (`BODY_IDX=slice(0,22)`) is unaffected. The group embedding is zero-initialized, so the startup behavior is identical to baseline regardless. The misclassification of one eye and one hand finger into wrong groups is a cosmetic inconsistency that does not affect the experiment's scientific validity.

**Builder action:** The Builder should correct `_TORSO` to use `[0, 3, 6, 9, 12, 15, 22, 23]` (new index space) or simply omit eyes from the torso list since they are non-body joints. This does not require a redesign.

### Constraint Adherence
- Zero initialization guarantees identical startup to baseline ✓
- New parameters in `head_params` group ✓
- Loss unchanged ✓
- No changes to `infra.py` or transforms ✓
- All baseline HPs preserved ✓

Core design is sound. The eye-index inconsistency is a minor implementation detail that the Builder should fix when writing the code.
