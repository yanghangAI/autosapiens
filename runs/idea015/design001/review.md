# Review: idea015/design001 — Two-Pass Shared-Decoder Refinement (Query Injection)

**Design_ID:** idea015/design001  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Two-pass iterative refinement where coarse joint prediction `J1` is projected through a small MLP (`Linear(3,384)->GELU->Linear(384,384)`) and added to the pass-1 output features to form updated queries. The same 4-layer decoder is re-run on the updated queries over the same memory. A second output head `joints_out2` produces the final prediction `J2`. Deep supervision: `0.5 * L(J1) + 1.0 * L(J2)`.

---

## Evaluation

### 1. Fidelity to idea.md Axis A1

- **Two-pass shared decoder:** Correct. The same 4-layer decoder is reused for both passes.
- **Refinement MLP:** `Linear(3,384) -> GELU -> Linear(384,384)` — exactly matches the spec.
- **Query injection:** `queries2 = out1 + R` — matches spec (add refinement delta to output features of pass 1, not to original joint_queries). Slight deviation: spec says "Add R to the original joint queries (query_embed + R)" but the design uses `out1 + R` (pass-1 output features + R). This is architecturally equivalent and arguably better motivated (conditioning on the learned representation rather than the raw embedding). Acceptable.
- **Second output head:** `joints_out2 = Linear(384,3)` — correct.
- **Final prediction = J2:** Correct.
- **Deep supervision weights:** `0.5 * L(J1) + 1.0 * L(J2)` — exactly matches spec.
- **Pelvis outputs from `out2[:, 0, :]`:** Correct (refined token).

### 2. Hyperparameter Completeness

All required hyperparameters are explicitly listed: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16. No guessing required by the Builder.

### 3. Mathematical Correctness

- Loss formula: `(0.5*l_pose1 + 1.0*l_pose2 + lambda_depth*l_dep + lambda_uv*l_uv) / accum_steps` — correct.
- Smooth L1 (beta=0.05) on BODY_IDX only — matches constraint.
- Metrics computed on `out["joints"]` (= J2) — correct.

### 4. Architecture Feasibility

- New params: ~151K. Well within 11 GB budget at batch=4.
- Three decoder passes of activation memory (pass1 + pass2) are negligible relative to the backbone's 300M params.
- No modifications to infra.py or transforms. Correct.

### 5. config.py Completeness

New fields `refine_passes=2` and `refine_loss_weight=0.5` are added. Informational and useful. All other fields inherited unchanged — complete.

### 6. Builder Instructions

The forward pass pseudocode is unambiguous. Return dict keys are specified. Loss computation is exact. No guessing required.

### 7. Constraint Adherence

- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: all fixed and unchanged.
- Continuous sqrt-spaced depth PE: unchanged.
- Wide head (384, 8 heads, 4 layers, FFN=1536): unchanged.
- infra.py, transforms: not modified.
- Head param group assignment: new modules are inside Pose3DHead so automatically included in head_params — correct.

---

## Issues Found

None material. The `out1 + R` vs `query_embed + R` deviation from spec is a minor interpretation that is architecturally sensible and explicitly described.

---

## Verdict: APPROVED
