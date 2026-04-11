# Review: idea015/design003 — Three-Pass Shared-Decoder Refinement (Deep Supervision 0.25/0.5/1.0)

**Design_ID:** idea015/design003  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Three-pass iterative refinement with shared decoder and query injection. Pass 1 → J1, Pass 2 → J2 (inject J1 via shared refine_mlp), Pass 3 → J3 (inject J2 via same shared refine_mlp). Three separate output heads (joints_out, joints_out2, joints_out3). Final prediction = J3. Deep supervision: `0.25*L(J1) + 0.5*L(J2) + 1.0*L(J3)`.

---

## Evaluation

### 1. Fidelity to idea.md Axis A3

- **Three passes:** Correct. Decoder reused three times with weight sharing.
- **Shared refine_mlp:** Correct. `self.refine_mlp` is a single module used for both J1→J2 and J2→J3 transitions.
- **Query injection:** `queries2 = out1 + R1`, `queries3 = out2 + R2` — consistent with design001 pattern.
- **Three output heads:** `joints_out` (existing), `joints_out2`, `joints_out3` — correct.
- **Deep supervision weights:** `0.25*L(J1) + 0.5*L(J2) + 1.0*L(J3)` — exactly matches spec.
- **Final prediction:** `J3 = out["joints"]` — correct.
- **Pelvis outputs from `out3[:, 0, :]`:** Correct (final pass token).

### 2. Hyperparameter Completeness

All required hyperparameters explicitly listed. New config fields: `refine_passes=3`, `refine_loss_w1=0.25`, `refine_loss_w2=0.5`, `refine_loss_w3=1.0`. Complete.

### 3. Mathematical Correctness

- Loss: `(0.25*l_pose1 + 0.5*l_pose2 + 1.0*l_pose3 + lambda_depth*l_dep + lambda_uv*l_uv) / accum_steps` — correct.
- Smooth L1 (beta=0.05) on BODY_IDX — matches constraint.
- Metrics on `out["joints"]` (= J3) — correct.

### 4. Architecture Feasibility

- New params: ~152K (refine_mlp 150K + joints_out2 1.2K + joints_out3 1.2K). Negligible.
- Three decoder passes: memory estimate of ~150MB extra activation is reasonable and well within 11 GB budget.
- Weight sharing of refine_mlp and decoder keeps param count bounded.

### 5. Iteration Logger

Design mentions adding `loss_pose1`, `loss_pose2`, `loss_pose3` to `iter_logger.log` — good practice, specified correctly.

### 6. Builder Instructions

Forward pass pseudocode is complete and unambiguous. Return dict keys specified. Loss computation exact.

### 7. Constraint Adherence

- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: fixed.
- Continuous sqrt depth PE, wide head (384, 8, 4, FFN=1536): unchanged.
- infra.py, transforms: not modified.
- No modifications to LLRD or optimizer structure beyond auto-inclusion of new head params.

---

## Issues Found

None. This design is clean, complete, and correctly extends design001 to three passes. Weight sharing is correctly identified and specified. The memory estimate is slightly conservative but clearly within budget.

---

## Verdict: APPROVED
