# Review: idea015/design004 — Two-Pass Two-Decoder Refinement (Independent 2-Layer Refine Decoder)

**Design_ID:** idea015/design004  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Two-pass refinement using an independent 2-layer TransformerDecoder for the refinement pass (non-shared weights). Pass 1 uses the existing 4-layer decoder → J1. A shared-weight refine_mlp projects J1 → R, added to out1 to form queries2. An independent 2-layer `refine_decoder` (d=384, 8 heads, FFN=1536) processes queries2 over the same memory → out2 → J2 via `joints_out2`. Deep supervision: `0.5*L(J1) + 1.0*L(J2)`. New params: ~4.87M (refine_decoder dominates).

---

## Evaluation

### 1. Fidelity to idea.md Axis A4

- **Independent 2-layer refinement decoder:** Correct. `nn.TransformerDecoder(refine_layer, num_layers=2)` with separate weights.
- **Same memory:** Both decoders cross-attend the same projected memory — correct.
- **Refinement MLP:** `Linear(3,384)->GELU->Linear(384,384)` — matches spec.
- **Deep supervision weights:** `0.5*L(J1) + 1.0*L(J2)` — exactly matches spec.
- **Optimizer group:** refine_decoder, refine_mlp, joints_out2 go into head_params (LR=1e-4, WD=0.3, no LLRD) — as required by design constraint. Correctly noted that these are attributes of Pose3DHead and so automatically included.

### 2. Hyperparameter Completeness

All required hyperparameters listed. New config fields: `refine_passes=2`, `refine_decoder_layers=2`, `refine_loss_weight=0.5`. Complete.

### 3. Mathematical Correctness

- refine_decoder spec: d=384, nhead=8, dim_feedforward=384*4=1536, dropout=0.1, batch_first=True, norm_first=True — all matching the baseline decoder style.
- Parameter count: ~4.87M — detailed calculation provided. Designer notes the idea.md said ~3M but actual is ~4.87M, which is still well within budget. This is transparent and correct.
- Loss formula: `(0.5*l_pose1 + 1.0*l_pose2 + lambda_depth*l_dep + lambda_uv*l_uv) / accum_steps` — correct.

### 4. Architecture Feasibility

- ~4.87M new params at batch=4: well within 11 GB.
- Activation memory increase: ~100 MB (half of the 4-layer decoder). Budget is fine.
- refine_decoder and refine_mlp inside Pose3DHead — auto-included in head optimizer group without changes to LLRD builder code.

### 5. Builder Instructions

Forward pass pseudocode is clear. Optimizer group assignment is explicitly addressed and correctly reasoned. Return dict keys specified. Loss computation exact.

### 6. Constraint Adherence

- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: fixed.
- Continuous sqrt depth PE, wide head (384, 8, 4): unchanged.
- infra.py, transforms: not modified.
- New decoder in head_params group: correct.

---

## Issues Found

None. The parameter count discrepancy from idea.md (3M stated vs. 4.87M actual) is explicitly flagged and is a non-issue for feasibility. The design is otherwise complete and accurate.

---

## Verdict: APPROVED
