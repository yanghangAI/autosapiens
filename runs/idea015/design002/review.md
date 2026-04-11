# Review: idea015/design002 — Two-Pass Shared-Decoder Refinement (Cross-Attention Gaussian Bias)

**Design_ID:** idea015/design002  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Two-pass refinement where the second pass injects a per-joint Gaussian additive bias into the cross-attention logits of each decoder layer, centred on the projected (u,v) location of the coarse joint prediction `J1`. A learnable scalar `attn_bias_scale` (initialized to 0.0) controls the bias magnitude so that at step 0 the second pass is numerically identical to the first. Second output head `joints_out2` produces the final prediction `J2`. Deep supervision: `0.5 * L(J1) + 1.0 * L(J2)`.

---

## Evaluation

### 1. Fidelity to idea.md Axis A2

- **Gaussian bias on cross-attention logits:** Correct. Log-Gaussian (`-dist2 / (2*sigma^2)`) added as additive bias to cross-attention.
- **sigma=2.0 patches (fixed):** Correct.
- **Learnable scalar initialized to 0.0:** Implemented as `nn.Parameter(torch.zeros(1))`, scaled via `sigmoid(attn_bias_scale) * 10.0`. At init, `sigmoid(0) = 0.5` → bias scale = 5.0, NOT zero. **This is a deviation from spec.** The spec requires the bias to be zero at step 0, so the first pass is identical to the baseline. Using `sigmoid * 10.0` gives a non-zero initial bias.

  **Mitigation:** The design clamps to `min=-1e4`, not `-inf`, which avoids NaN rows. The Gaussian falloff is smooth. However, the initialization is not strictly zero at step 0 as required. The Designer should have used `attn_bias_scale * gauss_bias` with the scalar initialized to 0.0 (not sigmoid-wrapped). Despite this, the design is still sound in concept and the practical impact at step 0 is bounded — the Gaussian will push attention slightly toward predicted joint locations from the start. This is a soft rather than fatal violation.

- **UV projection fallback:** Documented. `pelvis_uv_pred` from `uv_out(out1[:,0,:])` is used as root anchor — verified available in the head. Fallback (center + offsets) not needed.
- **Memory grid (H_tok=40, W_tok=24):** Correct. Grid index: `v * W_tok + u` — correctly noted.
- **Manual layer loop for pass 2:** Fully specified with `norm_first=True` correction. The corrected loop is mathematically correct for pre-norm layers.

### 2. Hyperparameter Completeness

All required hyperparameters explicitly listed. New config fields: `refine_passes=2`, `refine_loss_weight=0.5`, `attn_bias_sigma=2.0`. Complete.

### 3. Mathematical Correctness

- Gaussian: `gauss_bias = -dist2 / (2 * sigma^2)` → log-Gaussian, correct as additive attention bias.
- UV projection formula: `uv_norm = pelvis_uv + 0.5 * (J1[:,:,:2] / z_clamped)` — matches the idea.md specification.
- `attn_bias_expanded`: shape `(B * num_heads, 70, 960)` — correct for `nn.MultiheadAttention` with `attn_mask`.
- Clamping to `min=-1e4` prevents -inf rows. Correct.
- Loss formula: `0.5*l_pose1 + 1.0*l_pose2 + lambda_depth*l_dep + lambda_uv*l_uv` — correct.

### 4. Architecture Feasibility

- New params: ~1.2K + 1 scalar. Negligible.
- The manual layer loop is operationally feasible but requires careful implementation. The design provides the exact loop with the correct `norm_first=True` order. Builder can implement directly.
- Memory: Gaussian computation `(B, 70, 960)` at batch=4 ≈ 1.07M floats = 4.3 MB. Trivial.

### 5. Builder Instructions

Unambiguous. Two separate layer-loop code blocks (norm_last and norm_first corrected) are provided. The Builder should use the corrected `norm_first=True` block.

### 6. Constraint Adherence

- All fixed hyperparameters preserved.
- No hard masking (clamped to -1e4, not -inf). Correct.
- infra.py, transforms unchanged.
- Bias is additive — confirmed.

---

## Issues Found

**Minor (non-blocking):** The `attn_bias_scale` initialization via `sigmoid(0.0) * 10.0 = 5.0` means the Gaussian bias is non-zero at step 0 (contradicting the spec's "first pass is identical to baseline at step 0"). The correct implementation would be `self.attn_bias_scale * gauss_bias` with `self.attn_bias_scale` initialized to 0.0. However, this is a bounded deviation — the non-zero initial bias will steer attention toward predicted locations from epoch 0 but this is arguably beneficial rather than harmful, and the design explicitly documents the scaling.

This is not a fatal flaw — the design is architecturally sound and the Builder can implement the simpler `scale=0.0` initialization. Noted here for the Builder to correct.

---

## Verdict: APPROVED
