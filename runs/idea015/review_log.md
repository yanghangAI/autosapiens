# idea015 Design Review Log

---

## idea015/design001 — Two-Pass Shared-Decoder Refinement (Query Injection)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Two-pass shared decoder with coarse prediction injected into queries via refine_mlp (Linear 3→384→384). Deep supervision 0.5/1.0. ~151K new params. All hyperparameters complete. Minor deviation: `out1 + R` vs. `query_embed + R` is architecturally sound. No fatal issues.

---

## idea015/design002 — Two-Pass Shared-Decoder Refinement (Gaussian Cross-Attention Bias)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Gaussian additive cross-attention bias in pass 2 steers decoder toward coarse joint locations. Manual layer loop with norm_first=True correctly specified. Minor deviation: `sigmoid(0.0)*10.0=5.0` initial bias scale is non-zero (spec requires zero at step 0); Builder should use `attn_bias_scale=0.0` directly. Not fatal. ~1.2K new params.

---

## idea015/design003 — Three-Pass Shared-Decoder Refinement (Deep Supervision 0.25/0.5/1.0)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Three passes, shared decoder and refine_mlp, three output heads. Progressive deep supervision 0.25/0.5/1.0. ~152K new params. Iteration logger additions specified. Clean design, no issues.

---

## idea015/design004 — Two-Pass Two-Decoder Refinement (Independent 2-Layer Refine Decoder)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Independent 2-layer refine_decoder (non-shared weights) specializes for refinement. ~4.87M new params (slight overestimate vs. idea.md's 3M, transparently flagged). Head optimizer group assignment correct. Deep supervision 0.5/1.0. No issues.
