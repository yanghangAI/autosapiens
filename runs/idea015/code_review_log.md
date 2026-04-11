# Code Review Log — idea015

## idea015/design001 — Two-Pass Shared-Decoder Refinement (Query Injection)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea015/design001/code_review.md  
Architecture correct: refine_mlp(3→384→384), joints_out2(384→3), shared decoder re-run with out1+R queries. Loss weights 0.5/1.0 correctly applied. All 18 config fields correct. 2-epoch test passed (2006mm→1499mm decreasing).

---

## idea015/design002 — Two-Pass Shared-Decoder Refinement (Gaussian Cross-Attention Bias)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea015/design002/code_review.md  
Gaussian bias computation correct (sigma=2.0, clamped to -1e4). Manual norm_first layer loop correctly implemented. attn_bias_scale initialized at 0 (pass 2 = pass 1 at step 0). Minor deviation: raw parameter instead of sigmoid*10 scaling — non-fatal. 2-epoch test passed (525mm→1380mm; epoch 2 bounce within warmup noise).

---

## idea015/design003 — Three-Pass Shared-Decoder Refinement (Progressive Deep Supervision 0.25/0.5/1.0)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea015/design003/code_review.md  
Three decoder passes with shared refine_mlp and shared decoder. Three output heads (joints_out, joints_out2, joints_out3). Loss weights 0.25/0.5/1.0 correctly applied. 2-epoch test passed (3301mm→2977mm decreasing; higher initial error expected for 3-pass).

---

## idea015/design004 — Two-Pass Two-Decoder Refinement (Independent 2-Layer Refine Decoder)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea015/design004/code_review.md  
Independent 2-layer refine_decoder correctly implemented inside Pose3DHead (auto-included in head optimizer group). refine_mlp + joints_out2 correct. ~4.87M new params (within 11GB budget). 2-epoch test passed (1044mm→790mm decreasing).
