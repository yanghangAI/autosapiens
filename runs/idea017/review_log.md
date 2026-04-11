# idea017 Design Review Log

---

## idea017/design001 — Delta-Input Channel Stacking (8-channel, single backbone pass)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** 8-channel input [RGB_t, D_t, RGB_{t-5}, D_{t-5}]. patch_embed widened to Conv2d(8,768); new 4 channels init as mean of original 4. DepthBucketPE reads index 3 (correct). Single backbone pass, no memory concern. Minor: normalization of rgb_prev not explicitly stated — Builder should use identical normalisation to centre frame.

---

## idea017/design002 — Cross-Frame Memory Attention (2-frame, both trainable, gradient checkpointing)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Both backbone passes gradient-checkpointed. Concatenated memory (B,1920,384). Shared backbone, LLRD unchanged. OOM dry-run required (estimated ~9-10 GB). Fallback to batch=2/accum=16 specified. Validation uses single-frame fallback. No new params.

---

## idea017/design003 — Cross-Frame Memory Attention (2-frame, past frozen no_grad, centre trainable)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Past-frame backbone in no_grad+detach, centre trainable. No checkpointing needed. Estimated ~7-8 GB, comfortable. LLRD unchanged. Concatenated memory (B,1920,384). Validation single-frame fallback. Clean design, no issues.

---

## idea017/design004 — Three-Frame Symmetric Temporal Fusion (t-5, t, t+5)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Summary:** Three frames; past+future frozen (no_grad+detach), centre trainable. Memory (B,2880,384). Correct [prev,t,next] ordering. Estimated ~8-9 GB; dry-run recommended. Fallback to batch=2 specified. Minor clarity issue in mems[] indexing — Builder should comment. Symmetric window correct.
