# Code Review Log — idea017

## idea017/design001 — Delta-Input Channel Stacking (8-channel, single backbone pass)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea017/design001/code_review.md  
patch_embed widened to 8-channel; new 4 channels initialized as mean of original 4 channels. DepthBucketPE unchanged (still reads index 3 = centre-frame depth). TemporalBedlamDataset subclass fetches past frame with same crop bbox. 2-epoch test passed (879mm→800mm, pelvis 1923mm→458mm).

---

## idea017/design002 — Cross-Frame Memory Attention (2-frame, both trainable, gradient checkpointing)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea017/design002/code_review.md  
Both backbone passes gradient-checkpointed (use_reentrant=False). Memory = cat([mem_prev, mem_t], dim=1) → (B, 1920, 384). Single-frame validation fallback. No OOM in 2-epoch test (epoch time 73s confirms double backbone). val_mpjpe_body 1409mm→863mm, pelvis 839mm→401mm. Strong convergence.

---

## idea017/design003 — Cross-Frame Memory Attention (2-frame, past frozen no_grad, centre trainable)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea017/design003/code_review.md  
Past frame: no_grad + detach. Centre frame: full gradient with LLRD. Memory cat same as design002. No checkpointing needed. 2-epoch test passed (epoch 41s, 1385mm→974mm, pelvis 845mm→482mm).

---

## idea017/design004 — Three-Frame Symmetric Temporal Fusion (t-5, t, t+5; past/future frozen)
**Date:** 2026-04-11  
**Verdict:** APPROVED  
**Path:** runs/idea017/design004/code_review.md  
Past + future both frozen (no_grad+detach). Memory = cat([prev, t, next]) → (B, 2880, 384). Boundary clamped correctly (past at 0, future at n_frames-1). Three-frame dataset correctly implemented. Minor: mems[] indexing slightly non-obvious but correct. 2-epoch test passed (epoch 50s, 1488mm→1173mm, pelvis 916mm→390mm — largest pelvis improvement of idea017).
