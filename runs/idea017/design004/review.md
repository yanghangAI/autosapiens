# Review: idea017/design004 — Three-Frame Symmetric Temporal Fusion (t-5, t, t+5; past/future frozen, centre trainable)

**Design_ID:** idea017/design004  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Most expressive temporal design. Three frames: `t-5`, `t`, `t+5` (clamped at boundaries). Past and future backbone forwards run `no_grad + detach`. Centre-frame backbone is fully trainable with LLRD. Three projected feature maps concatenated: `memory = cat([proj(feat_prev), proj(feat_t), proj(feat_next)], dim=1)` → `(B, 2880, 384)`. Only centre-frame joints supervised. Symmetric temporal window.

---

## Evaluation

### 1. Fidelity to idea.md Axis B4

- **Three frames:** `past_idx = max(0, frame_idx-1)`, `future_idx = min(n_frames-1, frame_idx+1)`. Correct dataset-index clamping.
- **Past + future frozen:** Both wrapped in `torch.no_grad()` with `.detach()`. Correct.
- **Centre trainable, LLRD:** Single backbone module, unchanged LLRD. Correct.
- **Memory concatenation:** `cat([proj(feat_prev), proj(feat_t), proj(feat_next)], dim=1)` → `(B, 2880, 384)`. Correct.
- **Only centre GT:** Loss computed on `out["joints"][:, BODY_IDX]` vs. centre-frame joints. Correct.
- **Validation: single-frame fallback.** `model(x_t, None, None)`. Correct.

### 2. Memory Ordering

The design specifies memory ordering `[prev, t, next]` — placing centre frame between context frames. However, in the head code:

```python
if len(mems) == 3:
    memory = torch.cat([mems[1], mems[0], mems[2]], dim=1)  # [prev_mem, t_mem, next_mem]
```

Here `mems[0] = mem_t` and `mems[1], mems[2]` are context frames. So `mems[1]` is the first context (prev) and `mems[2]` is the second (next). The result is `[prev, t, next]` — **correct**.

The comment says "context frames bracket the centre" but the actual ordering puts centre (`t`) first in mems, then prev and next. The final `cat([mems[1], mems[0], mems[2]])` correctly reorders to `[prev, t, next]`. The ordering is correct as intended.

### 3. Dataloader

Three-frame fetch: `rgb_prev`, `depth_prev`, `rgb_next`, `depth_next`, all using the same crop bbox as centre frame. Sequence boundary clamping documented. This is the correct extension of design003's approach.

### 4. Memory Estimate

One trainable pass + two `no_grad` passes (activations freed). Peak memory dominated by trainable pass and cross-attention over 2880 tokens (3× the baseline 960). Cross-attention memory scales as O(N_q × N_k) = O(70 × 2880) which is 3× the baseline's 70 × 960. Estimated ~8-9 GB. Within 11 GB. Builder should verify with 1-step dry run. Fallback to batch=2 specified. Correct.

### 5. Hyperparameter Completeness

New config fields: `temporal_mode="three_frame_symmetric"`. All required HPs inherited. `in_channels=4`. Complete.

### 6. Constraint Adherence

- infra.py constants: not modified.
- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: fixed.
- Deep sqrt PE, wide head (384, 8, 4): unchanged.
- Symmetric window (past+future): as specified in idea.md.
- GT = centre frame only. Correct.

### 7. Architecture Feasibility

- No new parameters. Shared `input_proj` for all three frames. Correct.
- The head's `forward` signature `(feat_t, feats_context)` cleanly supports the 3-frame case.
- Decoder handles 2880-token memory natively.

---

## Issues Found

**Minor:** The memory ordering logic in the head is slightly confusing (`mems[1]` is prev, `mems[0]` is centre) because of the order items are appended to `mems`. This is an implementation-level clarity issue — the final ordering `[prev, t, next]` is correct. Builder should add a comment to clarify.

No fatal issues.

---

## Verdict: APPROVED
