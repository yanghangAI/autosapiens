# Review: idea017/design003 — Cross-Frame Memory Attention (2-frame, past frozen no_grad, centre trainable)

**Design_ID:** idea017/design003  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Same two-frame architecture as design002 but the past-frame backbone forward runs inside `torch.no_grad()` and is explicitly `.detach()`ed. No gradient checkpointing needed (past pass has no gradient; centre pass runs normally). Concatenated memory (1920 tokens). Only centre-frame backbone participates in LLRD. Lower memory than design002 (~7-8 GB estimated).

---

## Evaluation

### 1. Fidelity to idea.md Axis B3

- **Past-frame `no_grad`:** `with torch.no_grad(): feat_prev = self.backbone(x_prev); feat_prev = feat_prev.detach()`. Correct. `detach()` after `no_grad` is belt-and-suspenders but not harmful.
- **Centre-frame: full gradient.** `feat_t = self.backbone(x_t)`. Correct.
- **No checkpointing.** Correct — past pass has no gradient, so no need.
- **LLRD on centre-frame only.** Correct — single backbone module, LLRD groups unchanged.
- **Memory concatenation:** identical to design002. `(B, 1920, 384)`. Correct.
- **Validation: single-frame fallback.** Same as design002 pattern. Correct.

### 2. Memory Management

Frozen pass: activations immediately freed after `.detach()` (no backward graph retained). Centre pass: standard activation storage for backprop. Peak memory ~7-8 GB. Comfortable within 11 GB. No dry-run requirement flagged (but would be prudent — not a design flaw).

### 3. Optimizer / LLRD

LLRD groups: unchanged. Single shared backbone module covers both forward calls. Both calls use the same parameters. Only the centre-frame call contributes gradients (past is `no_grad + detach`). LLRD logic is identical to idea014/design003. Correct.

### 4. Dataloader

Same as design002. Two-frame fetch with same crop bbox. `rgb_prev`, `depth_prev` exposed as batch tensors. Correct.

### 5. Hyperparameter Completeness

New config fields: `temporal_mode="cross_attn_past_frozen"`. All required HPs inherited. `in_channels=4`. Complete.

### 6. Constraint Adherence

- infra.py constants: not modified.
- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: fixed.
- Continuous sqrt depth PE, wide head: unchanged.
- GT: centre frame only. Correct.

---

## Issues Found

None. Design003 is a clean, memory-efficient variant of design002. The frozen-past-frame approach is a well-understood technique (semi-online temporal encoding). The design is complete and unambiguous.

---

## Verdict: APPROVED
