# Review: idea017/design002 — Cross-Frame Memory Attention (2-frame, both trainable, gradient checkpointing)

**Design_ID:** idea017/design002  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

True temporal cross-attention: the dataloader yields two 4-channel frames (`x_t`, `x_prev`). Both backbone forwards are gradient-checkpointed. The two projected feature maps are concatenated along the spatial token axis to form a 1920-token memory. The decoder cross-attends over both time steps. Only centre-frame joints supervised. Shared backbone weights → single LLRD grouping. OOM dry run required.

---

## Evaluation

### 1. Fidelity to idea.md Axis B2

- **Two trainable backbone passes with checkpointing:** `ckpt.checkpoint(self.backbone, x_t, use_reentrant=False)` and `ckpt.checkpoint(self.backbone, x_prev, ...)`. Correct.
- **Concatenated memory:** `memory = cat([proj(feat_prev), proj(feat_t)], dim=1)` → `(B, 1920, 384)`. Correct.
- **Single shared backbone:** LLRD groups unchanged — one backbone module, one set of LLRD groups. Correct.
- **Supervision: centre frame only.** Correct.
- **Validation: single-frame fallback** via `model(x_t, None)` — correctly specified.

### 2. Gradient Checkpointing

`torch.utils.checkpoint.checkpoint` with `use_reentrant=False` is the modern PyTorch pattern. Applied to both backbone forwards — halves activation memory per pass. This is the spec's requirement. Correct.

### 3. Dataloader

Same two-frame fetch as design001 pattern. `rgb_prev`, `depth_prev` via same crop bbox. `x_t = cat([rgb, depth], dim=1)` and `x_prev = cat([rgb_prev, depth_prev], dim=1)`. Correct construction.

### 4. Memory Estimate

Estimated ~9-10 GB at batch=4, accum=8. The design correctly flags the OOM risk and specifies that the Builder must verify with a 1-step dry run. Fallback: reduce batch_size to 2, increase accum_steps to 16 to preserve effective batch size. This is correctly specified.

**Memory reasoning:** Two checkpointed backbone passes — each stores only the input tensor, not intermediate activations (recomputed during backward). This is correct. The doubled decoder memory (1920 tokens vs. 960) adds modest attention memory. The estimate of ~9-10 GB is plausible but requires empirical verification.

### 5. Architecture Feasibility

- No new parameters. Shared `input_proj` for both frames — intentional weight tying. Correct.
- `TransformerDecoder` handles 1920-length memory natively. Correct.
- No changes to decoder architecture.

### 6. Hyperparameter Completeness

New config fields: `temporal_mode="cross_attn_both_trainable"`, `use_grad_ckpt=True`. All required HPs inherited. `in_channels=4` (each frame is 4-channel separately). Complete.

### 7. Constraint Adherence

- No checkpoint bypass of LLRD — LLRD applied to single shared backbone.
- Validation runs single-frame (no temporal context) — design allows this explicitly.
- infra.py constants: not modified.
- BATCH_SIZE=4, epochs=20: fixed (with conditional fallback to batch=2 if OOM).

### 8. Potential Issues

**Memory concern:** The idea.md rates this as "~9-10 GB" — the tight-but-plausible regime. The Builder MUST run the dry run. This is flagged correctly. Not a design rejection reason since the spec explicitly sanctions this dry-run requirement.

**Validation discrepancy:** During validation, `model(x_t, None)` uses single-frame memory (960 tokens). This means validation measures the single-frame capability, not the two-frame capability. This is a limitation of the design (acknowledged in idea.md as a necessary evaluation simplification). Not a flaw — correctly specified.

---

## Issues Found

None fatal. Memory risk is acknowledged and mitigated by dry-run requirement and fallback specification.

---

## Verdict: APPROVED
