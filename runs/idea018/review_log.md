# Review Log — idea018: Weight Averaging (EMA and SWA)

---

## design001 — EMA of Full Model Weights (decay=0.999)
**Date:** 2026-04-11
**Verdict:** APPROVED

EMA update formula correct; fires after each effective optimizer step; validates on EMA model; both live and EMA state_dicts saved; ema_decay=0.999; memory budget verified (≈345 MB extra); sanity checks valid. No issues.

---

## design002 — EMA with Warmup (decay ramps to 0.9995)
**Date:** 2026-04-11
**Verdict:** APPROVED

Warmup formula `decay_t = min(0.9995, (1+step)/(10+step))` mathematically sound; saturates at target by ~step 2000 (≈3 epochs); dict-based step counter is clean and correct; validation on EMA model; checkpoint saves ema_step for resumability. `ema_warmup` config flag is effectively documentation (always active) — harmless. No issues.

---

## design003 — Stochastic Weight Averaging over Last 5 Epochs
**Date:** 2026-04-11
**Verdict:** APPROVED

SWA triggers at epoch 15 (0-indexed), covering epochs 15–19. Constant LR computed as 0.5 × cosine scale at epoch 15 ≈ 0.051 × initial_lr — math verified. Running-average formula `(swa*n + live)/(n+1)` is arithmetically correct. SWA model allocated only at epoch 15 (no wasted RAM for first 15 epochs). Validation switches to SWA model at epoch 15. No BN recalibration needed (LayerNorm only). No issues.

---

## design004 — EMA (decay=0.999) + Last-Epoch Polish Pass
**Date:** 2026-04-11
**Verdict:** APPROVED (with flagged implementation risk)

EMA phase identical to design001 — correct. Polish pass loads EMA weights into live model, re-enables gradients, creates fresh AdamW at polish_lr=1e-6. metrics.csv append workaround provided. **Risk flagged:** `iter_metrics.csv` append mode not addressed — if IterLogger writes headers unconditionally, reopening the file corrupts existing 20-epoch data. Builder must check IterLogger.__init__ and apply same append pattern. Design otherwise sound.

---
