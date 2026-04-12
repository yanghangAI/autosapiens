
---

## idea018/design001 — EMA decay=0.999
**Verdict:** APPROVED

EMA init/update/validation all correct. decay=0.999 in config. Validation exclusively on EMA model. Checkpoint saves live + EMA state_dicts. LLRD/depth PE/wide head/wd=0.3 baseline fully preserved. 2-epoch smoke test passed (4.17GB/8.51GB GPU).

---

## idea018/design002 — EMA with warmup (target=0.9995)
**Verdict:** APPROVED

Warmup formula `min(target, (1+s)/(10+s))` implemented exactly with dict-based step counter. `ema_step` saved in checkpoint. Validation on EMA model. `ema_warmup` config field present but informational only (always applied) — harmless. 2-epoch smoke test passed.

---

## idea018/design003 — SWA over last 5 epochs
**Verdict:** APPROVED

SWA start at epoch 15 (never activates in 2-epoch smoke test, as expected). Constant LR computed once at SWA entry and locked. Running average formula correct. SWA model initialized lazily at first SWA epoch. Validation on SWA model during SWA phase. 2-epoch smoke test passed (3.00GB GPU — no SWA copy allocated yet).

---

## idea018/design004 — EMA + Polish pass
**Verdict:** APPROVED

EMA identical to design001. Polish pass: delete optimizer/scaler, load EMA→live, re-enable grads, delete EMA model, build flat AdamW at 1e-6, train 1 epoch, validate. Uses `iter_metrics_polish.csv` to avoid header collision. Appends to metrics.csv via csv.DictWriter. 3-stage smoke test passed (main 2 epochs + 1 polish epoch, polished.pth created).
