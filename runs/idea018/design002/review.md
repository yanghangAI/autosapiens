# Review: idea018 / design002 — EMA with Warmup (decay ramps to 0.9995)

**Verdict: APPROVED**

## Summary
Well-specified EMA warmup variant. The warmup formula is mathematically sound, the step counter management is clean, and the validation/checkpoint protocol is correct.

## Checklist

### Correctness
- [x] Warmup formula: `decay_t = min(target_decay, (1 + step) / (10 + step))`.
  - step=0 → 1/10 = 0.10 (nearly no memory, EMA ≈ live weights). Correct.
  - step=100 → 101/110 ≈ 0.918.
  - step=1000 → 1001/1010 ≈ 0.9911; saturates at target_decay=0.9995 by ~step 2000. Correct.
- [x] Step counter uses dict `_ema_state = {"step": 0}` to avoid `global` — the recommended cleaner pattern is provided.
- [x] Update fires after each effective optimizer step (every `accum_steps` micro-batches) — consistent with design001.
- [x] `ema_model` initialised identically to design001 (`deepcopy`, `.eval()`, `requires_grad_(False)`).
- [x] Validation exclusively on `ema_model` — correct.
- [x] Checkpoint saves `model`, `ema`, and `ema_step` state — correct; `ema_step` enables proper resumption.

### Configuration Completeness
- [x] `ema_decay = 0.9995` (target) added.
- [x] `ema_warmup = True` added.
- [x] `output_dir` updated to `runs/idea018/design002`.
- [x] All other fields inherited from idea014/design003.

### Architecture / Loss / Optimizer
- [x] No architecture, loss, or optimizer changes.

### Memory Budget
- [x] Same as design001: one extra copy ≈ 345 MB fp32. Within 11 GB.

### 20-Epoch Proxy Limit
- [x] Runs exactly 20 epochs.

### Sanity Checks
- [x] `ema_target_decay = 0.0` → `decay_t = 0` always → EMA always equals live model → recovers baseline. Valid.

## Minor Observations
- The design provides both a `global _ema_step` version and the cleaner `_ema_state` dict pattern. The Builder should use the dict pattern; both are functionally equivalent.
- `ema_warmup = True` in config is declared but not actually used in the update logic — the warmup is always applied in design002. The flag is effectively documentation. Harmless.

## Decision
**APPROVED.** No changes required.
