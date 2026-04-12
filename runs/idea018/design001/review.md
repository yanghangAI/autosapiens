# Review: idea018 / design001 — EMA of Full Model Weights (decay=0.999)

**Verdict: APPROVED**

## Summary
Clean, minimal EMA implementation that correctly extends the idea014/design003 baseline. All requirements from the idea.md are satisfied.

## Checklist

### Correctness
- [x] EMA update formula: `ema_p = decay * ema_p + (1 - decay) * live_p` — correct.
- [x] Update fires after each **effective optimizer step** (every `accum_steps` micro-batches), not every micro-batch — correct placement.
- [x] `ema_model` is a `copy.deepcopy` of the live model, `.eval()`, all params `requires_grad_(False)` — correct initialisation.
- [x] Validation exclusively runs on `ema_model` — matches idea.md spec.
- [x] Checkpoint saves both `model` and `ema` state_dicts — correct.

### Configuration Completeness
- [x] `ema_decay = 0.999` added to config.py.
- [x] `output_dir` updated to `runs/idea018/design001`.
- [x] All other fields inherited from idea014/design003 unchanged.

### Architecture / Loss / Optimizer
- [x] No architecture changes. No new parameters. No loss changes. LLRD, depth PE, wide head, wd=0.3 preserved.

### Memory Budget
- [x] Extra EMA copy ≈ 86M params × 4 bytes ≈ 345 MB fp32. Well within 11 GB on 1080ti.
- [x] EMA model holds no gradients (`requires_grad_(False)`).

### 20-Epoch Proxy Limit
- [x] Runs exactly 20 epochs. No structural extension.

### Sanity Checks
- [x] `ema_decay=0.0` → EMA always equals live model → recovers baseline val metric. Valid.
- [x] `ema_decay=1.0` → EMA never moves from init → very high val error → confirms EMA path is live.

## Minor Observations
- `_update_ema` is defined as a closure capturing `model`, `ema_model`, and `_ema_decay` from the enclosing `main()` scope. This is standard Python closure behaviour and works correctly.
- `_ema_step` is initialised but never used in the update logic (the EMA formula is fixed decay, not step-dependent). This is harmless dead state.

## Decision
**APPROVED.** No changes required.
