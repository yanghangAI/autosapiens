# Code Review — idea018 / design002

**Design_ID:** idea018/design002  
**Verdict:** APPROVED

---

## Checklist

### 1. EMA initialization
- `copy.deepcopy(model).eval()` used correctly. All EMA params set to `requires_grad_(False)`. Matches design.

### 2. EMA warmup formula
- `decay_t = min(_ema_target_decay, (1.0 + s) / (10.0 + s))` — exact match to design spec.
- Dict-based step counter `_ema_state = {"step": 0}` used instead of `global`, which is the recommended cleaner closure pattern from the design. Counter increments correctly after each call.

### 3. EMA update formula
- `ema_p.data.mul_(decay_t).add_(live_p.data, alpha=1.0 - decay_t)` — correct, matches design.

### 4. EMA update placement
- Called inside `train_one_epoch` at the accumulation boundary `(i + 1) % accum_steps == 0`. Correct.

### 5. Validation protocol
- `validate(ema_model, val_loader, device, args)` for all epoch-end validation. Correct.

### 6. Checkpoint
- Saves `{"ema": ema_model.state_dict(), "ema_step": _ema_state["step"], ...}`. Matches design spec exactly including `ema_step`.

### 7. config.py
- `ema_decay = 0.9995` (target), `ema_warmup = True`. Output dir set to design002. All baseline fields preserved. Note: `ema_warmup` is present in config but not read in train.py (warmup is always applied); this is harmless and consistent with the design which uses the warmup unconditionally.

### 8. Baseline fidelity
- LLRD, depth PE, head_hidden=384, weight_decay=0.3 all preserved. No architecture or loss changes.

### 9. No hardcoded experiment params
- All parameters read from `args`. Confirmed.

### 10. Smoke test
- 2-epoch test passed. val body=793.3mm, val weighted=641.8mm at epoch 2. EMA warmup working correctly: at step 0, decay_t ≈ 0.091 meaning EMA closely tracks initial weights (hence the high body error at epoch 1 val — EMA copy is essentially the init model). By epoch 2 (ema_step=2), decay has ramped and EMA begins tracking a meaningful average. No errors or OOM.

### 11. Issues
- Minor: `ema_warmup = True` in config is never read — warmup is always unconditionally applied. This is acceptable because the design does not specify a code path to disable it; the field is informational only.
