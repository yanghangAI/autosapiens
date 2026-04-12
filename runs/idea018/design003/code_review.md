# Code Review — idea018 / design003

**Design_ID:** idea018/design003  
**Verdict:** APPROVED

---

## Checklist

### 1. SWA state initialization
- `_swa_n = 0`, `_swa_model = None`, `_swa_lr_scale = None` initialized before the epoch loop. Matches design.

### 2. SWA phase detection
- `_in_swa = (epoch >= args.swa_start_epoch)` — correct.

### 3. Constant LR override during SWA phase
- On entry to SWA (`_swa_lr_scale is None`), the code captures the cosine scale at that epoch multiplied by `args.swa_lr_factor`. Then all optimizer param groups are overridden with `initial_lr * _swa_lr_scale`. This matches the design exactly.
- `_swa_lr_scale` is set once and reused in subsequent SWA epochs, correctly making the LR constant.

### 4. SWA weight accumulation
- First SWA epoch: deepcopy of live model, `requires_grad_(False)`, then `load_state_dict` → `_swa_n = 1`. Correct.
- Subsequent epochs: running average `swa_p * n/(n+1) + live_p * 1/(n+1)`, then `_swa_n += 1`. Formula is the standard uniform running average and matches the design's `swa_param = (swa_param * n + param) / (n + 1)` exactly.

### 5. Validation protocol
- During non-SWA phase: `validate(model, ...)`.
- During SWA phase: `validate(_swa_model, ...)` (when not None). Matches design. `model.train()` called after to restore live model to train mode.

### 6. Checkpoint
- Saves `{"swa": _swa_model.state_dict() if _swa_model is not None else None, "swa_n": _swa_n, ...}`. Matches design.

### 7. config.py
- `swa_start_epoch = 15`, `swa_lr_factor = 0.5`. Output dir set to design003. All baseline fields preserved.

### 8. Baseline fidelity
- LLRD, unfreeze, depth PE, head_hidden=384, weight_decay=0.3 all unchanged. No architecture or loss changes. Correct.

### 9. No hardcoded experiment params
- All SWA params read from `args`. Confirmed.

### 10. Smoke test
- 2-epoch test passed. Since `swa_start_epoch=15` and test only runs 2 epochs, SWA phase never activates — validation is on live model. This is correct behavior (the 2-epoch test verifies the non-SWA path runs cleanly). val body=1166.4mm, val weighted=897.9mm. No errors or OOM. GPU: 3.00GB alloc / 7.33GB reserved (no SWA copy allocated, as expected for epochs 0-1).

### 11. Issues
- None. The SWA accumulation logic, constant-LR override, and validation protocol all exactly match the design spec.
