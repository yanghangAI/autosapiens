# Code Review — idea018 / design001

**Design_ID:** idea018/design001  
**Verdict:** APPROVED

---

## Checklist

### 1. EMA initialization
- `copy.deepcopy(model).eval()` used correctly. All EMA params set to `requires_grad_(False)`. Matches design spec exactly.

### 2. EMA update formula
- `ema_p.data.mul_(_ema_decay).add_(live_p.data, alpha=1.0 - _ema_decay)` — correct, matches design spec.
- Update wrapped in `torch.no_grad()` — correct.

### 3. EMA update placement
- Called inside `train_one_epoch` only when `(i + 1) % accum_steps == 0` (i.e. after `optimizer.step()`). Exactly as specified: outside the micro-batch gradient accumulation boundary, one update per effective optimizer step.

### 4. Validation protocol
- `validate(ema_model, val_loader, device, args)` used for all epoch-end validations. Live model is never validated. Matches design spec.

### 5. Checkpoint
- Saves `{"model": model.state_dict(), "ema": ema_model.state_dict(), ...}`. Matches design.

### 6. config.py
- `ema_decay = 0.999` present and correctly used via `args.ema_decay`. Output dir set to design001. All baseline config fields identical to idea014/design003.

### 7. Baseline fidelity
- LLRD (gamma=0.90, unfreeze_epoch=5), depth PE, head_hidden=384, weight_decay=0.3 all preserved unchanged. No architecture or loss changes. Correct.

### 8. No hardcoded experiment params in train.py
- All configurable parameters read from `args`. Confirmed.

### 9. Smoke test
- 2-epoch test passed cleanly. val body=590.6mm, val weighted=1063.2mm. EMA model validating correctly (values slightly differ from epoch 1 due to EMA smoothing from epoch-0 init). No errors or OOM. GPU: 4.17GB alloc / 8.51GB reserved at epoch 2 — within budget.

### 10. Issues
- None. Implementation is clean and matches the design precisely.
