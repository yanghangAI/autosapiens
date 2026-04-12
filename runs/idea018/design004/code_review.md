# Code Review — idea018 / design004

**Design_ID:** idea018/design004  
**Verdict:** APPROVED

---

## Checklist

### 1. EMA during main 20-epoch loop
- Implementation identical to design001: `copy.deepcopy(model).eval()`, `requires_grad_(False)`, `_ema_decay = args.ema_decay`, update at accumulation boundary with correct formula. Matches design.

### 2. Validation during main loop
- `validate(ema_model, val_loader, device, args)` at each val interval. Matches design spec.

### 3. Polish pass — EMA weight loading
- After `logger.close()` / `iter_logger.close()`, the code: (a) deletes `optimizer` and `scaler` and calls `gc.collect()` + `torch.cuda.empty_cache()` to free VRAM; (b) calls `model.load_state_dict(ema_model.state_dict())` to load EMA weights; (c) re-enables `requires_grad_(True)` for all params; (d) deletes `ema_model` to free the extra 345 MB. This is a correct and memory-efficient implementation.

### 4. Polish optimizer
- `torch.optim.AdamW(model.parameters(), lr=args.polish_lr, weight_decay=args.weight_decay)` — single flat-LR group for all params, exactly as designed. `polish_lr=1e-6` from config.

### 5. Polish epoch logging
- Uses a separate `iter_metrics_polish.csv` file (avoids header collision with the main `iter_metrics.csv`). This is a safe workaround for the IterLogger header issue flagged in the design review.
- `metrics.csv` is appended to via `csv.DictWriter` with `mode='a'` and `extrasaction="ignore"`. The design explicitly provided this pattern as a safe alternative; the Builder implemented it correctly.

### 6. Polish epoch output
- `validate(model, val_loader, device, args)` used for polish val. `polished.pth` checkpoint saved with polish optimizer state. Matches design.

### 7. config.py
- `ema_decay = 0.999`, `polish_lr = 1e-6`, `polish_epochs = 1`. Output dir set to design004. All baseline fields preserved. Note: `polish_epochs` is in config but not dynamically looped over in train.py (there is exactly 1 polish epoch hardcoded). This is acceptable since the design specifies `polish_epochs = 1` and the design spec says to handle the polish as a post-training block, not by extending the epoch loop.

### 8. Baseline fidelity
- LLRD, unfreeze, depth PE, head_hidden=384, weight_decay=0.3 all unchanged in the 20-epoch main loop. No architecture or loss changes.

### 9. No hardcoded experiment params
- EMA decay and polish LR read from `args`. Confirmed.

### 10. Smoke test
- 3-stage test (2 main epochs + 1 polish epoch) passed completely. Main: val body=590.6mm, val weighted=1063.2mm. Polish: val body=556.8mm, val weighted=1024.9mm. GPU memory freed correctly before polish (1.19GB alloc after del ema_model). `polished.pth` and `iter_metrics_polish.csv` both created. No errors or OOM.

### 11. Issues
- `polish_epochs` config field is informational only (unused as a loop count). Non-fatal; design only specifies 1 polish epoch and this is what the code does.
- The design review flagged that `iter_metrics.csv` would get a duplicate header if using IterLogger in append mode. The Builder solved this by writing to `iter_metrics_polish.csv` instead — an elegant and correct workaround.
