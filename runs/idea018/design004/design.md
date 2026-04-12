# idea018 / design004 — EMA (decay=0.999) + Last-Epoch Polish Pass

## Starting Point
`runs/idea014/design003/code/` — the current SOTA triple-combo baseline (LLRD γ=0.90, unfreeze=5, depth PE, wide head, wd=0.3, 20 epochs).

## Problem
Pure EMA averaging (design001) tracks a running mean of weights and typically lands close to a flat basin, but can end slightly off the true loss minimum on the exact data distribution. A short flat-LR "polish" pass starting from the EMA weights allows the averaged parameters to settle at the nearest loss-surface minimum without destroying the flatness benefit.

## Proposed Solution
Train 20 epochs with EMA (decay=0.999) exactly as in design001. At the end of epoch 20:
1. Load EMA weights into the live model.
2. Run **1 extra epoch** (epoch 21) with a constant flat LR = 1e-6 for **all** parameter groups.
3. Report val metrics on the polished live model at the end of epoch 21.

The extra epoch costs ≈5% more compute and uses no new parameters.

## Configuration (config.py fields to add)
```python
# EMA + polish settings
ema_decay         = 0.999   # same as design001
polish_lr         = 1e-6    # flat LR for the polish epoch
polish_epochs     = 1       # number of polish epochs after EMA training
```
All other config values identical to `runs/idea014/design003/code/config.py`, except:
```python
output_dir = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea018/design004"
```
Note: `epochs = 20` remains (the polish pass is handled separately in a post-training block, not by extending the epoch loop).

## train.py Changes (diff against runs/idea014/design003/code/train.py)

### 1. New import at top
```python
import copy
```

### 2. After model is built, initialise EMA model (identical to design001)
```python
# ── EMA model ──────────────────────────────────────────────────────────────
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)
_ema_decay = args.ema_decay

def _update_ema():
    with torch.no_grad():
        for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(_ema_decay).add_(live_p.data, alpha=1.0 - _ema_decay)
```

### 3. In `train_one_epoch` — add `update_ema` parameter (same as design001)
```python
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args,
                    iter_logger=None, update_ema=None) -> dict:
```
Call inside accumulation boundary:
```python
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if update_ema is not None:
                update_ema()
```

### 4. Main 20-epoch loop — validate on EMA model (same as design001)
```python
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args,
                                  iter_logger=iter_logger, update_ema=_update_ema)
        ...
            ema_model.eval()
            val_m = validate(ema_model, val_loader, device, args)
```

### 5. After the 20-epoch loop — polish pass
After `logger.close()` / `iter_logger.close()` calls (but before the final print), add:

```python
    # ── Polish pass: load EMA weights into live model, train 1 epoch at flat LR ──
    print("\n*** Polish pass: loading EMA weights into live model ***")
    model.load_state_dict(ema_model.state_dict())
    model.train()

    # Re-enable all parameters (EMA model had requires_grad=False)
    for p in model.parameters():
        p.requires_grad_(True)

    # Build a flat-LR optimizer (single group, all params)
    polish_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.polish_lr, weight_decay=args.weight_decay
    )
    polish_scaler = GradScaler("cuda", enabled=args.amp)

    # Re-open iter_logger for the polish epoch
    iter_logger_polish = IterLogger(str(out_dir / "iter_metrics.csv"))

    print(f"  Polish epoch: lr={args.polish_lr:.1e}, weight_decay={args.weight_decay}")
    t0 = time.time()
    polish_train_m = train_one_epoch(
        model, train_loader, polish_optimizer, polish_scaler, device,
        args.epochs + 1, args, iter_logger=iter_logger_polish,
        update_ema=None,   # no EMA during polish
    )
    iter_logger_polish.close()

    # Validate polished model
    polish_val_m = validate(model, val_loader, device, args)
    torch.cuda.empty_cache()
    epoch_time = time.time() - t0

    print(f"  [Polish] → loss={polish_train_m['train_loss']:.4f}"
          f"  body={polish_train_m['train_mpjpe_body']:.1f}mm"
          f"  | val body={polish_val_m['val_mpjpe_body']:.1f}mm"
          f"  pelvis={polish_val_m['val_pelvis_err']:.1f}mm"
          f"  w={polish_val_m['val_mpjpe_weighted']:.1f}mm"
          f"  ({epoch_time:.0f}s)")

    # Log polish epoch to metrics.csv
    logger_polish = Logger(str(out_dir / "metrics.csv"))  # re-opens in append mode
    logger_polish.log({"epoch": args.epochs + 1, "lr_backbone": args.polish_lr,
                       "lr_head": args.polish_lr,
                       **polish_train_m, **polish_val_m,
                       "epoch_time": epoch_time})
    logger_polish.close()

    # Save final polished checkpoint
    save_checkpoint(
        {"epoch": args.epochs, "model": model.state_dict(),
         "ema_pre_polish": ema_model.state_dict(),
         "optimizer": polish_optimizer.state_dict(),
         "scaler": polish_scaler.state_dict(),
         "best_mpjpe": polish_val_m["val_mpjpe_weighted"]},
        str(out_dir / "polished.pth"),
    )
    final_val_body = polish_val_m["val_mpjpe_body"]
    print(f"Training + polish complete. Final val body = {final_val_body:.1f}mm")
    print(final_val_body)
```

**Important:** The existing `logger.close()` and `iter_logger.close()` should be called before the polish block. Remove the existing `print(final_val_body)` at the very end (it is replaced by the final print in the polish block above).

### 6. Logger append mode
`Logger` in infra.py must support append mode to re-open `metrics.csv` without overwriting. Check infra.py; if it always writes headers, open a separate logger with `mode='a'` or write the polish row manually with `csv.writer`. A safe alternative:
```python
import csv
with open(str(out_dir / "metrics.csv"), "a", newline="") as _f:
    _writer = csv.DictWriter(_f, fieldnames=list(polish_val_m.keys()) + list(polish_train_m.keys()) + ["epoch", "lr_backbone", "lr_head", "epoch_time"])
    _writer.writerow({"epoch": args.epochs + 1, "lr_backbone": args.polish_lr,
                      "lr_head": args.polish_lr, **polish_train_m, **polish_val_m,
                      "epoch_time": epoch_time})
```

## Where EMA Update Happens
- After each effective optimizer step (every `accum_steps` micro-batches) during epochs 0–19.
- No EMA update during the polish epoch (pass `update_ema=None`).

## Which State Dict is Reported
- Epochs 0–19: `validate(ema_model, ...)` → EMA-model val metrics in CSV.
- Epoch 21 (polish): `validate(model, ...)` after loading EMA weights → polished live-model val metrics appended to CSV.
- **The canonical final result is the polished val metric from epoch 21.**
- `polished.pth` is the final checkpoint; `ema_pre_polish` key stores the EMA state before polishing.

## Sanity Check
- Set `polish_epochs = 0`: skip polish block entirely → results identical to design001 (pure EMA).
- Set `ema_decay = 0`: EMA always equals live model → polish pass starts from final live-model weights with 1e-6 LR → marginal change only (sanity-checks polish path separately).
- Set `polish_lr = 0` (or effectively 0): polish epoch does no learning → polished weights == EMA weights → should recover design001 result.

## Memory Budget
- Same as design001 for epochs 0–19: one extra EMA copy ≈ 345 MB fp32.
- During polish: EMA weights loaded into live model; `ema_model` can be freed after `model.load_state_dict(ema_model.state_dict())` to reclaim memory: `del ema_model; torch.cuda.empty_cache()`.
- Polish optimizer holds Adam state: ≈2× param count × 4 bytes ≈ 690 MB. Within budget.
