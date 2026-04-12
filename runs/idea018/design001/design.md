# idea018 / design001 — EMA of Full Model Weights (decay=0.999)

## Starting Point
`runs/idea014/design003/code/` — the current SOTA triple-combo baseline (LLRD γ=0.90, unfreeze=5, depth PE, wide head, wd=0.3, 20 epochs).

## Problem
The final-epoch weights land in a sharp local minimum (train_body≈76.8 mm vs val_body≈106.85 mm). Weight-space averaging across the tail of training smooths the loss surface and typically reduces val error by 0.5–2%.

## Proposed Solution
Maintain an exponential moving average (EMA) shadow copy of the live model. After each **effective optimizer step** (i.e. every `accum_steps` micro-batches), update the EMA copy with `ema_param = decay * ema_param + (1 - decay) * live_param`. Validate exclusively on the EMA copy. No architecture, loss, or optimizer changes.

## Configuration (config.py fields to add)
```python
# EMA settings
ema_decay = 0.999   # standard MAE/DINO value; effective window ≈ 1000 steps ≈ 2-3 epochs
```
All other config values are identical to `runs/idea014/design003/code/config.py`, except:
```python
output_dir = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea018/design001"
```

## train.py Changes (diff against runs/idea014/design003/code/train.py)

### 1. New import at top
```python
import copy
```

### 2. After model is built and moved to device (after the `model.to(device)` line), initialise EMA model
```python
# ── EMA model ──────────────────────────────────────────────────────────────
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)
_ema_decay = args.ema_decay
_ema_step = 0  # counts effective optimizer steps

def _update_ema():
    """Update EMA weights after each effective optimizer step."""
    with torch.no_grad():
        for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(_ema_decay).add_(live_p.data, alpha=1.0 - _ema_decay)
```

### 3. In `train_one_epoch` — pass `ema_model` and `update_ema` callable as arguments
Change function signature:
```python
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args,
                    iter_logger=None, update_ema=None) -> dict:
```
Inside the accumulation boundary block (after `optimizer.zero_grad()`), call:
```python
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if update_ema is not None:
                update_ema()
```

### 4. In the main epoch loop, pass the EMA updater
```python
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args,
                                  iter_logger=iter_logger, update_ema=_update_ema)
```

### 5. Validation — run on EMA model instead of live model
Replace:
```python
            val_m = validate(model, val_loader, device, args)
```
with:
```python
            ema_model.eval()
            val_m = validate(ema_model, val_loader, device, args)
```

### 6. Checkpoint — save both live and EMA state_dicts
At the end of the epoch loop (where best checkpoint is saved), wrap `save_checkpoint` calls to include EMA:
```python
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "ema": ema_model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(),
                     "best_mpjpe": best_mpjpe},
                    str(out_dir / "best.pth"),
                )
```
(The `save_checkpoint` helper in infra.py just calls `torch.save(state, path)`, so no infra change needed.)

## Where EMA Update Happens
- **Outside** the gradient-accumulation micro-batch loop, i.e. only when `(i + 1) % accum_steps == 0`.
- This means one EMA update per 8 micro-batches, consistent with one actual weight update.

## Which State Dict is Reported
- All `val_*` metrics in `iter_metrics.csv` and `metrics.csv` come from running `validate(ema_model, ...)`.
- The live model is never validated; its metric is never written to CSV.

## Sanity Check
- If `ema_decay = 0.0`, then `ema_param ← live_param` every step, so EMA == live model → identical val metric to baseline. This recovers the un-averaged result.
- If `ema_decay = 1.0`, EMA never moves from its init (clone of epoch-0 model) → very high val error. Confirms the EMA path is active.

## Memory Budget
- Extra ViT-B copy: ≈86 M params × 4 bytes = ~345 MB fp32. Well within 11 GB budget alongside activations.
- No gradient storage for EMA model (requires_grad=False).
