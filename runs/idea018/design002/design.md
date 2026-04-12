# idea018 / design002 — EMA with Warmup (decay ramps from 0 to 0.9995)

## Starting Point
`runs/idea014/design003/code/` — the current SOTA triple-combo baseline (LLRD γ=0.90, unfreeze=5, depth PE, wide head, wd=0.3, 20 epochs).

## Problem
Standard fixed-decay EMA (design001) can be degraded by the noisy early training steps being baked into the average. A warmup schedule that starts the effective decay near 0 and ramps up to the target prevents the very noisy initial weight snapshots from polluting the long-run average.

## Proposed Solution
Use a momentum-style EMA warmup formula:
```
decay_t = min(target_decay, (1 + step) / (10 + step))
```
where `step` is the count of effective optimizer steps. At step=0: decay=0.091 (nearly no memory). At step=100: decay=0.917. At step=990: decay=0.9990. At step ≥ 2000: saturates at `target_decay=0.9995`. Validation is run on the EMA model exactly as in design001.

## Configuration (config.py fields to add)
```python
# EMA settings
ema_decay        = 0.9995   # target; warmup ramps up to this
ema_warmup       = True     # enable warmup schedule
```
All other config values identical to `runs/idea014/design003/code/config.py`, except:
```python
output_dir = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea018/design002"
```

## train.py Changes (diff against runs/idea014/design003/code/train.py)

### 1. New import at top
```python
import copy
```

### 2. After model is built, initialise EMA model with warmup state
```python
# ── EMA model with warmup ──────────────────────────────────────────────────
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)
_ema_target_decay = args.ema_decay   # 0.9995
_ema_step = 0   # mutable via list for closure capture

def _update_ema():
    global _ema_step
    # Warmup formula: decay_t = min(target, (1+step)/(10+step))
    decay_t = min(_ema_target_decay, (1.0 + _ema_step) / (10.0 + _ema_step))
    with torch.no_grad():
        for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay_t).add_(live_p.data, alpha=1.0 - decay_t)
    _ema_step += 1
```
Note: use `global _ema_step` since `_update_ema` is defined in `main()` scope. Alternatively, wrap step counter in a single-element list `[0]` to avoid the global.

**Cleaner closure pattern (recommended):**
```python
_ema_state = {"step": 0}

def _update_ema():
    s = _ema_state["step"]
    decay_t = min(_ema_target_decay, (1.0 + s) / (10.0 + s))
    with torch.no_grad():
        for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay_t).add_(live_p.data, alpha=1.0 - decay_t)
    _ema_state["step"] = s + 1
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

### 4. Main loop — pass updater
```python
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args,
                                  iter_logger=iter_logger, update_ema=_update_ema)
```

### 5. Validation on EMA model
```python
            ema_model.eval()
            val_m = validate(ema_model, val_loader, device, args)
```

### 6. Checkpoint includes EMA state
```python
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "ema": ema_model.state_dict(),
                     "ema_step": _ema_state["step"],
                     "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(),
                     "best_mpjpe": best_mpjpe},
                    str(out_dir / "best.pth"),
                )
```

## Where EMA Update Happens
- After each **effective optimizer step** (every `accum_steps` micro-batches).
- Step counter `_ema_state["step"]` increments each call, controlling the warmup ramp.

## Which State Dict is Reported
- All `val_*` metrics come from `validate(ema_model, ...)`.
- The live model metric is never written to CSV.

## Sanity Check
- At step=0, decay_t ≈ 0.091 → EMA is almost all live weights → nearly identical to live model for the first few batches.
- After ~2000 steps (≈5 epochs at batch=4, accum=8, ~5k batches/epoch → ~625 accum steps/epoch → 2000 steps ≈ 3.2 epochs), decay saturates at 0.9995.
- Setting `ema_target_decay = 0.0` makes `decay_t = 0` always → EMA always equals live model → sanity-checks the path.

## Rationale vs Design001
- Design001 uses fixed decay=0.999 from epoch 0, so the very first few thousand (noisy warmup-phase) weight snapshots contribute disproportionately.
- Design002 uses decay≈0 in early steps, ramping to 0.9995 (even slower than design001's 0.999) by mid-training, giving the model time to escape the warmup noise region before locking in a slow EMA.

## Memory Budget
- Same as design001: one extra model copy ≈ 345 MB fp32. Comfortably within 11 GB.
