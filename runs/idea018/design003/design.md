# idea018 / design003 — Stochastic Weight Averaging (SWA) over Last 5 Epochs

## Starting Point
`runs/idea014/design003/code/` — the current SOTA triple-combo baseline (LLRD γ=0.90, unfreeze=5, depth PE, wide head, wd=0.3, 20 epochs).

## Problem
The final-epoch weights land in a sharp local minimum. SWA averages weights at the end of several epochs (rather than every optimizer step) over a flat constant-LR tail, which pushes the average into the wider, flatter basin identified by Izmailov et al. (2018).

## Proposed Solution
Run epochs 0–14 exactly as baseline (cosine LR + LLRD). At epoch 15, switch the optimizer's LR groups to a constant LR = `0.5 × cosine_LR_at_epoch15` and begin accumulating a uniform running average of the weights at the end of each epoch. After 5 SWA snapshots (epochs 15–19), report the averaged weights.

No architecture, loss, or optimizer-type changes. No BN recalibration needed (model uses LayerNorm only).

## Configuration (config.py fields to add)
```python
# SWA settings
swa_start_epoch = 15    # first epoch to include in SWA average (0-indexed)
swa_lr_factor   = 0.5   # constant LR = swa_lr_factor * cosine_LR_at_swa_start
```
All other config values identical to `runs/idea014/design003/code/config.py`, except:
```python
output_dir = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea018/design003"
```

## train.py Changes (diff against runs/idea014/design003/code/train.py)

### 1. New import at top
```python
import copy
```

### 2. Before the main epoch loop, initialise SWA state
```python
# ── SWA state ───────────────────────────────────────────────────────────────
_swa_n = 0          # number of snapshots accumulated
_swa_model = None   # will be set at first SWA epoch
```

### 3. Inside the main epoch loop — constant LR override during SWA phase
After the standard cosine LR scale block, add:
```python
        # Compute cosine LR scale (existing code)
        scale = get_lr_scale(epoch, args.epochs, args.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale

        # SWA phase: override with constant LR
        _in_swa = (epoch >= args.swa_start_epoch)
        if _in_swa:
            if epoch == args.swa_start_epoch:
                # Record the constant SWA LR from the cosine value at this epoch
                _swa_lr_scale = scale * args.swa_lr_factor
                print(f"  *** SWA phase begins at epoch {epoch+1}."
                      f" Constant LR factor={_swa_lr_scale:.4f} of initial_lr ***")
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * _swa_lr_scale
```

### 4. After `train_one_epoch`, accumulate SWA snapshot
```python
        # SWA weight accumulation (at end of each SWA epoch, after training)
        if _in_swa:
            if _swa_model is None:
                _swa_model = copy.deepcopy(model).eval()
                for p in _swa_model.parameters():
                    p.requires_grad_(False)
                # First snapshot: copy live weights directly
                _swa_model.load_state_dict(model.state_dict())
                _swa_n = 1
            else:
                # Running average: swa = (swa * n + live) / (n + 1)
                with torch.no_grad():
                    for swa_p, live_p in zip(_swa_model.parameters(), model.parameters()):
                        swa_p.data.mul_(_swa_n / (_swa_n + 1)).add_(
                            live_p.data, alpha=1.0 / (_swa_n + 1)
                        )
                _swa_n += 1
            print(f"  SWA snapshot {_swa_n} accumulated (epoch {epoch+1})")
```

### 5. Validation — use SWA model during SWA phase, live model otherwise
```python
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            eval_model = _swa_model if (_in_swa and _swa_model is not None) else model
            eval_model.eval()
            val_m = validate(eval_model, val_loader, device, args)
            torch.cuda.empty_cache()
            model.train()   # only live model needs to return to train mode
```

### 6. Checkpoint saves both live and SWA state dicts
```python
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "swa": _swa_model.state_dict() if _swa_model is not None else None,
                     "swa_n": _swa_n,
                     "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(),
                     "best_mpjpe": best_mpjpe},
                    str(out_dir / "best.pth"),
                )
```

## Where SWA Update Happens
- Once per epoch, **after** `train_one_epoch` returns, starting from `epoch >= swa_start_epoch (= 15)`.
- Not per-step; SWA is a uniform epoch-level average, not an exponential filter.

## Which State Dict is Reported
- During epochs 0–14: `validate(model, ...)` → live-model val metrics in CSV.
- During epochs 15–19: `validate(_swa_model, ...)` → SWA-model val metrics in CSV.
- Final reported metric is the SWA-averaged weights after epoch 19 (5 snapshots).

## Constant LR During SWA Phase
- At epoch 15 with `warmup_epochs=3`, `total_epochs=20`:
  `scale = 0.5 * (1 + cos(π * (15−3)/(20−3))) = 0.5 * (1 + cos(π * 12/17)) ≈ 0.5 * (1 − 0.797) ≈ 0.102`
- SWA LR = `0.5 × 0.102 = 0.051` of initial_lr per group.
- For block 23 (initial_lr = 1e-4): SWA LR ≈ 5.1e-6. For head (initial_lr = 1e-4): SWA LR ≈ 5.1e-6.

## Sanity Check
- `swa_start_epoch = 20` (beyond training): SWA never triggers; entire run uses live weights → should match baseline.
- `swa_start_epoch = 0`: All 20 epochs averaged with constant LR from epoch 0 → likely worse than baseline due to early-training noise (verifies SWA averaging path is active).

## Memory Budget
- SWA model copy initialised only at epoch 15 (not before), so no extra RAM for first 15 epochs.
- At epoch 15: ≈345 MB extra. Comfortably within 11 GB budget.
- After training: can free the live model copy and evaluate purely on _swa_model.
