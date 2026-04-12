# Review: idea018 / design003 — Stochastic Weight Averaging (SWA) over Last 5 Epochs

**Verdict: APPROVED**

## Summary
Correct SWA implementation over epochs 15–19. Running-average formula is arithmetically sound, constant-LR override is properly computed, and the validation protocol switches to the SWA model at the right time. Memory is deferred until epoch 15 as intended.

## Checklist

### Correctness
- [x] SWA triggers at `epoch >= swa_start_epoch (= 15)`, where epochs are 0-indexed (0..19). This means epochs 0–14 (first 15 epochs) use cosine LLRD, and epochs 15–19 (last 5) use SWA. Correct.
- [x] LR math at epoch 15 (warmup_epochs=3, total_epochs=20):
  `scale = 0.5 * (1 + cos(π * (15−3)/(20−3))) = 0.5 * (1 + cos(π * 12/17))`
  `cos(π * 12/17) ≈ cos(2.214) ≈ −0.797`
  `scale ≈ 0.5 * (1 − 0.797) ≈ 0.1015`
  `swa_lr = 0.5 × 0.1015 ≈ 0.051 of initial_lr per group`. Verified correct.
- [x] `_swa_lr_scale` is computed once at `epoch == swa_start_epoch` and reused for all subsequent SWA epochs — correct; constant LR means all groups use the same scale factor applied to their `initial_lr`.
- [x] Running average formula: `swa = (swa * n + live) / (n + 1)` implemented as
  `swa_p.mul_(n/(n+1)).add_(live_p, alpha=1/(n+1))`. Arithmetic verified:
  - n=1 (second snapshot): `swa = (swa1 + live2)/2`. Correct.
  - n=2 (third snapshot): `swa = (swa2 * 2 + live3)/3`. Correct.
- [x] First snapshot: `_swa_model` cloned from live model with `load_state_dict`, `_swa_n = 1`. Correct; sets the baseline for the running average.
- [x] Validation uses `_swa_model` during SWA phase, `model` otherwise. Correct.
- [x] No BN recalibration needed — model uses LayerNorm only. Confirmed correct.

### Configuration Completeness
- [x] `swa_start_epoch = 15` added.
- [x] `swa_lr_factor = 0.5` added.
- [x] `output_dir` updated to `runs/idea018/design003`.
- [x] All other fields inherited from idea014/design003.

### Architecture / Loss / Optimizer
- [x] No architecture, loss, or new-optimizer changes.
- [x] The SWA phase only overrides LR in existing param groups — not a structural optimizer change.

### Memory Budget
- [x] SWA model allocated only at epoch 15 (not epochs 0–14). At epoch 15: +345 MB. Within 11 GB.

### 20-Epoch Proxy Limit
- [x] Runs exactly 20 epochs. The SWA phase is contained within the normal epoch loop.

### Sanity Checks
- [x] `swa_start_epoch = 20`: SWA never triggers, entire run = live weights = baseline result. Valid.
- [x] `swa_start_epoch = 0`: All 20 epochs averaged from constant LR epoch 0 — degrades performance but confirms averaging path is active. Valid.

## Minor Observations
- The design validates on live `model` for epochs 0–14 and on `_swa_model` for epochs 15–19. This means the CSV shows a potential discontinuity in val metrics at epoch 15 (live → SWA). This is expected and acceptable — it actually makes the improvement attributable to SWA directly observable.
- After training, the design could optionally `del model; torch.cuda.empty_cache()` before final evaluation to free VRAM — not required but a reasonable implementation note for the Builder.

## Decision
**APPROVED.** No changes required.
