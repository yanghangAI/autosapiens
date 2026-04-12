# idea018 — Weight Averaging: EMA and SWA on the SOTA Triple-Combo

**Expected Designs:** 4

## Baseline starting point
`runs/idea014/design003/code/` (current SOTA: 106.85 mm val_mpjpe_body, 96.73 mm val_pelvis, 103.51 mm val_mpjpe_weighted).

This idea modifies only the *training loop* around the SOTA model — no architecture, loss, data, or optimizer changes — so any gain is cleanly attributable to weight-space averaging.

## Motivation (broader reflection)

Across 14 completed ideas, every experiment uses the *final-epoch weights* for evaluation. Weight-space averaging is a universally-effective, practically-free training trick (Izmailov et al. 2018 SWA; Polyak averaging / EMA used by MAE, DINO, EfficientNet, YOLO, etc.) that has never been tried in this project. A persistent 22–30 mm train/val gap across every run (see e.g. idea014/design003: 76.8 train_body vs 106.85 val_body) means the final-epoch weights are almost certainly in a sharp local minimum. Averaging weights across the tail of training flattens the minimum and typically yields 0.5–2 % relative improvement on dense prediction tasks with near-zero added compute and zero added parameters.

Key observations from prior results used here:
- idea014/design003 is SOTA (106.85 body / 103.51 weighted). It combines LLRD(γ=0.90, unfreeze=5) + continuous depth PE + wide head + wd=0.3.
- idea011/design001 has the best pelvis depth (94.2 mm). If weight averaging helps, idea014/design003 with EMA should retain or improve on the 96.73 pelvis value without needing another depth-PE variant.
- idea012 (regularization idea) already probed the dropout/weight-decay axis; none of those dominated d003 + wd=0.3. EMA is a fundamentally different regularizer (weight-space, not activation-space) and therefore worth trying as a standalone new axis.

## Design axes and category breakdown

All 4 designs below are **Category B — Novel Exploration**. None of them have been tried before in this project.

### design001 — EMA of full model weights (decay=0.999)
- Maintain an `ema_model` copy (same architecture, `requires_grad_(False)`) updated after each optimizer step:
  `ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)`.
- Update starts from epoch 0 (initialised as a clone of the live model).
- Validation each epoch is run on `ema_model` (not the live model). Training is unchanged.
- Checkpoint saves both live and EMA weights; final reported metric is on EMA weights.
- Decay = 0.999 (standard MAE/DINO value; effective window ≈ 1000 steps ≈ 2–3 epochs given BATCH_SIZE=4 × ACCUM_STEPS=8).

### design002 — EMA with warmup (decay starts at 0, ramps to 0.9995)
- `decay_t = min(target_decay, (1 + step) / (10 + step))` — this standard warmup prevents the EMA from being dragged by the very noisy early steps.
- `target_decay = 0.9995` (slower, longer window ≈ 2000 steps ≈ 5 epochs).
- Same evaluation protocol: validation on EMA copy after each epoch.
- Rationale: the first 3 warmup epochs of cosine schedule produce extremely noisy gradients; a cold EMA start avoids polluting the average.

### design003 — Stochastic Weight Averaging (SWA) over the last 5 epochs
- First 15 epochs: train exactly as idea014/design003.
- Epochs 16–20 (the last 5): at the end of each epoch, accumulate a running average of the weights:
  `swa_param = (swa_param * n + param) / (n + 1)` with n = count of snapshots so far.
- During SWA phase, switch optimizer to a constant LR equal to `0.5 * current cosine LR at epoch 15` (standard SWA recipe).
- Validation during SWA phase is run on the averaged weights. BN/LN statistics do not require recalibration (LN only, no BN in Sapiens/Pose3DHead).
- Final reported metric is on the SWA weights.

### design004 — EMA (decay=0.999) + last-epoch SWA fine-tune
- Train with EMA as in design001 throughout.
- At the end of epoch 20, take the EMA weights, load them into the live model, and run **one extra epoch** of training at a constant LR = `1e-6` (flat, no schedule) — this is a "polish" pass that lets the averaged weights settle at a true loss-surface minimum for the exact validation distribution.
- Final reported metric is on the polished weights at the end of epoch 21. The extra epoch is cheap (1/20 ≈ 5 % more compute).
- Rationale: pure weight averaging can end slightly off the true loss minimum; a short flat-LR polish exploits flatness without destroying it.

## Shared implementation notes for the Designer

- **Baseline file:** `runs/idea014/design003/code/train.py` and `model.py` / `config.py` / `transforms.py` alongside it. Do NOT change `model.py`, `config.py`, or `transforms.py` except to add a trivial EMA/SWA option to the training loop.
- **No architecture changes.** No new parameters. No loss changes. LLRD, depth PE, wide head, wd=0.3 all stay exactly as in idea014/design003.
- **EMA copy:** `copy.deepcopy(model).eval()`, then `for p in ema_model.parameters(): p.requires_grad_(False)`. Update after `optimizer.step()` inside the accumulation boundary (not every micro-batch).
- **Validation protocol:** all 4 designs must write the *averaged-weights* metric as the canonical `val_*` in `iter_metrics.csv`, so results.csv comparisons remain apples-to-apples with prior ideas.
- **LayerNorm statistics:** only LayerNorm is used (no BatchNorm). LN has no running stats, so no BN recalibration is needed after weight averaging.
- **Memory budget:** an extra full model copy (ViT-B backbone ≈ 86 M params ≈ 345 MB fp32) fits comfortably in 11 GB alongside the live model's activations. Designs 1, 2, 4 need one extra copy; design 3 needs one extra copy only during the last 5 epochs.
- **Checkpoint format:** save `{ 'model': live.state_dict(), 'ema': ema_model.state_dict() }` (or `swa`) so downstream evaluation can pick either.
- **Designer should NOT re-design the baseline** — idea014/design003 is the baseline control point and is already trained.

## Number of novel variations for the Designer

**4 novel designs** (design001, design002, design003, design004). The Designer should generate exactly 4 `design.md` files, one per axis above, each with:
1. Exact `train.py` diff against `runs/idea014/design003/code/train.py`.
2. Where in the training loop the EMA/SWA update happens (inside or outside the gradient-accumulation boundary).
3. Which state_dict is used for the reported `val_*` metric.
4. A short sanity check (e.g., decay=0 should recover the live-model baseline).
