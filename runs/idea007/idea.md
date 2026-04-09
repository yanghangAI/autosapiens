# Depth-Bucket Positional Embeddings with Layer-Wise Fine-Tuning

**Expected Designs:** 3

## Starting Point

The baseline starting point for this idea is:

`runs/idea005/design001/code/`

That design achieved the best completed validation score so far in `results.csv`
(`val_mpjpe_weighted = 121.4 mm`) by replacing the backbone's standard 2D positional
embedding with a row + column + depth-bucket positional embedding. The completed
`idea004` runs then showed that conservative backbone optimization schedules further
improved stability, with the best layer-wise schedule reaching `130.7 mm` on the
baseline architecture.

## Concept

Exploit the best completed architecture from `idea005/design001` and combine it with
the best completed backbone adaptation strategy from `idea004`: layer-wise learning-rate
decay (LLRD) and progressive unfreezing. The central question is whether the explicit
depth-aware positional signal benefits even more when shallow pretrained ViT layers are
protected early in training.

## Broader Reflection

### Strong prior results to build on

- `idea005/design001` (`121.4 mm`) was the best completed result overall. The decomposed
  row/column/depth positional embedding appears to add meaningful geometric signal with
  very little parameter overhead.
- `idea004/design002` (`130.7 mm`) was the best completed schedule variant. It used
  constant-decay LLRD with `gamma=0.90` and progressive unfreezing at epoch 5.

### Patterns to avoid

- `idea003/design001` (`163.0 mm`) showed that aggressive curriculum changes can hurt
  convergence when the proxy budget is only 20 epochs.
- Several `idea004` variants clustered near `131-132 mm`, so the schedule search should
  stay narrow and not explode into many low-value variants.
- `idea006` is still in flight, so this idea should not depend on unfinished augmentation
  results.

## Search Axes

### Category A — Exploit & Extend

1. Combine `idea005/design001` depth-bucket positional embeddings with the
   `idea004/design002` LLRD schedule (`gamma=0.90`, `unfreeze_epoch=5`).
2. Test whether the gentler `idea004/design001` schedule (`gamma=0.95`, `unfreeze_epoch=5`)
   is a better fit for the already-strong depth-aware positional embedding backbone.
3. Keep the stronger `gamma=0.90` decay but move the unfreeze earlier to epoch 3 to test
   whether the stronger geometry prior can tolerate earlier whole-backbone adaptation.

### Category B — Novel Exploration

This idea does not introduce a brand-new architectural module; the novelty is the
first-time combination of the best positional encoding from `idea005` with the
best-performing backbone adaptation schedules from `idea004`, plus one new unfreeze-time
variant that was not previously tested on the depth-bucket backbone.

## Expected Designs

The Designer should generate **3** novel designs:

1. `gamma=0.95`, `unfreeze_epoch=5`
2. `gamma=0.90`, `unfreeze_epoch=5`
3. `gamma=0.90`, `unfreeze_epoch=3`

Each design should keep the depth-bucket positional embedding module from
`runs/idea005/design001/code/` unchanged and vary only the optimization schedule and
related config fields needed to support LLRD.
