# Design Review — idea011/design003

**Reviewer verdict: APPROVED**

## Summary

LLRD (gamma=0.90) with later unfreezing at epoch 10, combined with sqrt-spaced continuous depth PE. Tests whether the head benefits from longer exclusive training with the enriched depth PE.

## Evaluation

### Completeness
- Starting point: `runs/idea008/design003/`. Correct.
- LLRD formula with gamma=0.90, same LR values as design001. Correct.
- Progressive unfreezing delayed to epoch 10: blocks 0-11 + embeddings frozen epochs 0-9, unfrozen at epoch 10.
- Design correctly notes that warmup (3 epochs) completes before unfreeze, so newly unfrozen layers start at reduced effective LR from cosine decay. Good observation.
- Depth PE params at high LR, never frozen. Correct.
- Param group counts: 14 frozen (epochs 0-9), 27 full (epochs 10-19). Correct.

### Mathematical Correctness
- LR values identical to design001 (same gamma). Verified.
- The cosine schedule at epoch 10/20 gives scale ~ 0.5 + 0.5*cos(pi*7/17) ~ 0.5 + 0.5*cos(1.294) ~ 0.5 + 0.5*0.275 ~ 0.638. So newly unfrozen shallowest block starts at ~8.9e-6 * 0.638 ~ 5.7e-6. Reasonable.

### Feasibility
- No new parameters. VRAM unchanged. Within 11GB.
- Only 10 epochs of full backbone training (vs 15 in design001). May limit adaptation but is a valid exploration within 20-epoch budget.

### Config Fields
- All 19 config fields listed. `unfreeze_epoch=10` is the only change from design001. Correct.

### Builder Instructions
- Clear: reuse design001 pattern, change only unfreeze_epoch.

## Issues Found
None.
