# Design Review — idea011/design001

**Reviewer verdict: APPROVED**

## Summary

Direct combination of LLRD (gamma=0.90, unfreeze_epoch=5) with sqrt-spaced continuous depth PE from idea008/design003. This is the most conservative combination of the two best orthogonal improvements.

## Evaluation

### Completeness
- Starting point clearly specified: `runs/idea008/design003/`.
- LLRD formula explicitly stated with correct math: `lr_i = base_lr * gamma^(num_blocks - 1 - i)`.
- Computed LR values verified: Block 0 = 1e-4 * 0.90^23 ~ 8.904e-6 (correct). Embed = 1e-4 * 0.90^24 ~ 8.014e-6 (correct).
- Progressive unfreezing clearly described: blocks 0-11 + embeddings frozen epochs 0-4, unfrozen at epoch 5 with full optimizer rebuild.
- Depth PE params (`row_emb`, `col_emb`, `depth_emb`) correctly assigned to high-LR group (1e-4), never frozen.
- Param group counts: 14 frozen phase (12 blocks + depth_pe + head), 27 full phase (24 blocks + embed + depth_pe + head). Correct.

### Mathematical Correctness
- LLRD formula matches idea004/design002 exactly.
- Architecture table confirms no modifications to depth PE components.

### Feasibility (1080ti / 20 epochs)
- No new parameters added — only optimizer schedule changes. VRAM unchanged from idea008/design003 baseline. Well within 11GB.

### Config Fields
- All 19 config fields explicitly listed with correct values.
- `llrd_gamma=0.90`, `base_lr_backbone=1e-4`, `unfreeze_epoch=5`, `lr_depth_pe=1e-4` all present.
- `weight_decay=0.03` matches the idea constraint.

### Builder Instructions
- File-level edit plan is clear: no model.py changes, all LLRD in train.py, config additions specified.
- Depth PE parameter identification by name documented.

## Issues Found
None.
