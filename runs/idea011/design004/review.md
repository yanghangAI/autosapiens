# Design Review — idea011/design004

**Reviewer verdict: APPROVED**

## Summary

LLRD (gamma=0.90, unfreeze=5) combined with gated continuous depth PE from idea008/design002 (linear anchor spacing + learned scalar sigmoid gate). Tests whether the gate interacts beneficially with per-block LR decay.

## Evaluation

### Completeness
- Starting point: `runs/idea008/design002/`. Correct per idea.md constraint (design 4 uses idea008/design002, not design003).
- LLRD formula with gamma=0.90, same values as design001. Correct.
- Progressive unfreezing at epoch 5. Correct.
- Depth PE params now include `depth_gate` in addition to `row_emb`, `col_emb`, `depth_emb`. All at high LR (1e-4), never frozen. Correct.
- Gated depth PE recap clearly describes the gate mechanism: `sigmoid(self.depth_gate)` with zero-init (sigmoid=0.5). Consistent with idea008/design002.
- Architecture table correctly shows uniform linear anchor spacing (not sqrt) and scalar sigmoid gate.

### Mathematical Correctness
- LR values identical to design001. Verified.
- Gate mechanism recap is accurate.

### Feasibility
- No new parameters beyond what idea008/design002 already has (1 scalar gate param). VRAM unchanged. Within 11GB.

### Config Fields
- All 19 config fields explicitly listed. Values match design001 except the starting point differs.
- Note: `num_depth_bins=16` matches idea008/design002. Correct.

### Builder Instructions
- Clear: start from idea008/design002 (not design003), reuse design001 LLRD pattern, include `depth_gate` in depth_pe param group.

## Issues Found
None.
