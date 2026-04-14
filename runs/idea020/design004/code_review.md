# Code Review — idea020 / design004 — Higher LR for Refine Decoder (2x Head LR)

**Design_ID:** idea020/design004
**Verdict: APPROVED**

## Summary of Changes Verified

### model.py
- Unchanged from baseline (idea015/design004). Correct.

### train.py
- `_coarse_head_params()` helper defined correctly: includes `input_proj`, `joint_queries`, `decoder`, `joints_out`, `depth_out`, `uv_out`.
- `_refine_head_params()` helper defined correctly: includes `refine_decoder`, `refine_mlp`, `joints_out2`.
- `LR_REFINE = getattr(args, "lr_refine_head", args.lr_head * 2.0)` — reads from config with fallback to 2x lr_head (2e-4).
- Both `_build_optimizer_frozen()` and `_build_optimizer_full()` correctly use the two-group split for head parameters:
  - `{"params": _coarse_head_params(), "lr": args.lr_head}` (1e-4)
  - `{"params": _refine_head_params(), "lr": LR_REFINE}` (2e-4)
- Group index reporting is correct:
  - Frozen phase: `param_groups[11]` = block 23, `param_groups[13]` = coarse head (groups 0..11 = blocks 12-23, group 12 = depth_pe, group 13 = coarse head, group 14 = refine head).
  - Full phase: `param_groups[24]` = block 23, `param_groups[26]` = coarse head (groups 0 = embed, 1..24 = blocks 0-23, 25 = depth_pe, 26 = coarse head, 27 = refine head).
- These indices match the design specification exactly.

### config.py
- `output_dir` correctly set to `runs/idea020/design004`.
- `lr_refine_head = 2e-4` — ADDED as specified.
- `lr_head = 1e-4`, `refine_loss_weight = 0.5` — unchanged from baseline.
- All other config fields match design spec.

## Smoke Test
- 2-epoch test passed without errors: Training complete. Best val weighted MPJPE = 648.3mm.

## Issues Found
None.
