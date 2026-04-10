# Code Review — idea012/design001

**Design:** Head Dropout 0.2
**Reviewer verdict:** APPROVED

## config.py

Verified the single changed field:

- `head_dropout = 0.2` -- correct (was 0.1 in baseline)
- output_dir: `.../idea012/design001` -- correct
- All other fields match idea004/design002 baseline:
  - gamma=0.90, unfreeze_epoch=5, lr_backbone=1e-4, lr_head=1e-4 -- correct
  - weight_decay=0.03, drop_path=0.1 -- correct (unchanged)
  - epochs=20, warmup_epochs=3, grad_clip=1.0 -- correct
  - lambda_depth=0.1, lambda_uv=0.2 -- correct

Note: config uses `gamma` (not `llrd_gamma`) and `lr_backbone` (not `base_lr_backbone`) which matches the idea004/design002 starting point naming convention. The train.py references these correctly.

## train.py

Uses the idea004/design002 LLRD train.py pattern. Verified:
- `_block_lr` and `_embed_lr` formulas match LLRD spec
- `_build_optimizer_frozen`: 13 groups (blocks 12-23 + head). Correct -- no depth_pe group since this baseline doesn't have depth PE.
- `_build_optimizer_full`: 26 groups (embed + 24 blocks + head). Correct.
- Progressive unfreezing at `args.unfreeze_epoch`. Correct.
- `head_dropout` is passed to SapiensPose3D constructor and propagated to the decoder. The 0.2 value from config is correctly consumed.
- No hardcoded dropout values in train.py.

## model.py

Unchanged from idea004/design002 baseline. No depth PE module. Correct per design -- only config changes.

## transforms.py

No changes. Correct.

## Issues Found

None. This is a clean config-only change (head_dropout 0.1 -> 0.2).
