# Code Review — idea011/design001

**Design:** LLRD (gamma=0.90, unfreeze=5) + Sqrt-Spaced Continuous Depth PE
**Reviewer verdict:** APPROVED

## config.py

All 19 specified fields match the design exactly:

| Field | Design | Code | Match |
|-------|--------|------|-------|
| output_dir | `.../idea011/design001` | `.../idea011/design001` | OK |
| arch | sapiens_0.3b | sapiens_0.3b | OK |
| img_h / img_w | IMG_H / IMG_W | IMG_H / IMG_W | OK |
| head_hidden | 256 | 256 | OK |
| head_num_heads | 8 | 8 | OK |
| head_num_layers | 4 | 4 | OK |
| head_dropout | 0.1 | 0.1 | OK |
| drop_path | 0.1 | 0.1 | OK |
| epochs | 20 | 20 | OK |
| warmup_epochs | 3 | 3 | OK |
| base_lr_backbone | 1e-4 | 1e-4 | OK |
| llrd_gamma | 0.90 | 0.90 | OK |
| unfreeze_epoch | 5 | 5 | OK |
| lr_head | 1e-4 | 1e-4 | OK |
| lr_depth_pe | 1e-4 | 1e-4 | OK |
| weight_decay | 0.03 | 0.03 | OK |
| num_depth_bins | 16 | 16 | OK |
| grad_clip | 1.0 | 1.0 | OK |
| lambda_depth / lambda_uv | 0.1 / 0.2 | 0.1 / 0.2 | OK |

## train.py

1. **LLRD formula:** `_block_lr(i) = BASE_LR * (GAMMA ** (NUM_BLOCKS - 1 - i))` matches design exactly.
2. **Embed LR:** `_embed_lr() = BASE_LR * (GAMMA ** NUM_BLOCKS)` matches design.
3. **Progressive unfreezing:** Blocks 0-11 and patch_embed frozen at init; all unfrozen at `epoch == UNFREEZE_EPOCH` (epoch 5). Correct.
4. **Depth PE params:** Collected via `model.backbone.depth_bucket_pe.parameters()` and placed in dedicated group at `lr_depth_pe = 1e-4`. Never frozen. Correct.
5. **Optimizer frozen phase:** 12 block groups (12-23) + depth_pe + head = 14 groups. Matches design.
6. **Optimizer full phase:** embed + 24 blocks + depth_pe + head = 27 groups. Matches design.
7. **LR schedule:** Linear warmup + cosine decay via `get_lr_scale`. `initial_lr` set correctly on both initial build and rebuild.
8. **Param group index math:** Frozen: block23 at index 11, head at 13. Full: block23 at index 24, head at 26. Verified correct.
9. **pos_embed handling:** Noted as buffer, not Parameter. No action needed for freeze/unfreeze. Correct.

## model.py

No changes from idea008/design003 starting point. Depth PE architecture (sqrt spacing, continuous interpolation, row/col/depth decomposition) untouched. Correct.

## transforms.py

No changes. Correct.

## Issues Found

None.
