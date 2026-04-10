# Code Review — idea014/design002

**Design:** LLRD + Depth PE + Wide Head (Triple Combination)
**Reviewer verdict:** APPROVED

## Checklist

1. **Config verification:**
   - `output_dir` = correct path (idea014/design002)
   - `head_hidden` = 384 — widened, correct
   - `head_num_heads` = 8, `head_num_layers` = 4 — correct
   - `lr_backbone` = 1e-4, `base_lr_backbone` = 1e-4 — LLRD base rate, correct
   - `llrd_gamma` = 0.90 — correct
   - `unfreeze_epoch` = 5 — correct
   - `lr_head` = 1e-4, `lr_depth_pe` = 1e-4 — correct
   - `weight_decay` = 0.03 — correct
   - `num_depth_bins` = 16, `warmup_epochs` = 3, `epochs` = 20 — correct
   - `grad_clip` = 1.0, `lambda_depth` = 0.1, `lambda_uv` = 0.2 — correct

2. **LLRD implementation in train.py:**
   - `_block_lr(block_idx)`: `BASE_LR * (GAMMA ** (NUM_BLOCKS - 1 - block_idx))` = `1e-4 * 0.90^(23-i)`. Correct per design.
   - `_embed_lr()`: `BASE_LR * (GAMMA ** NUM_BLOCKS)` = `1e-4 * 0.90^24 ~ 8.014e-6`. Correct.

3. **Frozen phase (epochs 0-4):**
   - `_build_optimizer_frozen()` freezes blocks 0-11 + patch_embed. Blocks 12-23 active with per-block LLRD. Depth PE at `lr_depth_pe`. Head at `lr_head`.
   - Param groups: 12 blocks + depth_pe + head = 14. Correct per design.
   - Group index 11 = block23, group 12 = depth_pe, group 13 = head. LR reporting uses indices 11 and 13. Correct.

4. **Full phase (epoch 5+):**
   - `_build_optimizer_full()` unfreezes all. Groups: embed(1) + blocks 0-23(24) + depth_pe(1) + head(1) = 27. Correct per design.
   - Group index 24 = block23, group 25 = depth_pe, group 26 = head. LR reporting uses indices 24 and 26. Correct.

5. **Depth PE params not subject to LLRD** — In both phases, depth_pe_params get `lr_depth_pe=1e-4` (head-level), in their own group separate from backbone LLRD groups. Correct per design.

6. **Optimizer rebuild at epoch 5** — Line 275-279: rebuilds optimizer, sets `initial_lr` for all groups. Correct.

7. **Cosine schedule with warmup** — `get_lr_scale` applied multiplicatively to all `initial_lr` values. Correct.

8. **model.py** — Identical to design001. DepthBucketPE with sqrt spacing, wide head via `hidden_dim=384`. Correct.

9. **Loss formulation** — Unchanged: standard pose_loss, lambda_depth=0.1, lambda_uv=0.2. Correct.

## Issues

None found. The triple combination is correctly implemented with all three components (LLRD, depth PE, wide head) properly integrated.
