# Code Review — idea013/design001

**Design:** Small-Beta Smooth L1 (beta=0.01)
**Reviewer verdict:** APPROVED

## Checklist

1. **Key change: beta=0.01 in F.smooth_l1_loss** — Line 185 of train.py uses `F.smooth_l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX], beta=0.01)`. This matches the design exactly. The import `import torch.nn.functional as F` is present at line 35.

2. **Depth/UV losses unchanged** — Lines 186-187 still use `pose_loss()` from infra for `l_dep` and `l_uv`. Correct per design (only pose loss beta changes).

3. **Loss composition** — Line 188: `loss = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps`. Standard formula, unchanged.

4. **Config verification:**
   - `output_dir` = correct path (idea013/design001)
   - `head_hidden` = 256, `head_num_heads` = 8, `head_num_layers` = 4 — unchanged
   - `lr_backbone` = 1e-4, `lr_head` = 1e-4, `gamma` = 0.90, `unfreeze_epoch` = 5 — matches design
   - `weight_decay` = 0.03, `epochs` = 20, `warmup_epochs` = 3 — correct
   - `grad_clip` = 1.0, `lambda_depth` = 0.1, `lambda_uv` = 0.2 — correct

5. **LLRD optimizer logic** — Present and correct (frozen blocks 0-11, progressive unfreeze at epoch 5, per-block LR decay). This is inherited from the idea004/design002 starting point. Consistent with design spec.

6. **No config.py changes needed** — Design says beta is not a config field; it's hardcoded in the F.smooth_l1_loss call. Acceptable.

7. **del statement** — Line 218 correctly includes all loss variables.

## Issues

None found.
