# Code Review — idea014/design003

**Design:** LLRD + Depth PE + Wide Head + Weight Decay 0.3
**Reviewer verdict:** APPROVED

## Checklist

1. **Config verification:**
   - `output_dir` = correct path (idea014/design003)
   - `head_hidden` = 384 — correct
   - `lr_backbone` = 1e-4, `base_lr_backbone` = 1e-4, `llrd_gamma` = 0.90, `unfreeze_epoch` = 5 — correct
   - `lr_head` = 1e-4, `lr_depth_pe` = 1e-4 — correct
   - `weight_decay` = 0.3 — **the key difference from design002**. Correct per design (10x baseline 0.03).
   - `num_depth_bins` = 16, `warmup_epochs` = 3, `epochs` = 20 — correct
   - `grad_clip` = 1.0, `lambda_depth` = 0.1, `lambda_uv` = 0.2 — correct

2. **train.py** — Identical to design002. The LLRD logic, optimizer build functions, freeze/unfreeze, depth PE grouping, and LR schedule are all the same. `weight_decay` is read from config and applied to all param groups via `torch.optim.AdamW(groups, weight_decay=args.weight_decay)`. Correct.

3. **model.py** — Identical to design001/design002. No changes needed. Correct.

4. **Only difference is weight_decay in config.py** — Verified: config.py line 43 has `weight_decay = 0.3`. All other fields identical to design002 except `output_dir`. Correct per design spec ("identical to design002 except for the weight_decay value").

## Issues

None found. This is a config-only variant of design002 with the single change of weight_decay from 0.03 to 0.3.
