# Code Review Log — idea014

## design001 — Depth PE + Wide Head (No LLRD)
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** head_hidden=384 in config; flat optimizer with 3 groups (backbone lr=1e-5, depth_pe lr=1e-4, head lr=1e-4); no LLRD; model.py has DepthBucketPE with sqrt spacing and continuous interpolation; Pose3DHead parameterized via hidden_dim; all 19 config fields correct.

## design002 — LLRD + Depth PE + Wide Head (Triple Combination)
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** head_hidden=384; LLRD gamma=0.90 with per-block LR decay; unfreeze at epoch 5; depth PE params at head-level LR (not subject to LLRD); frozen phase 14 groups, full phase 27 groups; all config fields correct including base_lr_backbone and llrd_gamma.

## design003 — LLRD + Depth PE + Wide Head + Weight Decay 0.3
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** Identical to design002 except weight_decay=0.3 (10x baseline). Config-only variant. train.py and model.py identical to design002. All config fields correct.
