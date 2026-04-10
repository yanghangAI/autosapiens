# Review Log — idea014

## design001 — Depth PE + Wide Head (No LLRD)
**Date:** 2026-04-10
**Verdict:** APPROVED
Wide head (hidden_dim=384) on idea008/design003 with flat optimizer. Config-only change (head_hidden=384). 384/8=48 per head valid. ~308M params within VRAM. Flat optimizer groups (backbone 1e-5, depth_pe 1e-4, head 1e-4) preserved. All 19 config fields specified.

## design002 — LLRD + Depth PE + Wide Head (Triple Combination)
**Date:** 2026-04-10
**Verdict:** APPROVED
Full triple combination: LLRD gamma=0.90 unfreeze=5, continuous depth PE sqrt, wide head 384. LLRD schedule fully specified for both phases (14 groups frozen, 27 groups unfrozen). Depth PE params correctly at head-level LR (not LLRD-decayed). train.py ports LLRD from idea004/design002. All 21 config fields specified.

## design003 — LLRD + Depth PE + Wide Head + Weight Decay 0.3
**Date:** 2026-04-10
**Verdict:** APPROVED
Identical to design002 with weight_decay=0.3 (10x increase). Config-only variant. Tests regularization counterbalance for the triple combination's larger parameter space. All 21 config fields specified.
