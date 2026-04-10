# Review Log — idea012

## design001 — Head Dropout 0.2
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** Single config change: head_dropout 0.1->0.2. No code changes needed. All other params match idea004/design002.

## design002 — Weight Decay 0.3
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** Single config change: weight_decay 0.03->0.3. Aggressive but within standard ViT range. Note: idea.md incorrectly states baseline is 0.1; actual is 0.03. Design itself is correct.

## design003 — Stochastic Depth 0.2
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** Single config change: drop_path 0.1->0.2. Linearly distributed across blocks by constructor. No code changes needed.

## design004 — R-Drop Consistency
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** New config field rdrop_alpha=1.0. Second forward pass under torch.no_grad() + detach. Consistency on body joints only. Implementation pseudocode thorough and unambiguous.

## design005 — Combined Regularization
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** Three config changes: head_dropout=0.2, weight_decay=0.2, drop_path=0.2. Weight decay deliberately set to 0.2 (not 0.3) to avoid over-regularizing. No R-Drop. Config-only changes.
