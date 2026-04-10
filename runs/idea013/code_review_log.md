# Code Review Log — idea013

## design001 — Small-Beta Smooth L1 (beta=0.01)
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** F.smooth_l1_loss with beta=0.01 matches design; depth/UV losses unchanged; all config fields correct; LLRD optimizer logic correct.

## design002 — Large-Beta Smooth L1 (beta=0.1)
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** F.smooth_l1_loss with beta=0.1 matches design; depth/UV losses unchanged; all config fields correct; LLRD optimizer logic correct.

## design003 — Bone-Length Auxiliary Loss (lambda_bone=0.1)
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** BODY_BONES filter from SMPLX_SKELETON correct; bone_length_loss function matches design exactly; 0.1*l_bone in loss composition correct; l_bone in del statement; all config fields correct; minor omission of loss_bone in iter_logger is cosmetic only.

## design004 — Hard-Joint-Weighted Loss
**Date:** 2026-04-10
**Verdict:** APPROVED
**Notes:** One-shot weight computation after epoch 0 correct; clamp [0.5, 2.0] and re-normalize to sum=22 match design; weighted per-joint Smooth L1 (beta=0.05) for epochs 1-19; err_accum dict pattern clean; all config fields correct.
