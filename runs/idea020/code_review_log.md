# Code Review Log: idea020

## design001 — Stop-Gradient on Coarse J1 Before Refinement (Axis A1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design001/code_review.md`

`J1.detach()` correctly applied in model.py; train.py and config.py match spec; 2-epoch test passed (w=803.6mm).

---

## design002 — Reduced Coarse Supervision Weight 0.1 (Axis A2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design002/code_review.md`

`0.1 * l_pose1 + 1.0 * l_pose2` correctly in train.py; config refine_loss_weight=0.1; model.py unchanged; 2-epoch test passed (w=825.9mm).

---

## design003 — L1 Loss on Refinement Pass Only (Axis B1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design003/code_review.md`

`F.l1_loss` for l_pose2; coarse retains Smooth L1; config refine_loss_weight=0.5; model.py unchanged; 2-epoch test passed (w=787.6mm).

---

## design004 — Higher LR for Refine Decoder (2x Head LR, Axis B2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design004/code_review.md`

`_coarse_head_params()`/`_refine_head_params()` split correct; group indices match spec (frozen: 11=block23, 13=coarse_head; full: 24=block23, 26=coarse_head); config lr_refine_head=2e-4; 2-epoch test passed (w=648.3mm).

---

## design005 — Residual Refinement Formulation (Axis B3)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design005/code_review.md`

`delta = self.joints_out2(out2); J2 = J1 + delta` correctly applied in model.py; zero-bias init preserved; train.py and config.py unchanged; 2-epoch test passed (w=988.5mm, higher due to warm-start expected).

---
