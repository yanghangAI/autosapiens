# Review Log: idea020

## design001 — Stop-Gradient on Coarse J1 Before Refinement (Axis A1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design001/review.md`

One-line model.py change: `self.refine_mlp(J1.detach())`. Gradient decoupling is correct and well-justified. Zero new parameters. All config fields specified.

---

## design002 — Reduced Coarse Supervision Weight 0.1 (Axis A2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design002/review.md`

One-line train.py change: `l_pose = 0.1 * l_pose1 + 1.0 * l_pose2`. `refine_loss_weight=0.1` in config. Non-zero coarse weight prevents degenerate collapse. Zero new parameters. All config fields specified.

---

## design003 — L1 Loss on Refinement Pass Only (Axis B1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design003/review.md`

One-line train.py change: `l_pose2 = F.l1_loss(...)`. Coarse pass retains Smooth L1. Asymmetric loss choice well-justified. Zero new parameters. All config fields specified.

---

## design004 — Higher LR for Refine Decoder (2x Head LR, Axis B2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design004/review.md`

Optimizer group split into coarse-head (LR=1e-4) and refine-head (LR=2e-4). Helper functions fully specified. Group index references in LR reporting block correctly updated. `lr_refine_head=2e-4` in config. Zero new parameters.

---

## design005 — Residual Refinement Formulation (Axis B3)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea020/design005/review.md`

Two-line model.py change: `delta = self.joints_out2(out2); J2 = J1 + delta`. Zero-init warm-start analysis correct. `depth_out`/`uv_out` unchanged (feature-based). Loss applied to absolute J2. Zero new parameters. All config fields specified.
