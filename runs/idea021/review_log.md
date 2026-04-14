# Review Log: idea021

## design001 — Kinematic Soft-Attention Bias in Refine Decoder Only (Axis A1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design001/review.md`

BFS kin_bias helper fully specified; float mask (additive) semantics correct; `kin_bias_scale` zero-init; applied to `self.refine_decoder` only; coarse decoder unchanged. 1 new scalar param in head group.

---

## design002 — Joint-Group Query Injection Before Refine Decoder (Axis A2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design002/review.md`

`group_emb` Embedding(4,384) zero-init; `joint_group_ids` buffer (70,) covers all 70 joints; `unsqueeze(0)` broadcast correct; 1,536 new params in head group; zero-init warm-start correct.

---

## design003 — Bone-Length Loss on J2 with lambda=0.05 (Axis B1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design003/review.md`

`bone_length_loss()` formula correct; BODY_EDGES filter `a < 22 and b < 22` correct; indexing into (B, 22, 3) sub-tensor correct; `lambda_bone=0.05` in config; applied to J2 only; zero new parameters.

---

## design004 — Kinematic Bias + Joint-Group Injection Combined (Axis B2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design004/review.md`

Clean combination of design001 + design002; both zero-initialized; `tgt_mask=bias_matrix` to refine_decoder only; coarse decoder unchanged; 1,537 new params total; no bone/symmetry loss (correctly avoids idea019/design005 over-regularization failure).
