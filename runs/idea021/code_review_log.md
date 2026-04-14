# Code Review Log: idea021

## design001 — Kinematic Soft-Attention Bias in Refine Decoder Only (Axis A1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design001/code_review.md`

`_compute_kin_bias()` BFS correct; `kin_bias` buffer + `kin_bias_scale` Parameter(zeros(1)) in head; `tgt_mask=bias_matrix` passed to refine_decoder only; coarse decoder unchanged; 2-epoch test passed (w=797.9mm).

---

## design002 — Joint-Group Query Injection Before Refine Decoder (Axis A2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design002/code_review.md`

`group_emb Embedding(4,384)` zero-init; `joint_group_ids` buffer (70,) correct assignments (0-3: torso, 4-9: arms, 10-15: legs, 22-69: extremities); `queries2 + group_bias.unsqueeze(0)` applied before refine_decoder; 2-epoch test passed (w=929.5mm).

---

## design003 — Bone-Length Loss on J2 with lambda=0.05 (Axis B1)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design003/code_review.md`

`bone_length_loss()` formula correct; `BODY_EDGES` from SMPLX_SKELETON with a<22 and b<22; applied to J2 only; `args.lambda_bone=0.05`; config has `lambda_bone=0.05`; model.py unchanged; 2-epoch test passed (w=795.6mm).

---

## design004 — Kinematic Bias + Joint-Group Injection Combined (Axis B2)
**Date:** 2026-04-13
**Verdict:** APPROVED
**Review file:** `runs/idea021/design004/code_review.md`

Clean union of design001+design002; both zero-initialized; ordering correct (group_emb added to queries2 first, then tgt_mask=bias_matrix to refine_decoder); 1,537 new params; train.py unchanged; 2-epoch test passed (w=929.5mm).

---
