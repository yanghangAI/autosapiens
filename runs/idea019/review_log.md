# idea019 Design Review Log

## design001 — Bone-Length Auxiliary Loss on Refinement Output (Axis A1)
**Verdict: APPROVED**
Clean loss-only change. `bone_length_loss` over 21 body edges (SMPLX_SKELETON filtered to a<22 and b<22), applied to J2 with lambda_bone=0.1. Loss formula matches spec. Zero new parameters. All baseline HPs preserved. No issues.

---

## design002 — Kinematic-Chain Soft Self-Attention Bias in Refinement Pass (Axis A2)
**Verdict: APPROVED**
BFS-computed (70,70) kinematic distance buffer + learnable scalar kin_bias_scale (init=0.0). Manual pass-2 decoder layer loop passes tgt_mask=bias as additive float. No -inf entries; critical invariant holds. kin_bias_scale auto-captured in head_params. 1 new parameter. All baseline HPs preserved. No issues.

---

## design003 — Left-Right Symmetry Loss (Axis B1)
**Verdict: APPROVED**
symmetry_loss over 6 L/R limb segment pairs, applied to J2 with lambda_sym=0.05. All 6 SYM_PAIRS indices verified against infra.py skeleton edges — all valid and within BODY_IDX range. Zero new parameters. Loss formula matches spec. No issues.

---

## design004 — Joint-Group Query Initialization in Refinement Pass (Axis B2)
**Verdict: APPROVED (with flagged non-fatal index inconsistency)**
group_emb Embedding(4,384) zero-init added to queries2 before pass 2. Body joints 0-21 all correctly assigned to groups 0/1/2. Minor issue: _TORSO list contains [23,24] as new-space indices but design intends original joints 23/24 (eyes at new indices 22/23). This misassigns new index 22 (left_eye) to group 3 and new index 24 (hand joint) to group 0. Impact is cosmetic — body joints unaffected, loss unaffected, zero-init guarantees identical startup. Builder should fix _TORSO to [0,3,6,9,12,15,22,23] (new index space) or omit eyes entirely. 1,536 new params. No blocking issues.

---

## design005 — Combined Anatomical Priors: Bone-Length + Symmetry + Kinematic Bias (Axis B3)
**Verdict: APPROVED**
Correct union of designs 001+002+003. Loss formula matches idea.md Axis B3 spec exactly. Forward pass code complete and correct including pelvis_token from out2. Minor style issue (redundant `import collections as _col`) is inert. 1 new scalar parameter. All baseline HPs preserved. No issues.
