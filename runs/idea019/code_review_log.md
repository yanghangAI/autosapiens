# Code Review Log — idea019

---

## idea019/design001 — Bone-Length Auxiliary Loss on Refinement Output (Axis A1)

**Verdict: APPROVED**

Implementation matches design exactly. `bone_length_loss` applied to J2 only via `out["joints"]`. Loss formula `0.5*L(J1) + 1.0*L(J2) + args.lambda_bone * l_bone` correct. `lambda_bone=0.1` in config.py. `BODY_EDGES` computed from `SMPLX_SKELETON` with `a < 22 and b < 22`. No model.py changes (correct). No hardcoded values.

---

## idea019/design002 — Kinematic-Chain Soft Self-Attention Bias in Refinement Pass (Axis A2)

**Verdict: APPROVED**

`_build_kin_bias` BFS function correct (hop weights 1.0/0.5/0.25, body_n=22). `kin_bias` registered as non-trainable buffer (70,70). `kin_bias_scale` as `nn.Parameter(torch.zeros(1))` (init=0.0). Manual pass-2 decoder loop injects `tgt_mask=bias` per layer; final norm guard correct. `kin_bias_scale` auto-included in head_params. Loss unchanged from baseline. Config has `kin_bias_max_hops=3` and `kin_bias_scale_init=0.0`.

---

## idea019/design003 — Left-Right Symmetry Loss (Axis B1)

**Verdict: APPROVED**

`symmetry_loss` applied to J2 only. `SYM_PAIRS` 6 pairs all within BODY_IDX (0-21). Loss formula `0.5*L(J1) + 1.0*L(J2) + args.lambda_sym * l_sym` correct. `lambda_sym=0.05` in config.py. No model.py changes (correct). No hardcoded values.

---

## idea019/design004 — Joint-Group Query Initialization in Refinement Pass (Axis B2)

**Verdict: APPROVED**

`group_emb = nn.Embedding(4, 384)` zero-initialized. `joint_group_ids` buffer (70,) correctly assigns body joints to groups 0-2 and non-body joints (22-69 minus eyes at 23,24) to group 3. Group delta added to `queries2` before pass 2 with `.unsqueeze(0)` broadcast. `group_emb` auto-included in head_params. Loss unchanged from baseline. Minor: indices 23,24 assigned to group 0 (torso) rather than group 3 — cosmetic, non-functional, inherited from design. Config has `group_emb_init=0.0` and `num_joint_groups=4`.

---

## idea019/design005 — Combined Anatomical Priors: Bone-Length + Symmetry + Kinematic Bias (Axis B3)

**Verdict: APPROVED**

Combines designs 001+002+003 correctly. `_build_kin_bias` with module-level `import collections` (cleaner than design pseudocode). `kin_bias` buffer + `kin_bias_scale` scalar in model.py. Manual pass-2 loop with `tgt_mask=bias`. `bone_length_loss` + `symmetry_loss` in train.py. Combined loss formula `0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss + 0.05*sym_loss` correct. Config has `lambda_bone=0.1`, `lambda_sym=0.05`, `kin_bias_max_hops=3`, `kin_bias_scale_init=0.0`. No issues.
