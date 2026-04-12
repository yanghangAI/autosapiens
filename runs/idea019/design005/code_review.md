# Code Review — idea019/design005
## Combined Anatomical Priors: Bone-Length Loss + Symmetry Loss + Kinematic Bias (Axis B3)

**Design_ID:** idea019/design005
**Verdict: APPROVED**

---

## Summary

The implementation correctly combines all three anatomical priors from designs 001, 002, and 003: bone-length loss (lambda_bone=0.1), symmetry loss (lambda_sym=0.05), and kinematic self-attention bias in the refinement decoder pass (kin_bias buffer + kin_bias_scale scalar). Config contains all four required loss/bias fields. All values read from config (none hardcoded). The combined loss formula matches the design exactly.

---

## config.py

All required fields present and correct:
- `lambda_bone = 0.1` — matches design spec
- `lambda_sym = 0.05` — matches design spec
- `kin_bias_max_hops = 3` — informational, matches design
- `kin_bias_scale_init = 0.0` — informational, matches design
- All baseline hyperparameters preserved: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `epochs=20`, `warmup_epochs=3`, `llrd_gamma=0.90`, `unfreeze_epoch=5`, `weight_decay=0.3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`, `num_depth_bins=16`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `base_lr_backbone=1e-4`
- All loss weights accessed as `args.lambda_bone`, `args.lambda_sym`, `args.lambda_depth`, `args.lambda_uv`

---

## model.py

Architecture changes combine design002's kinematic bias with no changes from designs 001/003:

1. `_build_kin_bias` function at module level — identical to design002:
   - BFS over SMPLX_SKELETON adjacency restricted to body joints (a < 22, b < 22)
   - Hop weights 1.0/0.5/0.25 for hops 1/2/3
   - Returns (70, 70) tensor, non-body rows/cols zero
   - No -inf entries possible

2. In `Pose3DHead.__init__`:
   - `kin_bias = _build_kin_bias(num_joints, SMPLX_SKELETON)` — correct
   - `self.register_buffer("kin_bias", kin_bias)` — non-trainable, correct
   - `self.kin_bias_scale = nn.Parameter(torch.zeros(1))` — init=0.0, correct

3. In `Pose3DHead.forward` — matches design005 spec:
   - Pass 1: `self.decoder(queries, memory)` — unchanged
   - Refinement: `queries2 = out1 + self.refine_mlp(J1)` — unchanged
   - Pass 2 with kinematic bias (manual loop):
     ```python
     bias = self.kin_bias_scale * self.kin_bias   # (70, 70)
     out2 = queries2
     for layer in self.decoder.layers:
         out2 = layer(out2, memory, tgt_mask=bias)
     if self.decoder.norm is not None:
         out2 = self.decoder.norm(out2)
     J2 = self.joints_out2(out2)
     ```
   - Final norm guard is correct
   - Return dict includes all four keys: `joints`, `joints_coarse`, `pelvis_depth`, `pelvis_uv`
   - `pelvis_token = out2[:, 0, :]` — pelvis from refined pass, correct

4. `kin_bias_scale` is part of `model.head` — auto-included in head_params optimizer group

---

## train.py

- `BODY_EDGES` computed at module level from `SMPLX_SKELETON` — correct
- `SYM_PAIRS` 6 pairs — identical to design003 spec
- `bone_length_loss` function — matches design001/005 spec exactly
- `symmetry_loss` function — matches design003/005 spec exactly
- Combined loss composition:
  ```python
  l_bone  = bone_length_loss(out["joints"], joints, BODY_EDGES)
  l_sym   = symmetry_loss(out["joints"], SYM_PAIRS)
  l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_bone * l_bone + args.lambda_sym * l_sym
  loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
  ```
  Matches design formula exactly: `0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2) + 0.05*sym_loss(J2)`
- Both `l_bone` and `l_sym` included in `del` statement at end of batch loop — no memory leak
- LLRD optimizer, freeze/unfreeze, cosine schedule all preserved unchanged

---

## Issues

**Cosmetic (non-fatal):** The `_build_kin_bias` function contains a redundant `import collections as _col` inside the function body (line `import collections as _col` is present in design005's design.md pseudocode but not present in the actual implementation — the module-level `import collections` is used instead). This is actually an improvement over the design pseudocode. No issue.

No other issues. The implementation is a clean union of the three individual designs and is correct.
