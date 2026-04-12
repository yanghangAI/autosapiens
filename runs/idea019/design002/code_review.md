# Code Review — idea019/design002
## Kinematic-Chain Soft Self-Attention Bias in Refinement Pass (Axis A2)

**Design_ID:** idea019/design002
**Verdict: APPROVED**

---

## Summary

The implementation correctly adds the kinematic self-attention bias to the second decoder pass only. The BFS hop-distance computation, buffer registration, learnable scalar, and manual decoder layer loop all match the design specification precisely. Config values are in config.py, not hardcoded. train.py loss is unchanged from baseline.

---

## config.py

All required fields present and correct:
- `kin_bias_max_hops = 3` — matches design spec
- `kin_bias_scale_init = 0.0` — matches design spec (informational)
- All baseline hyperparameters preserved: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `epochs=20`, `warmup_epochs=3`, `llrd_gamma=0.90`, `unfreeze_epoch=5`, `weight_decay=0.3`, `grad_clip=1.0`, `lambda_depth=0.1`, `lambda_uv=0.2`, `num_depth_bins=16`, `lr_head=1e-4`, `lr_depth_pe=1e-4`, `base_lr_backbone=1e-4`
- No extra loss weights needed (loss is unchanged); no values hardcoded in train.py

---

## model.py

Architecture changes match design spec:

1. `_build_kin_bias` function at module level:
   - BFS over `SMPLX_SKELETON` adjacency restricted to `a < body_n and b < body_n` (body_n=22)
   - Hop weights: hop-1 → 1.0, hop-2 → 0.5, hop-3 → 0.25 — matches design
   - Returns (70, 70) float tensor with zeros for non-body joint pairs
   - Critical invariant satisfied: no row is all-finite-negative; bias only adds positive values
   - No -inf entries possible — non-body rows/cols remain exactly zero

2. In `Pose3DHead.__init__`:
   - `kin_bias = _build_kin_bias(num_joints, SMPLX_SKELETON)` called at init
   - `self.register_buffer("kin_bias", kin_bias)` — non-trainable buffer, correct
   - `self.kin_bias_scale = nn.Parameter(torch.zeros(1))` — learnable scalar, init=0.0, correct
   - Both registered before `_init_weights()` call

3. In `Pose3DHead.forward`:
   - Pass 1 uses `self.decoder(queries, memory)` unmodified — correct
   - Refinement injection: `queries2 = out1 + self.refine_mlp(J1)` — preserved from baseline
   - Pass 2 uses manual layer loop:
     ```python
     bias = self.kin_bias_scale * self.kin_bias   # (70, 70)
     out2 = queries2
     for layer in self.decoder.layers:
         out2 = layer(out2, memory, tgt_mask=bias)
     if self.decoder.norm is not None:
         out2 = self.decoder.norm(out2)
     ```
   - This is correct: `norm_first=True` TransformerDecoderLayer accepts additive `tgt_mask`
   - Final norm guard (`if self.decoder.norm is not None`) is correct — equivalent to `self.decoder(queries2, memory)` but with bias injection
   - `J2 = self.joints_out2(out2)` — uses second output head, correct
   - `pelvis_token = out2[:, 0, :]` — pelvis from refined pass, correct

4. Optimizer group: `kin_bias_scale` is a parameter of `model.head`, automatically captured in the `head_params` group (LR=1e-4, weight_decay=0.3) — correct, no manual changes to train.py needed

---

## train.py

Loss is unchanged from baseline (0.5*L(J1) + 1.0*L(J2)):
```
l_pose = 0.5 * l_pose1 + 1.0 * l_pose2
```
Correct — no new loss terms for design002.

LLRD, freeze/unfreeze, cosine schedule all preserved unchanged.

---

## Issues

None. The implementation is technically correct and matches the design. The `tgt_mask` shape (70, 70) broadcasts correctly over batch and heads in PyTorch's `nn.MultiheadAttention` with `batch_first=True`. Starting with `kin_bias_scale=0.0` ensures training begins identical to the baseline.
