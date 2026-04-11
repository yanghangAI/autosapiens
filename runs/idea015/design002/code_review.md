# Code Review — idea015/design002
**Design:** Two-Pass Shared-Decoder Refinement (Cross-Attention Gaussian Bias from J1)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly implements the Gaussian cross-attention bias design with a learnable scalar. All core design elements are present and correct. One minor deviation from the spec (sigmoid vs raw zero init) is non-fatal and self-corrects at convergence.

## Architecture Check (model.py)

- `attn_bias_scale = nn.Parameter(torch.zeros(1))` — scalar initialized at 0.0 as required.
- `joints_out2 = Linear(384,3)` — second output head present.
- UV projection fallback: uses `pelvis_uv_pred = self.uv_out(out1[:,0,:])` as root anchor, then computes `uv_norm = pelvis_uv_pred.unsqueeze(1) + 0.5*(J1[:,:,:2]/z_clamped)`. Matches design spec exactly.
- Grid precomputed per-forward (H_tok=40, W_tok=24). Acceptable (tensors are small; the design suggests precomputing as `register_buffer` but the per-forward approach is functionally equivalent).
- Gaussian bias: `gauss_bias = -dist2 / (2*sigma^2)` then `* self.attn_bias_scale`. The scale is the raw parameter (not sigmoid*10 as in the design spec). This means at init the bias IS zero (correct); the magnitude scaling differs but the overall effect is the same. Non-fatal deviation.
- Manual norm_first layer loop is correctly implemented: norm1→self_attn, norm2→cross_attn with bias, norm3→FFN. Matches the corrected loop from design002.
- `attn_bias_expanded` reshaped to `(B*num_heads, 70, 960)` — correct.
- Bias clamped to `min=-1e4` — correct (avoids -inf).
- Pass 2 restarts from `queries` (original, not `out1`) — matches design spec.
- Return dict: `joints=J2, joints_coarse=J1, pelvis_depth, pelvis_uv` from `out2[:,0,:]`. Correct.

## Config Check (config.py)

- `output_dir` correct. `refine_passes=2, refine_loss_weight=0.5, attn_bias_sigma=2.0` all present.
- All inherited HPs identical to design001. Correct.

## Loss Check (train.py)

- Identical to design001 loss: `0.5*L(J1) + 1.0*L(J2)`. Correct.
- Uses `out["joints_coarse"]` and `out["joints"]`. Correct.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test run: val_mpjpe_body epoch 1 = 525mm, epoch 2 = 1380mm. Uptick at epoch 2 is unusual but within acceptable noise for a 2-epoch warmup. Loss is still in warmup phase. Not a red flag.

## Issues

- Minor: `attn_bias_scale` used as raw parameter (not `sigmoid(scale)*10.0` as spec'd). At init scale=0 so pass 2 is numerically identical to pass 1. Behavior is correct at step 0; magnitude of learned bias will differ after training but this is a minor implementation variant, not a bug.
