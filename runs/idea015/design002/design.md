# design002 — Two-Pass Shared-Decoder Refinement (Cross-Attention Gaussian Bias from J1)

## Starting Point

`runs/idea014/design003/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE.

## Problem

Design001 refines queries by injecting the coarse prediction back into the query space. This design takes a different approach: instead of modifying queries, we bias the *cross-attention logits* of the second decoder pass using a Gaussian centered on the projected `(u, v)` location of each coarse joint prediction on the memory grid (40×24). This soft spatial prior steers attention without hard masking, building on the philosophy of the continuous depth PE from idea005/idea008.

## Proposed Solution

**Two-pass shared-decoder refinement with Gaussian cross-attention bias (memory-gated attention).**

Pass 1 is identical to the baseline single-pass. After obtaining `J1 (B, 70, 3)`, we compute a per-joint Gaussian bias over the memory grid (40×24 = 960 patches) and inject it as an additive bias to the cross-attention logits of pass 2. A learnable scalar `attn_bias_scale` (initialized to 0.0) scales the Gaussian, so at step 0 the second pass is numerically identical to the first.

### Architecture Changes (model.py — Pose3DHead)

1. **Learnable scalar**: `self.attn_bias_scale = nn.Parameter(torch.zeros(1))`.
2. **Second output head**: `joints_out2 = nn.Linear(384, 3)`.
3. **Gaussian bias computation** (inline in forward):

   **UV projection fallback** (no camera intrinsics available in head):
   The data pipeline provides `pelvis_uv` in `[0,1]` normalized image coords via `out["pelvis_uv"]` from the first pass pelvis token. However, individual body joint UV coords are not directly provided — we use a simple approximation:
   ```
   # J1: (B, 70, 3) — root-relative 3D predictions
   # pelvis_uv_pred: (B, 2) from out1 pelvis token via self.uv_out — normalized [0,1]
   z_clamped = J1[:, :, 2:3].clamp(min=0.1)                    # (B, 70, 1)
   uv_norm = pelvis_uv_pred.unsqueeze(1) + 0.5 * (J1[:, :, :2] / z_clamped)  # (B, 70, 2)
   uv_norm = uv_norm.clamp(0.0, 1.0)                           # keep in [0,1]
   ```
   Convert to grid coords: `u_grid = uv_norm[:,:,0] * (W_tok - 1)` and `v_grid = uv_norm[:,:,1] * (H_tok - 1)` where `H_tok=40, W_tok=24` (note: memory is flattened H×W in row-major, so grid index = v * W_tok + u).

   **Gaussian bias** (shape `(B, 70, 960)` — one bias value per joint per memory patch):
   ```python
   # Precompute grid coordinates: (960, 2) — (row, col) for each patch
   rows = torch.arange(H_tok).unsqueeze(1).expand(H_tok, W_tok).reshape(-1)  # (960,)
   cols = torch.arange(W_tok).unsqueeze(0).expand(H_tok, W_tok).reshape(-1)  # (960,)
   # grid: (1, 1, 960, 2) — broadcast over B and joints
   grid = torch.stack([rows, cols], dim=-1).float().unsqueeze(0).unsqueeze(0)

   # Joint projected positions: (B, 70, 1, 2)
   mu = torch.stack([v_grid, u_grid], dim=-1).unsqueeze(2)

   # Gaussian: sigma=2.0 patches (fixed)
   sigma = 2.0
   dist2 = ((grid - mu) ** 2).sum(dim=-1)          # (B, 70, 960)
   gauss_bias = -dist2 / (2 * sigma ** 2)           # log-Gaussian, additive to logits

   # Scale by learnable scalar (init=0 → no bias at step 0)
   gauss_bias = gauss_bias * torch.sigmoid(self.attn_bias_scale) * 10.0
   # Clamp to avoid -inf; minimum is -1e4 (fully suppressed, never -inf)
   gauss_bias = gauss_bias.clamp(min=-1e4)          # (B, 70, 960)
   ```

4. **Injecting the bias into pass 2**: The standard `nn.TransformerDecoder` does not expose a per-query per-key cross-attention additive bias argument. We implement pass 2 by manually calling each decoder layer with the bias injected via `memory_key_padding_mask` not available — instead we use the `tgt_mask` / `memory_mask` argument.

   **Implementation approach**: Subclass `nn.TransformerDecoderLayer` or manually loop over layers. Preferred approach: loop over `self.decoder.layers` manually and call `layer.multihead_attn.forward` with `attn_mask` set to the Gaussian bias reshaped to `(B * num_heads, 70, 960)`.

   Specifically:
   ```python
   # Pass 1 (standard)
   out1 = self.decoder(queries, memory)

   # Compute Gaussian bias for pass 2
   # ... (as above)
   # bias shape: (B, 70, 960) → expand over heads: (B*num_heads, 70, 960)
   attn_bias_expanded = gauss_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
   attn_bias_expanded = attn_bias_expanded.reshape(B * self.num_heads, 70, 960)

   # Pass 2: manual layer loop
   tgt = queries  # re-start from original queries (same as pass 1 start)
   for layer in self.decoder.layers:
       # Self-attention (no mask)
       tgt2 = layer.norm1(tgt)
       tgt2 = layer.self_attn(tgt2, tgt2, tgt2)[0]
       tgt = tgt + layer.dropout1(tgt2)
       # Cross-attention with additive Gaussian bias
       tgt2 = layer.norm2(tgt)
       tgt2, _ = layer.multihead_attn(
           tgt2, memory, memory,
           attn_mask=attn_bias_expanded,
           need_weights=False,
       )
       tgt = tgt + layer.dropout2(tgt2)
       # FFN
       tgt2 = layer.norm3(tgt)
       tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
       tgt = tgt + layer.dropout3(tgt2)
   out2 = tgt
   ```

   Note: `norm_first=True` is set in the baseline decoder layer. Adjust the layer forward loop accordingly (norms applied *before* sublayers, not after).

5. **norm_first=True corrected loop**:
   ```python
   for layer in self.decoder.layers:
       # Self-attention (norm first)
       tgt = tgt + layer.dropout1(layer.self_attn(layer.norm1(tgt), layer.norm1(tgt), layer.norm1(tgt))[0])
       # Cross-attention with bias (norm first)
       tgt2 = layer.norm2(tgt)
       ca_out, _ = layer.multihead_attn(tgt2, memory, memory,
                                         attn_mask=attn_bias_expanded, need_weights=False)
       tgt = tgt + layer.dropout2(ca_out)
       # FFN (norm first)
       tgt = tgt + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(tgt))))))
   out2 = tgt
   ```

6. Return dict: `{"joints": J2, "joints_coarse": J1, "pelvis_depth": depth_out(out2[:,0,:]), "pelvis_uv": uv_out(out2[:,0,:])}`.

### Loss (train.py)

Same as design001:
```python
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
l_dep   = pose_loss(out["pelvis_depth"], gt_pd)
l_uv    = pose_loss(out["pelvis_uv"],    gt_uv)
loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

### Memory Estimate

- `attn_bias_scale`: 1 scalar param.
- `joints_out2`: Linear(384,3) ≈ 1.2K params.
- Gaussian computation: pure tensor ops, no extra params.
- Total new params: ~1.2K + 1. Fits easily in 11GB at batch=4.

### UV Projection Fallback Note

As required by design constraints: `pelvis_uv` is predicted by the model's first-pass `uv_out` head (normalized [0,1] pelvis UV). It is available within the head's own forward pass as `pelvis_uv_pred = self.uv_out(out1[:, 0, :])`. This is used as the root anchor. The fallback (center of grid + joint offsets) is not needed since the model always predicts pelvis_uv.

## config.py Changes

Add the following informational fields:
```python
refine_passes        = 2      # number of decoder passes (A2 variant)
refine_loss_weight   = 0.5    # weight for coarse pass loss
attn_bias_sigma      = 2.0    # Gaussian sigma in patches (fixed)
# attn_bias_scale initialized to 0.0 in model (learnable)
```

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea014/design003/code/ |
| New modules | attn_bias_scale (scalar), joints_out2 (Linear 384→3) |
| Decoder passes | 2 (shared weights, manual layer loop in pass 2) |
| Bias | Additive Gaussian on cross-attention logits, sigma=2.0, init scale=0 |
| Loss weights | 0.5 × L(J1) + 1.0 × L(J2) |
| Extra params | ~1.2K |
| All other HPs | Identical to idea014/design003 |
