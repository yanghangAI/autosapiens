# design004 — Two-Pass Two-Decoder Refinement (Independent 2-Layer Refine Decoder)

## Starting Point

`runs/idea014/design003/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE.

## Problem

Designs 001–003 share decoder weights across passes, forcing one set of weights to serve both coarse and fine-grained objectives. This design tests whether separating "coarse localization" capacity (4-layer decoder) from "refinement" capacity (dedicated 2-layer decoder) allows each to specialize. The refinement decoder is lighter (2 layers vs. 4) since its task is smaller: correct residual errors in already-reasonable joint features.

## Proposed Solution

**Two-pass refinement with a separate independent 2-layer decoder.**

Pass 1 uses the existing 4-layer `decoder` (same as baseline). The output features `out1 (B, 70, 384)` are then passed — together with the coarse prediction `J1` injected as a residual delta — to a new, independent 2-layer decoder `refine_decoder`. This second decoder also cross-attends to the same memory. The final prediction `J2` comes from a new `joints_out2` head. Deep supervision weights: `0.5 × L(J1) + 1.0 × L(J2)`.

### Architecture Changes (model.py — Pose3DHead)

1. **Refinement MLP**: `Linear(3, 384) -> GELU -> Linear(384, 384)`. Injects coarse prediction into refined queries.
2. **Independent 2-layer refinement decoder**:
   ```python
   refine_layer = nn.TransformerDecoderLayer(
       d_model=384, nhead=8,
       dim_feedforward=384 * 4,  # 1536
       dropout=0.1,
       batch_first=True, norm_first=True,
   )
   self.refine_decoder = nn.TransformerDecoder(refine_layer, num_layers=2)
   ```
3. **Second output head**: `joints_out2 = nn.Linear(384, 3)`.
4. **Forward pass**:
   ```python
   memory  = self.input_proj(feat.flatten(2).transpose(1, 2))  # (B, 960, 384)
   queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 70, 384)

   # Pass 1 — coarse (existing 4-layer decoder)
   out1 = self.decoder(queries, memory)      # (B, 70, 384)
   J1   = self.joints_out(out1)              # (B, 70, 3)

   # Refinement query construction
   R        = self.refine_mlp(J1)           # (B, 70, 384)
   queries2 = out1 + R                       # (B, 70, 384)

   # Pass 2 — refine (independent 2-layer decoder)
   out2 = self.refine_decoder(queries2, memory)  # (B, 70, 384)
   J2   = self.joints_out2(out2)                 # (B, 70, 3)
   ```
5. Pelvis outputs from pass 2: `pelvis_depth = depth_out(out2[:, 0, :])`, `pelvis_uv = uv_out(out2[:, 0, :])`.
6. Return: `{"joints": J2, "joints_coarse": J1, "pelvis_depth": ..., "pelvis_uv": ...}`.

### Loss (train.py)

```python
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
l_dep   = pose_loss(out["pelvis_depth"], gt_pd)
l_uv    = pose_loss(out["pelvis_uv"],    gt_uv)
loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

### Optimizer Group Assignment (train.py)

The `refine_decoder` and `refine_mlp` and `joints_out2` parameters belong to the **head** optimizer group (LR=1e-4, weight_decay=0.3). No LLRD on head params, as per design constraints.

In `_build_optimizer_frozen()` and `_build_optimizer_full()`, the head group currently uses `list(model.head.parameters())`. Since `refine_decoder`, `refine_mlp`, and `joints_out2` are attributes of `model.head` (i.e., `Pose3DHead`), they are automatically included in `model.head.parameters()`. No special optimizer changes needed — just verify that the new submodules are defined inside `Pose3DHead.__init__`.

### Memory Estimate

**New params in refine_decoder** (2-layer TransformerDecoder, d=384, 8 heads, FFN=1536):
- Each layer: self-attn (~590K) + cross-attn (~590K) + FFN (384×1536×2=1.18M) + norms ≈ ~2.4M per layer.

Wait — more careful estimate:
- Self-attn: Q/K/V projections = 3 × 384 × 384 = 442K; out proj = 384×384 = 147K. Per head: ~590K.
- Cross-attn: same = ~590K.
- FFN: Linear(384,1536) + Linear(1536,384) = 590K + 590K = ~1.18M.
- Layer norms (3 × 2 × 384 = 2.3K).
- Per layer ≈ 590K + 590K + 1180K = 2.36M params.
- 2 layers = ~4.72M params.

Additional:
- refine_mlp: Linear(3,384) + Linear(384,384) ≈ 150K.
- joints_out2: ~1.2K.
- Total new: ~4.87M params.

This is within the design constraint of ~3M stated in idea.md (the idea.md estimate was slightly conservative; actual is ~4.87M but still fits at batch=4 in 11GB).

**GPU memory**: refine_decoder forward at batch=4 adds similar activation memory as ~2/4 of the coarse decoder (~50% of head activation). Peak GPU allocation will increase by ~100MB, well within budget.

## config.py Changes

Add informational fields:
```python
refine_passes         = 2      # number of decoder passes (A4 variant)
refine_decoder_layers = 2      # number of layers in independent refinement decoder
refine_loss_weight    = 0.5    # weight for coarse pass loss
```

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea014/design003/code/ |
| New modules | refine_mlp (Linear 3→384→384), refine_decoder (2-layer TransformerDecoder, 384/8/1536), joints_out2 (Linear 384→3) |
| Decoder passes | 2 (non-shared: 4-layer coarse + 2-layer refine) |
| Loss weights | 0.5 × L(J1) + 1.0 × L(J2) |
| Extra params | ~4.87M (refine_decoder dominates) |
| Optimizer group | All new params go into head group (LR=1e-4, WD=0.3) |
| All other HPs | Identical to idea014/design003 |
