# design001 — Two-Pass Shared-Decoder Refinement (Query Injection)

## Starting Point

`runs/idea014/design003/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE.

## Problem

The baseline (idea014/design003) performs a single 4-layer transformer decoder pass over the 70 joint queries and emits one 3D prediction. The residual error (val_mpjpe_body=106.85 mm) is concentrated on body joint localization. A coarse-to-fine cascade — where each joint query is updated by its own coarse prediction before a second decoder pass — should allow the attention mechanism to re-attend near the predicted joint location, reducing localization error without changing width or depth.

## Proposed Solution

**Two-pass iterative refinement with query injection (shared decoder weights).**

After the first decoder pass produces coarse predictions `J1 (B, 70, 3)`, a small refinement MLP projects `J1` into the query embedding space and adds it to the original query features. The same 4-layer decoder is then re-run on the updated queries over the same memory. A second Linear(384→3) head produces refined predictions `J2 (B, 70, 3)`. The final prediction is `J2`. Deep supervision is applied on `J1` with weight 0.5 to stabilize early training.

### Architecture Changes (model.py — Pose3DHead)

1. **Refinement MLP**: `Linear(3, 384) -> GELU -> Linear(384, 384)`. Input: coarse `J1 (B, 70, 3)`. Output: refinement delta `R (B, 70, 384)`.
2. **Second output head**: `joints_out2 = nn.Linear(384, 3)`.
3. **Forward pass**:
   ```
   memory  = input_proj(feat)           # (B, 960, 384)
   queries = joint_queries.weight[None] # (B, 70, 384)
   out1    = decoder(queries, memory)   # (B, 70, 384) — first pass
   J1      = joints_out(out1)           # (B, 70, 3)  — coarse prediction
   R       = refine_mlp(J1)            # (B, 70, 384) — refinement delta
   queries2 = out1 + R                  # (B, 70, 384) — conditioned queries
   out2    = decoder(queries2, memory)  # (B, 70, 384) — second pass (shared weights)
   J2      = joints_out2(out2)          # (B, 70, 3)  — refined prediction
   ```
4. The `pelvis_depth` and `pelvis_uv` outputs are derived from `out2[:, 0, :]` (refined pelvis token).
5. Return dict: `{"joints": J2, "joints_coarse": J1, "pelvis_depth": ..., "pelvis_uv": ...}`.

### Loss (train.py)

```python
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])  # coarse supervision
l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])          # refined supervision
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
l_dep   = pose_loss(out["pelvis_depth"], gt_pd)
l_uv    = pose_loss(out["pelvis_uv"],    gt_uv)
loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

Metrics (mpjpe, pelvis_abs_error) are computed on the final refined prediction `out["joints"]`.

### Memory Estimate

- Refinement MLP: Linear(3,384) + Linear(384,384) ≈ 150K params.
- joints_out2: Linear(384,3) ≈ 1.2K params.
- Total new params: ~151K. Fits comfortably in 11GB at batch=4.

## config.py Changes

Add the following field (no other config changes):
```python
refine_passes = 2          # number of decoder passes (A1 variant)
refine_loss_weight = 0.5   # weight for coarse pass loss
```

These fields are informational for the Builder; the actual implementation is in model.py and train.py.

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea014/design003/code/ |
| New modules | refine_mlp (Linear 3→384→384), joints_out2 (Linear 384→3) |
| Decoder passes | 2 (shared weights) |
| Loss weights | 0.5 × L(J1) + 1.0 × L(J2) |
| Extra params | ~151K |
| All other HPs | Identical to idea014/design003 |
