# design003 — Three-Pass Shared-Decoder Refinement (Deep Supervision 0.25/0.5/1.0)

## Starting Point

`runs/idea014/design003/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE.

## Problem

Design001 tests whether a second decoder pass improves over one. This design asks: does a *third* pass yield additional gains, or does the model overfit under only 20 epochs of training? We cascade the same 4-layer decoder three times with progressive deep supervision (weights 0.25, 0.5, 1.0) and weight sharing across all passes. The progressive weights down-weight early coarse stages and push the network to allocate most capacity to the final refined prediction.

## Proposed Solution

**Three-pass iterative refinement with shared decoder and query injection (progressive deep supervision).**

The same query-injection mechanism from design001 is extended to three passes:
- Pass 1 → `J1 (B, 70, 3)` via `joints_out` (shared head)
- Pass 2 → `J2 (B, 70, 3)` via `joints_out2` (dedicated head)
- Pass 3 → `J3 (B, 70, 3)` via `joints_out3` (dedicated head)

Final prediction returned as `out["joints"] = J3`. Deep supervision: `0.25 × L(J1) + 0.5 × L(J2) + 1.0 × L(J3)`.

### Architecture Changes (model.py — Pose3DHead)

1. **Refinement MLP** (shared across passes): `Linear(3, 384) -> GELU -> Linear(384, 384)`.
2. **Output heads**:
   - `joints_out` (existing) → used for J1.
   - `joints_out2 = nn.Linear(384, 3)` → used for J2.
   - `joints_out3 = nn.Linear(384, 3)` → used for J3.
3. **Forward pass**:
   ```python
   memory  = self.input_proj(feat.flatten(2).transpose(1, 2))  # (B, 960, 384)
   queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 70, 384)

   # Pass 1
   out1 = self.decoder(queries, memory)        # (B, 70, 384)
   J1   = self.joints_out(out1)                # (B, 70, 3)

   # Pass 2 (inject J1)
   R1       = self.refine_mlp(J1)             # (B, 70, 384)
   queries2 = out1 + R1                        # (B, 70, 384)
   out2     = self.decoder(queries2, memory)   # (B, 70, 384)
   J2       = self.joints_out2(out2)           # (B, 70, 3)

   # Pass 3 (inject J2)
   R2       = self.refine_mlp(J2)             # (B, 70, 384) — same MLP, shared
   queries3 = out2 + R2                        # (B, 70, 384)
   out3     = self.decoder(queries3, memory)   # (B, 70, 384)
   J3       = self.joints_out3(out3)           # (B, 70, 3)
   ```
4. Pelvis outputs from the final pass: `pelvis_depth = depth_out(out3[:, 0, :])`, `pelvis_uv = uv_out(out3[:, 0, :])`.
5. Return: `{"joints": J3, "joints_pass1": J1, "joints_pass2": J2, "pelvis_depth": ..., "pelvis_uv": ...}`.

**Note on weight sharing**: `self.refine_mlp` is a *single* module used in both pass 1→2 and pass 2→3 transitions. Similarly, `self.decoder` is reused for all three passes. This keeps parameter count bounded.

### Loss (train.py)

```python
l_pose1 = pose_loss(out["joints_pass1"][:, BODY_IDX], joints[:, BODY_IDX])  # coarse
l_pose2 = pose_loss(out["joints_pass2"][:, BODY_IDX], joints[:, BODY_IDX])  # intermediate
l_pose3 = pose_loss(out["joints"][:, BODY_IDX],        joints[:, BODY_IDX]) # final
l_pose  = 0.25 * l_pose1 + 0.5 * l_pose2 + 1.0 * l_pose3
l_dep   = pose_loss(out["pelvis_depth"], gt_pd)
l_uv    = pose_loss(out["pelvis_uv"],    gt_uv)
loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

Metrics computed on final prediction `out["joints"]` (= J3).

### Iteration Logger

In `iter_logger.log`, add `loss_pose1`, `loss_pose2`, `loss_pose3` to track convergence of each stage.

### Memory Estimate

Three decoder passes at batch=4: roughly 3× the memory of one pass for activations. The baseline single-pass peak was well within 11GB; two extra passes (with grad stored) will increase activation memory by ~2× the head activation size. Head activations at batch=4: ~50MB per pass (estimated). Three passes = ~150MB extra vs. baseline ~75MB → total ~300MB for head activations. Well within budget.

New params:
- refine_mlp: Linear(3,384) + Linear(384,384) ≈ 150K.
- joints_out2: ~1.2K.
- joints_out3: ~1.2K.
- Total new params: ~152K.

## config.py Changes

Add informational fields:
```python
refine_passes        = 3      # number of decoder passes (A3 variant)
refine_loss_w1       = 0.25   # weight for pass-1 supervision
refine_loss_w2       = 0.50   # weight for pass-2 supervision
refine_loss_w3       = 1.00   # weight for pass-3 (final) supervision
```

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea014/design003/code/ |
| New modules | refine_mlp (shared, Linear 3→384→384), joints_out2, joints_out3 (Linear 384→3 each) |
| Decoder passes | 3 (all shared decoder weights) |
| Loss weights | 0.25 × L(J1) + 0.5 × L(J2) + 1.0 × L(J3) |
| Extra params | ~152K |
| All other HPs | Identical to idea014/design003 |
