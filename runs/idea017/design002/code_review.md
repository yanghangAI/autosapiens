# Code Review — idea017/design002
**Design:** Cross-Frame Memory Attention (2-frame, both trainable, gradient checkpointing)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly implements two-frame cross-attention with gradient checkpointing on both backbone passes and concatenated 1920-token memory. The shared backbone, concatenated memory, LLRD structure, and single-frame validation fallback are all correctly implemented.

## Architecture Check (model.py)

- Single `SapiensBackboneRGBD` instance (in_channels=4). Correct — backbone is shared between both time steps.
- `SapiensPose3D.forward(x_t, x_prev=None)` accepts optional second frame.
- Both backbone calls are gradient-checkpointed: `ckpt.checkpoint(self.backbone, x_t, use_reentrant=False)` and same for `x_prev`. Matches design spec.
- `Pose3DHead.forward(feat_t, feat_prev=None)`: computes `mem_t` and `mem_prev` via shared `input_proj`, then `memory = cat([mem_prev, mem_t], dim=1)` → shape `(B, 1920, 384)`. Correct.
- Single-frame fallback: `if feat_prev is None: memory = mem_t`. Correct for validation.
- Head output unchanged (joints, pelvis_depth, pelvis_uv). Correct.

## LLRD / Optimizer

- Single backbone module → LLRD groups identical to idea014/design003. No changes needed. Correct.

## Dataloader (train.py)

- `TemporalBedlamDataset` same pattern as design001 (past_idx = max(0, frame_idx-1)).
- Training loop: `x_prev = cat([rgb_prev, depth_prev], dim=1)` → `model(x_t, x_prev)`. Correct.
- Validation: `model(x_t, None)` — single-frame fallback. Correct.

## Config Check (config.py)

- `temporal_mode="cross_attn_both_trainable", use_grad_ckpt=True` present.
- `in_channels=4`. Correct — backbone takes 4-channel inputs, prev frame fed separately.
- All inherited HPs unchanged.

## Memory Concern

- Design requires Builder to verify with 1-step dry run (estimated 9-10 GB). Fallback: batch=2/accum=16. The test metrics show epoch time = 73s vs ~33s for design001 — confirms two backbone passes are running. OOM did not occur in the 2-epoch test run.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 1409mm, epoch 2 = 863mm. Large drop between epochs. Pelvis drops from 839mm to 401mm. Strong convergence signal. Epoch time 73s reflects the double backbone cost. No OOM. Healthy profile.

## Issues

None identified.
