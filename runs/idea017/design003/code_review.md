# Code Review — idea017/design003
**Design:** Cross-Frame Memory Attention (2-frame, past frozen no_grad, centre trainable)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly implements the frozen-past-frame variant. Past-frame backbone runs inside `torch.no_grad()` with explicit `.detach()`. Only the centre-frame backbone participates in LLRD and backprop. Memory concatenation and single-frame validation fallback are correct.

## Architecture Check (model.py)

- `SapiensPose3D.forward(x_t, x_prev=None)`:
  - Centre frame: `feat_t = self.backbone(x_t)` — full gradient path.
  - Past frame: `with torch.no_grad(): feat_prev = self.backbone(x_prev); feat_prev = feat_prev.detach()`. Correct — both no_grad AND explicit detach as required by design.
  - No gradient checkpointing on either path (not needed since past frame has no grad). Correct.
- `Pose3DHead.forward(feat_t, feat_prev=None)`: `memory = cat([mem_prev, mem_t], dim=1)` → shape `(B, 1920, 384)`. Same as design002. Correct.
- Single-frame fallback: `if feat_prev is None: memory = mem_t`. Correct.

## LLRD / Optimizer

- Identical to idea014/design003 — single backbone module, LLRD only on centre-frame path (automatically, since no_grad prevents gradients through the past-frame pass). Correct.

## Dataloader (train.py)

- Same `_crop_frame_with_bbox` helper replicated (same crop bbox). Correct.
- `TemporalBedlamDataset` fetches past frame. Correct.
- Training loop: past frame fed as `x_prev`; loss computed only on centre-frame joints. Correct.
- Validation: `model(x_t, None)`. Correct.

## Config Check (config.py)

- `temporal_mode="cross_attn_past_frozen"` present.
- `in_channels=4`. All inherited HPs unchanged. Output_dir correct.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 1385mm, epoch 2 = 974mm. Epoch time 41s (between design001's 33s and design002's 73s — one frozen + one trainable backbone). Pelvis drops from 845mm to 482mm. Consistent with design003 design expectation. No OOM. Healthy profile.

## Issues

None identified.
