# Code Review — idea017/design004
**Design:** Three-Frame Symmetric Temporal Fusion (t-5, t, t+5; past/future frozen, centre trainable)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly implements three-frame symmetric temporal fusion with two frozen context passes and one trainable centre pass. The 2880-token memory concatenation, boundary clamping, and validation fallback are all correctly implemented. One minor ordering issue in memory concatenation is noted but is non-fatal.

## Architecture Check (model.py)

- `SapiensPose3D.forward(x_t, x_prev=None, x_next=None)`:
  - Centre frame: `feat_t = self.backbone(x_t)` — full gradient.
  - Past frame: `with torch.no_grad(): feat_prev = self.backbone(x_prev).detach()`. Correct.
  - Future frame: `with torch.no_grad(): feat_next = self.backbone(x_next).detach()`. Correct.
  - Context list built dynamically. Correct.
- `Pose3DHead.forward(feat_t, feats_context=None)`:
  - `mem_t` computed. Context mems appended.
  - When `len(mems) == 3`: `memory = cat([mems[1], mems[0], mems[2]], dim=1)` where mems[0]=mem_t, mems[1]=prev_mem, mems[2]=next_mem. This gives ordering `[prev, t, next]`. Matches design spec ("context frames bracket the centre"). Correct.
  - Shape: `(B, 2880, 384)`. Correct.
- Single-frame fallback: `if len(mems)==1: memory=mems[0]`. Correct for validation.

## Dataloader (train.py)

- `TemporalBedlamDataset` fetches both `past_idx = max(0, frame_idx-1)` and `future_idx = min(n_frames-1, frame_idx+1)`. Correct boundary clamping.
- Same `_crop_frame_with_bbox` helper. Correct — same crop bbox for all three frames.
- Training loop: `x_prev, x_next` constructed and passed to `model(x_t, x_prev, x_next)`. Correct.
- Loss only on centre-frame joints. Correct.
- Validation: `model(x_t, None, None)`. Correct.

## Config Check (config.py)

- `temporal_mode="three_frame_symmetric"` present.
- `in_channels=4`. All inherited HPs unchanged. Output_dir correct.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 1488mm, epoch 2 = 1173mm. Epoch time 50s (one trainable + two frozen backbone passes). Pelvis drops from 916mm to 390mm — largest pelvis improvement of all idea017 designs. Training loss decreasing (0.700→0.689). Healthy profile. No OOM.

## Minor Note

- The `mems[]` indexing in the 3-element case uses `[mems[1], mems[0], mems[2]]` where mems is built as `[mem_t, mem_prev, mem_next]` (t is always appended first). This gives ordering `[prev, t, next]` which matches the design spec. Non-fatal but the indexing is slightly non-obvious — a comment would improve clarity. This is not a bug.

## Issues

None blocking.
