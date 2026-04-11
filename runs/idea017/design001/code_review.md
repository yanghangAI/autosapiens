# Code Review — idea017/design001
**Design:** Delta-Input Channel Stacking (8-channel, single backbone pass)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly widens the patch_embed to 8 channels and implements a `TemporalBedlamDataset` subclass that fetches the past frame using the same crop bbox. The DepthBucketPE is correctly left unchanged (depth channel still at index 3). All design requirements are met.

## Architecture Check (model.py)

- `SapiensBackboneRGBD` accepts `in_channels: int = 4` parameter and passes it to `VisionTransformer(in_channels=in_channels)`. For design001, `in_channels=8`. Correct.
- `load_sapiens_pretrained` performs the 3→4 expansion as baseline, then a second 4→8 expansion: `w_8ch = cat([w_4ch, w_4ch.mean(dim=1,keepdim=True).expand(-1,4,-1,-1)], dim=1)`. Matches design spec (new 4 channels = mean of original 4 channels). Correct.
- `DepthBucketPE` reads `depth_ch = x[:,3:4,:,:]`. With 8-channel input `[RGB_t(0:3), D_t(3), RGB_t-5(4:7), D_t-5(7)]`, index 3 is still the centre-frame depth. Unchanged and correct.
- Head unchanged from baseline. Correct.

## Dataloader (train.py)

- `TemporalBedlamDataset` subclasses `BedlamFrameDataset`. Overrides `__getitem__` to fetch `past_idx = max(0, frame_idx - 1)` (dataset-index space = FRAME_STRIDE=5 raw frames). Correct boundary clamping.
- Uses the same crop bbox from the centre frame to crop the past frame via `_crop_frame_with_bbox`. Same crop logic replicated. Correct per design spec (do NOT recompute CropPerson per-frame).
- Normalizes `rgb_prev` and `depth_prev` the same way as centre frame (same mean/std, same depth_max normalization). Correct.
- Exposes `sample["rgb_prev"]` and `sample["depth_prev"]` as tensors. Correct.
- In `train_one_epoch`: `x = torch.cat([rgb, depth, rgb_prev, depth_prev], dim=1)` → shape `(B, 8, H, W)`. Correct.

## Config Check (config.py)

- `in_channels=8, temporal_mode="stack"` present.
- `output_dir` correct.
- All inherited HPs unchanged.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test: val_mpjpe_body epoch 1 = 879mm, epoch 2 = 800mm. Decreasing trend. Pelvis error drops significantly from 1923mm to 458mm. This is consistent with the temporal channel providing depth disambiguation cues that converge quickly. Healthy profile.

## Issues

None identified.
