# design001 — Delta-Input Channel Stacking (8-channel, single backbone pass)

## Starting Point

`runs/idea014/design003/code/`

## Summary

This is the cheapest possible temporal fusion. The dataloader is extended to fetch the adjacent past frame at `t-5` (clamped to 0 at sequence boundaries). The centre frame and the past frame are concatenated along the channel dimension to form an 8-channel input: `[RGB_t, D_t, RGB_{t-5}, D_{t-5}]`. The Sapiens backbone's `patch_embed.projection` is widened from `Conv2d(4, 768, ...)` to `Conv2d(8, 768, ...)`. The 4 new channel weights are initialised as the mean of the corresponding original 4 channel weights (mirroring the RGB→RGBD expansion already present in `load_sapiens_pretrained`). No second backbone forward pass is required. Ground truth targets remain the centre-frame joints only. Everything else (head, loss weights, LLRD, LR schedule) is unchanged from idea014/design003.

## Problem

Every prior experiment processes a single frame in isolation. Temporal smoothness is a strong prior in 3D pose estimation — in particular, pelvis depth ambiguity (96 mm residual error) should be resolvable with even coarse previous-frame context.

## Proposed Solution

Concatenate `[RGB_t, D_t, RGB_{t-5}, D_{t-5}]` as an 8-channel input. The backbone patch embed is widened; all other backbone weights (all 24 ViT layers, DepthBucketPE) are unchanged. The model remains a single forward pass, preserving the full LLRD optimizer structure. The intuition: the backbone's attention layers can learn to extract temporal motion cues from the combined channel signal (optical-flow-like deltas emerge via learned filters).

## Dataloader Changes

Extend `BedlamFrameDataset.__getitem__` (or subclass it) to:
1. Compute the past-frame index: `past_frame_idx = max(0, frame_idx - 1)`. (The dataset indexes at the stride-5 frame level, so subtracting 1 in *dataset index space* corresponds to going back FRAME_STRIDE=5 raw frames.)
2. Load the past-frame RGB via `_read_frame(folder_name, seq_name, past_frame_idx, label_path)` and past-frame depth via `_read_depth(npy_path, npz_path, past_frame_idx, label_path)`.
3. Apply **the same crop bbox** (already computed from the centre frame) to the past-frame RGB and depth — do NOT recompute `CropPerson` per-frame. Specifically: after the centre-frame `sample` has been built (dict with `rgb`, `depth`, `bbox`, etc.), manually crop the past-frame rgb/depth using the same pixel coordinates that `CropPerson` computes for the centre frame.
4. Expose the cropped+resized past-frame as `sample["rgb_prev"]` (H×W×3 uint8) and `sample["depth_prev"]` (H×W float32) BEFORE calling `self.transform`. The standard `SubtractRoot` and `ToTensor` transforms will process only the normal `rgb`/`depth` fields; an additional manual normalization step converts `rgb_prev` and `depth_prev` into tensors.
5. In `train.py`, construct `x = torch.cat([rgb, depth, rgb_prev, depth_prev], dim=1)` — shape `(B, 8, H, W)`.

Implementation note: the simplest approach is to subclass `BedlamFrameDataset` in `model.py` or a new `dataset.py` file within the design folder, overriding `__getitem__` to fetch the extra frame after the base class has determined the crop bbox. Alternatively, implement as a wrapper `TemporalBedlamDataset` that wraps the base dataset.

## Architecture Changes (model.py)

### patch_embed widening

In `load_sapiens_pretrained` (or in `SapiensBackboneRGBD.__init__`):
- Build the ViT with `in_channels=8` instead of `4`.
- The `patch_embed.projection` weight shape becomes `(768, 8, k, k)`.
- When loading the pretrained checkpoint, expand the patch embed: after the existing `3→4` expansion, perform a second `4→8` expansion:
  ```python
  w_4ch = remapped[pe_key]  # (768, 4, k, k) — already RGB+depth-expanded
  w_8ch = torch.cat([w_4ch, w_4ch.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)], dim=1)
  remapped[pe_key] = w_8ch  # (768, 8, k, k)
  ```
  The 4 new channel weights are the mean of the 4 original channel weights, giving a near-identity initialisation for the temporal channels.

### DepthBucketPE

Unchanged. It reads `depth_ch = x[:, 3:4, :, :]` (the centre-frame depth channel). This is still correct since the centre-frame depth is at index 3 in the 8-channel input.

Wait — with 8 channels `[RGB_t(0:3), D_t(3), RGB_{t-5}(4:7), D_{t-5}(7)]`, the depth index is still `3:4`. No change needed.

### Head and decoder

Unchanged from idea014/design003.

## config.py Fields

All inherited from idea014/design003. Only additions/changes:

```python
# Temporal context
in_channels = 8          # NEW: 8-channel input (RGB_t, D_t, RGB_t-5, D_t-5)
temporal_mode = "stack"  # NEW: informational only, for logging

# Everything else UNCHANGED:
arch             = "sapiens_0.3b"
head_hidden      = 384
head_num_heads   = 8
head_num_layers  = 4
head_dropout     = 0.1
drop_path        = 0.1
epochs           = 20
batch_size       = 4      # from BATCH_SIZE in infra.py
lr_backbone      = 1e-4
base_lr_backbone = 1e-4
llrd_gamma       = 0.90
unfreeze_epoch   = 5
lr_head          = 1e-4
lr_depth_pe      = 1e-4
weight_decay     = 0.3
warmup_epochs    = 3
grad_clip        = 1.0
accum_steps      = 8      # from ACCUM_STEPS in infra.py
amp              = False
lambda_depth     = 0.1
lambda_uv        = 0.2
num_depth_bins   = 16
```

## Memory Estimate

Single backbone forward pass with 8-channel input vs. 4-channel: patch_embed weight doubles in size (negligible), but the backbone computation is identical. Peak memory is essentially identical to idea014/design003 (~6-7 GB). Easily within the 11 GB budget.

## Expected Outcome

Modest improvement. The backbone must learn temporal cues purely via channel-space operations. If this alone reduces body MPJPE by 2-5 mm and pelvis by 2-4 mm, it validates the temporal signal; designs 002-004 then explore richer fusion.
