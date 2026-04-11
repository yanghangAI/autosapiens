# Review: idea017/design001 — Delta-Input Channel Stacking (8-channel, single backbone pass)

**Design_ID:** idea017/design001  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Cheapest temporal fusion: the dataloader fetches the adjacent past frame at `t-5` (clamped at boundaries) and concatenates it with the centre frame along channels: `[RGB_t, D_t, RGB_{t-5}, D_{t-5}]` → 8-channel input. The backbone `patch_embed.projection` is widened from `Conv2d(4,768,...)` to `Conv2d(8,768,...)`. The 4 new channel weights are initialized as the mean of the corresponding 4 original channel weights (mirroring the RGB→RGBD expansion). Single backbone forward pass. Ground truth = centre-frame joints only. All else unchanged.

---

## Evaluation

### 1. Fidelity to idea.md Axis B1

- **8-channel input:** `[RGB_t(0:3), D_t(3), RGB_{t-5}(4:7), D_{t-5}(7)]` — correct channel ordering.
- **patch_embed widening:** `w_8ch = cat([w_4ch, mean(w_4ch,dim=1).expand(-1,4,-1,-1)], dim=1)`. Mean initialization for temporal channels — matches spec exactly.
- **DepthBucketPE:** Still reads `x[:,3:4,...]` for depth — unambiguously correct since centre-frame depth is at index 3. Explicitly noted.
- **Dataloader:** `past_frame_idx = max(0, frame_idx - 1)` in dataset-index space (= `t - FRAME_STRIDE*1` raw frames). Same crop bbox applied to past frame. Correct.
- **Single backbone forward:** Correct. No extra compute.
- **GT = centre-frame joints:** Correct.

### 2. Dataloader Implementation

The design correctly specifies:
- Fetch past-frame RGB and depth using `_read_frame` / `_read_depth`.
- Apply the same crop bbox (computed from centre frame) — do NOT recompute CropPerson.
- Expose as `sample["rgb_prev"]` and `sample["depth_prev"]` BEFORE calling transforms.
- Manual normalization of prev-frame tensors (standard transforms will process only `rgb`/`depth` fields).

**Implementation note flagged:** The design recommends subclassing `BedlamFrameDataset` or wrapping it as `TemporalBedlamDataset`. This is the correct pattern to avoid modifying infra.py.

### 3. Architecture Feasibility

- patch_embed weight doubles in channel dim (trivial memory addition).
- Single backbone forward, same FLOPs as baseline for ViT layers.
- Peak memory: ~identical to idea014/design003 (~6-7 GB). Confirmed within 11 GB.

### 4. Hyperparameter Completeness

New config fields: `in_channels=8`, `temporal_mode="stack"`. All required hyperparameters inherited. Complete.

### 5. Mathematical Correctness

- Weight initialization `w_4ch.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)` is the mean of 4 channels expanded to 4 new channels — identical to the `3→4` RGBD expansion already in the codebase. Correct.
- LLRD structure unchanged — single backbone, single optimizer. Correct.

### 6. Constraint Adherence

- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: fixed.
- Continuous sqrt depth PE, wide head (384, 8, 4): unchanged.
- infra.py constants: not modified.
- Transforms (SubtractRoot, CropPerson, ToTensor): not modified on the main pipeline. Past-frame is handled separately via manual cropping.

---

## Issues Found

**Minor:** The design does not specify exactly how `rgb_prev` and `depth_prev` are normalized (what normalisation constants, whether they use the same ImageNet mean/std). The Builder should use identical normalisation to the centre frame (same mean/std). This is implied but not stated explicitly.

No fatal issues.

---

## Verdict: APPROVED
