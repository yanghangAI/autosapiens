# idea017 -- Temporal Context via Adjacent-Frame Fusion

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea014/design003/code/`

Current SOTA: val_mpjpe_body = **106.85 mm**, val_pelvis = **96.73 mm**, val_mpjpe_weighted = **103.51 mm**. This idea introduces **temporal context** into the pipeline for the first time — exploiting the fact that the dataset is composed of *sequences* (FRAME_STRIDE=5) but every prior idea has treated frames independently.

## Concept

A fundamental limitation of the current pipeline: every prior experiment (idea001-014) processes a single frame in isolation, even though the dataset is sequential video. Temporal smoothness of pose is one of the strongest priors in 3D human pose estimation — methods like VideoPose3D, PoseFormer, MotionBERT all exploit multi-frame context to dramatically reduce per-frame error, especially on ambiguous depth and occluded joints.

The hypothesis: fusing features from ≥2 adjacent frames (t, t-k, t+k with k=FRAME_STRIDE=5) should reduce ambiguity in depth (particularly pelvis depth, which is the largest residual error source at 96 mm) and self-occlusion cases. The challenge is doing this inside the 11GB budget and 20-epoch proxy — running the 300M-param Sapiens backbone twice per training step is expensive.

This idea explores four fusion strategies at different depths in the architecture, ordered from cheapest to most expressive.

## Broader Reflection

### Strong prior results context

- **idea014/design003** (SOTA 106.85/96.73) — single-frame, direct regression. Serves as the baseline single-frame model against which temporal gains are measured.
- **idea008/design003** (pelvis 93.7 mm) — the strongest per-frame pelvis result. Pelvis error is the single largest component of the weighted MPJPE and is most amenable to temporal smoothing.
- **idea010** (multi-scale features) — tried *spatial* feature aggregation across backbone layers. This idea does the analogous *temporal* aggregation across frames — the unexplored axis.

### Patterns to avoid

- **idea002** (attention masking) — hard masks hurt. Temporal fusion here uses soft cross-attention only.
- **idea003** (curriculum weighting) — aggressive loss scheduling hurt. This idea uses fixed loss weights identical to idea014/design003.
- **idea006/design001** (flip) — any geometric augmentation broke pelvis; transforms are unchanged here. Temporal neighbours simply use the same crop as the centre frame (no re-cropping) to preserve pixel correspondence.

### Why this is novel and promising

- **14 ideas have never touched the temporal axis**, despite working with sequential video data (FRAME_STRIDE=5 is already in `infra.py`).
- The pelvis error floor (~94 mm) persists across nearly all designs — this suggests a per-frame depth ambiguity that temporal context is well-suited to resolve.
- Lightweight variants (A1 below) add zero backbone overhead by *sharing* features computed for neighbouring frames in the dataloader without re-running the backbone.

## Design Axes

### Category B -- Novel Exploration

All 4 designs are Category B. The key trade-off is compute/memory cost vs. fusion expressiveness. Designs proceed from cheapest to most ambitious; the Orchestrator can drop the most expensive if OOM occurs.

**Axis B1: Delta-input channel stacking (cheapest).**
The dataloader returns the centre frame plus the image at `t-5` (concatenated along channels to form an 8-channel input: `[RGB_t, D_t, RGB_{t-5}, D_{t-5}]`). The patch_embed of the Sapiens backbone is widened from 4 channels to 8; the 4 extra channel weights are initialized as the mean of the corresponding 4 original channel weights. No second backbone forward pass. Ground truth targets are unchanged (frame t). Cheapest possible temporal augmentation.

*Justification:* Similar to the 3->4 channel trick already used for the depth channel. Zero extra backbone forward passes. Tests whether the backbone can learn to use a raw previous-frame signal via channel expansion alone.

**Axis B2: Cross-frame attention at the head (shared backbone).**
The dataloader yields TWO frames: `t-5` and `t` (2 samples per input). The backbone runs **once per frame** inside the training step (using `torch.utils.checkpoint` to save memory), producing two memory banks of shape `(B, 960, 384)` each (projected through input_proj). The decoder memory becomes the *concatenation* of both banks `(B, 1920, 384)`. Everything else (queries, depth PE, LLRD, losses) is identical. Only the *centre frame's* joints are supervised.

*Justification:* True temporal fusion at the cross-attention layer. The decoder learns to attend to both time steps. Memory cost: 2x backbone forward + gradient memory. Uses gradient checkpointing on the backbone to stay under 11GB. Supervision is identical, so evaluation is directly comparable.

**Axis B3: Cross-frame attention with a frozen past-frame backbone.**
Same as B2 but the **past-frame (t-5) backbone forward pass runs in `torch.no_grad()` and is not checkpointed**. Only the centre-frame backbone is trained. This halves gradient memory vs. B2 and allows `FRAME_STRIDE=5` temporal context without the full 2x cost. Uses LLRD on centre-frame backbone only.

*Justification:* Explicitly trades expressiveness for memory. Tests whether "frozen temporal context" is sufficient, or whether gradients through the past-frame path are necessary.

**Axis B4: Three-frame TCN-style temporal head (symmetric window).**
The dataloader yields THREE frames: `t-5`, `t`, `t+5`. Each is passed through the backbone (the two neighbours in `torch.no_grad()`, the centre frame normally — same "frozen past" trick as B3, extended symmetrically). The decoder memory is the concatenation of all three banks `(B, 2880, 384)`. Only the centre frame's joints are supervised. This uses a symmetric temporal window, which is known to be stronger than asymmetric for mid-sequence frames in VideoPose3D.

*Justification:* The most expressive temporal fusion. Gradient memory overhead is roughly the same as B3 (one trainable backbone pass), but activation memory for the two frozen branches still requires careful handling (use `torch.no_grad() + del intermediate activations`).

## Expected Designs

The Designer should generate **4** novel designs:

1. **Delta-input channel stacking (2-frame, 8-channel input, single backbone pass).**
2. **Cross-frame memory attention (2-frame, both trainable, gradient checkpointing).**
3. **Cross-frame memory attention (2-frame, past frozen `no_grad`, centre trainable).**
4. **Three-frame symmetric temporal fusion (`t-5, t, t+5`, past/future frozen, centre trainable).**

## Design Constraints

- All designs start from `runs/idea014/design003/code/` and copy its optimizer, LR schedule, LLRD config (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, losses (Smooth L1 beta=0.05 on body joints).
- The continuous sqrt-spaced depth PE, wide head (hidden_dim=384, 4 layers, 8 heads), and core data transforms (CropPerson, SubtractRoot, ToTensor) MUST NOT be modified.
- **Dataloader changes.** The Designer MUST extend `BedlamFrameDataset.__getitem__` (or wrap it) to return additional frames. For B1/B2/B3: fetch frame at `max(0, idx - 5)`. For B4: fetch frames at `max(0, idx - 5)` and `min(len-1, idx + 5)`. Clamp at sequence boundaries. Use the *same crop bbox* (from the centre frame) for all temporal frames so pixel correspondence is preserved. Do NOT recompute CropPerson per-frame.
- **Target is always the centre-frame ground truth.** The loss is computed only on joints at time t. Neighbour frames are inputs only.
- **Backbone for B1:** widen `patch_embed.projection` from `Conv2d(4, 768, k, s)` to `Conv2d(8, 768, k, s)`. Initialize the 4 new channel weights as the mean of the corresponding 4 original channels (same trick as baseline RGB->RGBD). All other backbone weights unchanged.
- **Backbone for B2:** use `torch.utils.checkpoint.checkpoint(backbone_forward, frame)` for BOTH frame forwards to halve gradient memory. No `no_grad`. Concatenate memories along the spatial token axis before the decoder.
- **Backbone for B3/B4:** the non-centre forward passes run inside `with torch.no_grad():`. Their outputs must be detached before concatenation. Only the centre-frame pass participates in LLRD / backprop.
- **Memory check.**
  - B1: single backbone pass, +4 channels in patch_embed (trivial). Fits easily.
  - B2: two trainable backbone passes with checkpointing. Estimated peak memory ~9-10 GB at batch=4, accum=8. Designer MUST verify with a 1-step dry run before full training.
  - B3: one trainable + one `no_grad` backbone. Peak ~7-8 GB. Fits comfortably.
  - B4: one trainable + two `no_grad` backbones. Peak ~8-9 GB. Should fit; Designer to verify.
- **Decoder memory shape** must match the concatenated token count: `(B, num_frames * 960, 384)`. Depth PE is applied per-frame (reuse the same `row_emb + col_emb + depth_emb` module per frame). The decoder `TransformerDecoder` handles variable-length memory natively.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8`, `epochs=20`, `warmup_epochs=3` fixed.
- Do not modify `infra.py` constants.
- If any design OOMs in the first 10 steps, the Orchestrator should drop it and record it in the design_overview.csv as `Infeasible`.
