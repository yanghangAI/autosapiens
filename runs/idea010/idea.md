# idea010 — Multi-Scale Backbone Feature Aggregation

**Expected Designs:** 5

## Starting Point

The baseline starting point for this idea is:

`runs/idea004/design002/train.py`

That design produced the best completed validation body MPJPE at **112.3 mm** using LLRD (gamma=0.90, unfreeze_epoch=5). The LLRD schedule is kept fixed across all designs so that gains are attributable purely to the multi-scale feature aggregation changes.

## Concept

Every prior idea has consumed only the **final ViT layer output** `(B, 1024, 40, 24)` as input to the decoder head. In ViT architectures, intermediate layers capture qualitatively different information: early layers encode local spatial/texture patterns (useful for precise joint localization), middle layers encode part-level structure, and late layers encode global semantic context. By aggregating features from multiple ViT layers before feeding them to the cross-attention decoder, the head gains access to a richer, multi-resolution representation without any additional backbone parameters.

This is a well-established technique in dense prediction tasks (e.g., ViTDet, DPT) but has never been explored in this pipeline. The approach modifies only the feature extraction interface between backbone and head, keeping the backbone weights and head architecture unchanged.

## Broader Reflection

### Strong prior results to build on

- **idea004/design002** (val_mpjpe_body = **112.3 mm**) is the best completed body MPJPE result. Its LLRD schedule is well-characterized and will be the fixed optimization baseline for all designs here.
- **idea008/design003** (val_mpjpe_weighted = **112.0 mm**) showed that enriching the positional/geometric information available to the backbone improves pelvis localization significantly. Multi-scale features could provide complementary spatial detail.
- **idea009** (head refinement, still training) explores decoder depth and width. Multi-scale features are orthogonal to head architecture changes and could stack with head improvements in future combinations.

### Patterns to avoid

- **idea007** attempted to combine depth PE + LLRD but performed worse (129-135 mm) than either alone. This warns against naively stacking two complex changes. The designs here are careful to change only the feature aggregation path while keeping the LLRD schedule exactly as in idea004/design002.
- **idea002** (kinematic attention masking) showed that modifying attention patterns among queries hurts within 20 epochs. Multi-scale aggregation does not touch the decoder's self-attention or cross-attention masks.
- **idea003** (curriculum loss) showed dynamic loss changes need more than 20 epochs. All designs here use the standard loss.
- **OOM risk**: Extracting intermediate features adds negligible memory since the backbone already computes them during the forward pass. The aggregation modules (projection + fusion) add at most ~2M parameters. Safe within the 11GB budget at batch=4.

## Design Axes

### Category A -- Exploit & Extend

**Axis A1: Last-4-layer concatenation + linear projection.**
Extract the outputs of the last 4 ViT transformer blocks (layers 9, 10, 11, 12 for the 12-block Sapiens 0.3B). Concatenate along the channel dimension to get `(B, 4096, 40, 24)`, then project down via `Linear(4096, 1024)` to match the existing head input dimension. This is the simplest possible multi-scale aggregation and serves as the baseline multi-scale design.

*Derives from:* idea004/design002 (best LLRD schedule) combined with DPT/ViTDet-style multi-layer feature usage.

**Axis A2: Last-4-layer weighted sum (learned layer weights).**
Instead of concatenation, compute a learned weighted average of the last 4 layer outputs: `sum(w_i * layer_i) for i in {9,10,11,12}` where `w_i` are softmax-normalized learnable scalars (initialized uniformly). This adds only 4 scalar parameters and preserves the `(B, 1024, 40, 24)` shape without any projection layer. Inspired by ELMo-style layer mixing.

*Derives from:* idea004/design002 (best schedule); simpler alternative to concatenation that avoids extra projection parameters.

### Category B -- Novel Exploration

**Axis B1: Feature Pyramid with 3 scales.**
Extract features from ViT layers 4, 8, and 12 (early, middle, final). Project each to 256 channels via separate `Linear(1024, 256)` layers. Upsample/downsample spatial dimensions to a common 40x24 grid (layer 4 and 8 are already 40x24 in this ViT since all layers share the same spatial resolution). Concatenate the three 256-channel maps to get `(B, 768, 40, 24)`, then project to `(B, 1024, 40, 24)` via `Linear(768, 1024)`. This gives the head a true multi-scale pyramid.

**Axis B2: Cross-scale attention gate.**
Extract features from layers 6 and 12 (mid and final). Compute a spatial attention gate from the mid-layer features: `gate = sigmoid(Linear(1024, 1)(layer_6))` with shape `(B, 1, 40, 24)`. Apply it element-wise to the final-layer features: `output = layer_12 * gate + layer_12`. The gate learns to up-weight spatial locations where mid-level features indicate important local structure (e.g., limb boundaries). Adds only a single `Linear(1024, 1)` layer (~1K parameters).

**Axis B3: Alternating layer interleave.**
Extract features from even-indexed layers (2, 4, 6, 8, 10, 12) and compute a simple channel-wise average: `output = mean(layer_2, layer_4, ..., layer_12)`. This uniformly samples the full depth of the backbone. No additional parameters are needed. The head receives a smoothed representation that combines low-level spatial detail with high-level semantics.

## Expected Designs

The Designer should generate **5** novel designs:

1. **Last-4-layer concatenation** -- Extract layers {9,10,11,12}, concatenate to `(B, 4096, 40, 24)`, project via `Linear(4096, 1024)`. Xavier init on the projection. Keep LLRD schedule from idea004/design002.
2. **Learned layer weights** -- Extract layers {9,10,11,12}, compute softmax-weighted sum with 4 learnable scalar weights (init uniform 0.25 each). Output shape `(B, 1024, 40, 24)` unchanged. No extra projection needed.
3. **Feature pyramid (3 scales)** -- Extract layers {4, 8, 12}, project each via `Linear(1024, 256)`, concatenate to `(B, 768, 40, 24)`, final projection `Linear(768, 1024)`. Xavier init on all new projections.
4. **Cross-scale attention gate** -- Extract layers {6, 12}. Gate = `sigmoid(Linear(1024, 1)(layer_6))` applied as spatial multiplicative gate on layer_12 via `layer_12 * (1 + gate)`. Zero-init the gate projection bias so initial behavior matches baseline.
5. **Alternating layer average** -- Extract layers {2, 4, 6, 8, 10, 12}, compute channel-wise mean. Zero additional parameters. Simplest possible multi-scale baseline.

## Design Constraints

- Keep the LLRD schedule from idea004/design002 (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4) fixed across all designs.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- All designs must use the standard 4-channel RGBD input with baseline depth normalization.
- The backbone architecture (Sapiens ViT-B, 12 transformer blocks) is not modified -- only the feature extraction interface changes.
- To extract intermediate layer features, the Designer must modify `SapiensBackboneRGBD.forward()` to hook into `self.vit.layers` (the `nn.ModuleList` of transformer blocks) and return selected intermediate outputs. The mmpretrain `VisionTransformer` stores its blocks in `self.layers` (a `nn.ModuleList` of length 12 for `sapiens_0.3b`).
- New parameters (projection layers, learnable weights) should be placed in a separate optimizer param group with `lr=1e-4` (same as head LR) and included in the LLRD wiring.
- Do not modify `infra.py` or the loss computation.
- For design 4 (cross-scale gate), use residual form `output = layer_12 * (1 + gate)` so that at initialization (gate~0 due to zero-init bias), behavior is identical to the baseline.

## Implementation Notes for the Designer

- The mmpretrain `VisionTransformer` forward pass iterates through `self.layers`. To extract intermediate features, write a custom forward method in the backbone wrapper that manually iterates through the layers and saves outputs at the desired indices.
- All ViT layers output the same spatial shape `(B, 960, 1024)` (flattened 40x24 grid), so no spatial resampling is needed between layers.
- The existing `Pose3DHead.forward()` expects input of shape `(B, C, H, W)`. If the aggregation changes the channel count, update the head's `input_proj` accordingly (designs 1 and 3).
- For designs 1 and 3 where the channel dimension changes, the pretrained weight loading in `load_sapiens_pretrained` does not need modification since it only loads backbone weights, not the new aggregation parameters.
