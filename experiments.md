# Experiments Summary

This repository's `runs/` directory is organized as a research log:

- `runs/ideaNNN/idea.md` describes the hypothesis or experiment family.
- `runs/ideaNNN/designXXX/design.md` describes one concrete implementation of that idea.
- Each design folder usually contains `code/`, `metrics.csv`, snapshots, and review notes.

Most experiments start from an RGB-D Sapiens pose model with:

- 4-channel RGB+depth input
- ViT backbone
- Transformer decoder head
- 20-epoch training proxy

## Idea List

Metric shorthand in the design entries:

- `W` = validation weighted MPJPE, in millimeters
- `@eN` = metric recorded at epoch `N`
- `no consolidated result` = no row was present in `results.csv`

### `idea001`
Theme: RGB-D fusion strategy: test where depth should be fused into the model.

- `design001` early 4-channel fusion, <span><strong>W 142.5 @e20 (baseline)</strong></span>
- `design002` mid-layer fusion, <span style="color: green;">W 139.6 @e20</span>
- `design003` late cross-attention fusion, <span style="color: red;">W 142.8 @e20</span>

### `idea002`
Theme: Kinematic attention masking: enforce anatomical structure inside decoder attention.

- `design001` dense attention control, <span><strong>W 142.5 @e20 (baseline)</strong></span>
- `design002` soft kinematic mask, <span style="color: red;">W 144.5 @e20</span>
- `design003` hard kinematic mask, <span style="color: green;">W 139.5 @e20</span>

### `idea003`
Theme: Curriculum-based loss weighting: make hard objectives, especially depth, easier to learn early.

- `design001` homoscedastic uncertainty weighting, <span style="color: red;">W 163.0 @e20</span>
- `design002` linear warmup for depth loss, <span style="color: red;">W 143.2 @e20</span>

### `idea004`
Theme: Layer-wise learning rate decay plus progressive unfreezing: protect pretrained ViT features while adapting deeper layers faster.

- `design001` LLRD `gamma=0.95`, `unfreeze=5`, <span style="color: green;">W 131.9 @e20</span>
- `design002` LLRD `gamma=0.90`, `unfreeze=5`, <span style="color: green;">W 130.7 @e20</span>
- `design003` LLRD `gamma=0.85`, `unfreeze=5`, <span style="color: green;">W 131.7 @e20</span>
- `design004` LLRD `gamma=0.95`, `unfreeze=10`, <span style="color: green;">W 132.3 @e20</span>
- `design005` LLRD `gamma=0.90`, `unfreeze=10`, <span style="color: green;">W 131.6 @e20</span>
- `design006` LLRD `gamma=0.85`, `unfreeze=10`, <span style="color: green;">W 131.9 @e20</span>

### `idea005`
Theme: Depth-aware positional embeddings: inject geometry through position encoding rather than only through the input tensor.

- `design001` discretized depth buckets, <span style="color: green;">W 121.4 @e20</span>
- `design002` relative depth bias, <span style="color: green;">W 142.1 @e20</span>
- `design003` depth-conditioned positional embeddings, <span style="color: red;">W 146.1 @e20</span>

### `idea006`
Theme: Training augmentation for generalization: reduce the train/val gap through input perturbations.

- `design001` horizontal flip, <span style="color: red;">W 175.5 @e20</span>
- `design002` scale/crop jitter, <span style="color: red;">W 142.8 @e20</span>
- `design003` color jitter, <span style="color: green;">W 140.9 @e20</span>
- `design004` depth noise/dropout, <span style="color: red;">W 143.1 @e20</span>
- `design005` flip + scale jitter, <span style="color: red;">W 176.4 @e20</span>
- `design006` flip + color jitter + depth noise, <span style="color: red;">W 170.8 @e20</span>

### `idea007`
Theme: Combine depth-bucket positional embeddings with LLRD.

- `design001` gentle LLRD, <span style="color: green;">W 132.6 @e20</span>
- `design002` stronger LLRD, <span style="color: green;">W 135.0 @e20</span>
- `design003` stronger LLRD with earlier unfreeze, <span style="color: green;">W 134.9 @e20</span>

### `idea008`
Theme: Continuous depth positional encoding: replace bucketed depth PE with smoother interpolated variants.

- `design001` interpolated depth PE, <span style="color: green;">W 116.4 @e20</span>
- `design002` interpolated PE with residual gate, <span style="color: green;">W 118.8 @e20</span>
- `design003` near-emphasized or sqrt-spaced anchors, <span style="color: green;">W 112.0 @e20</span>
- `design004` hybrid two-resolution depth PE, <span style="color: green;">W 112.1 @e20</span>

### `idea009`
Theme: Decoder head refinement: vary the head instead of the backbone or input encoding.

- `design001` deeper 6-layer decoder, <span style="color: green;">W 131.4 @e20</span>
- `design002` wider head with hidden dim 384, <span style="color: green;">W 130.4 @e20</span>
- `design003` sine-cosine joint query initialization, <span style="color: green;">W 132.2 @e20</span>
- `design004` per-layer feature gate, <span style="color: green;">W 131.6 @e20</span>
- `design005` output LayerNorm, <span style="color: green;">W 133.0 @e20</span>

### `idea010`
Theme: Multi-scale backbone feature aggregation: expose intermediate ViT layers to the head instead of only the final feature map.

- `design001` last-4-layer concat + projection, <span style="color: green;">W 128.9 @e20</span>
- `design002` learned softmax-weighted layer mix, <span style="color: green;">W 132.3 @e20</span>
- `design003` 3-scale feature pyramid, <span style="color: green;">W 130.0 @e20</span>
- `design004` cross-scale attention gate, <span style="color: green;">W 130.0 @e20</span>
- `design005` alternating-layer average, <span style="color: green;">W 138.5 @e20</span>

### `idea011`
Theme: Combine the strongest depth PE variant with the strongest LLRD schedule.

- `design001` `gamma=0.90` + sqrt depth PE, <span style="color: green;">W 104.7 @e20</span>
- `design002` `gamma=0.85` + sqrt depth PE, <span style="color: green;">W 107.0 @e20</span>
- `design003` later unfreeze + sqrt depth PE, <span style="color: green;">W 109.5 @e20</span>
- `design004` gated continuous depth PE + LLRD, <span style="color: green;">W 110.3 @e20</span>

### `idea012`
Theme: Regularization for generalization: reduce overfitting without changing the input distribution.

- `design001` higher head dropout, <span style="color: green;">W 132.8 @e20</span>
- `design002` higher weight decay, <span style="color: green;">W 131.3 @e20</span>
- `design003` higher drop path, <span style="color: green;">W 136.7 @e20</span>
- `design004` R-Drop consistency, <span style="color: green;">W 132.5 @e20</span>
- `design005` combined regularization, <span style="color: green;">W 137.7 @e20</span>

### `idea013`
Theme: Joint prediction loss reformulation: keep the model fixed and vary the pose loss itself.

- `design001` smaller Smooth L1 beta, <span style="color: green;">W 130.6 @e20</span>
- `design002` larger Smooth L1 beta, <span style="color: green;">W 134.0 @e20</span>
- `design003` bone-length auxiliary loss, <span style="color: green;">W 130.9 @e20</span>
- `design004` hard-joint-weighted loss, <span style="color: green;">W 131.0 @e20</span>

### `idea014`
Theme: Best-of-breed combination: stack the strongest earlier ideas into a stronger composite system.

- `design001` depth PE + wide head, <span style="color: green;">W 121.5 @e20</span>
- `design002` LLRD + depth PE + wide head, <span style="color: green;">W 104.4 @e20</span>
- `design003` same triple + higher weight decay, <span style="color: green;">W 103.5 @e20</span>

### `idea015`
Theme: Iterative refinement decoder: move from one-shot prediction to multi-pass prediction.

- `design001` two-pass shared decoder with query injection, <span style="color: green;">W 101.9 @e20</span>
- `design002` two-pass shared decoder with J1-guided cross-attention bias, <span style="color: red;">W 143.7 @e6</span>
- `design003` three-pass shared decoder with deep supervision, <span style="color: red;">W 229.6 @e5</span>
- `design004` two-pass with an independent refine decoder, <span style="color: green;">W 98.9 @e20</span>

### `idea016`
Theme: 2.5D heatmap soft-argmax: replace direct coordinate regression with heatmap-style outputs.

- `design001` 2D heatmap + scalar depth on native grid, <span style="color: red;">W 180.1 @e20</span>
- `design002` same on upsampled grid, <span style="color: red;">W 178.7 @e20</span>
- `design003` full 3D volumetric heatmap, <span style="color: red;">W 335.7 @e6</span>
- `design004` 2D heatmap + scalar depth + auxiliary Gaussian supervision, <span style="color: red;">W 179.4 @e20</span>

### `idea017`
Theme: Temporal adjacent-frame fusion: use neighboring frames instead of single-frame inference only.

- `design001` delta-input channel stacking, no consolidated result
- `design002` 2-frame cross-frame memory attention, <span style="color: red;">W 154.6 @e5</span>
- `design003` 2-frame memory with frozen past branch, <span style="color: red;">W 167.1 @e5</span>
- `design004` 3-frame symmetric temporal fusion, <span style="color: red;">W 167.7 @e5</span>

### `idea018`
Theme: Weight averaging: improve the final model by averaging weights rather than changing architecture.

- `design001` EMA, <span style="color: red;">W 212.6 @e5</span>
- `design002` EMA with warmup, <span style="color: red;">W 148.5 @e5</span>
- `design003` SWA over the last 5 epochs, <span style="color: red;">W 169.7 @e5</span>
- `design004` EMA plus a final low-LR polish epoch, <span style="color: red;">W 212.6 @e5</span>

### `idea019`
Theme: Anatomical structure priors for iterative refinement: inject bones, symmetry, and joint grouping into the refine stage.

- `design001` bone-length auxiliary loss, <span style="color: green;">W 103.1 @e20</span>
- `design002` kinematic soft-attention bias, <span style="color: green;">W 102.9 @e20</span>
- `design003` left-right symmetry loss, <span style="color: green;">W 103.5 @e20</span>
- `design004` joint-group query initialization, <span style="color: green;">W 102.2 @e20</span>
- `design005` combined anatomical priors, <span style="color: green;">W 106.5 @e20</span>

### `idea020`
Theme: Refinement-specific loss and gradient strategy: tune how the coarse and refine passes interact during training.

- `design001` stop-gradient on coarse `J1`, <span style="color: green;">W 101.9 @e20</span>
- `design002` reduced coarse supervision weight, <span style="color: green;">W 101.2 @e20</span>
- `design003` L1 loss on refinement pass only, <span style="color: green;">W 99.4 @e20</span>
- `design004` higher LR for refine decoder, <span style="color: green;">W 100.0 @e20</span>
- `design005` residual refinement formulation, <span style="color: red;">W 168.3 @e5</span>

### `idea021`
Theme: Re-test anatomical priors on the stronger two-decoder SOTA rather than the weaker shared-decoder baseline.

- `design001` kinematic bias in refine decoder only, <span style="color: green;">W 98.6 @e20</span>
- `design002` joint-group injection before refine decoder, <span style="color: green;">W 100.7 @e20</span>
- `design003` bone-length loss on `J2`, <span style="color: green;">W 101.4 @e20</span>
- `design004` kinematic bias + group injection, <span style="color: green;">W 99.5 @e20</span>

## Evolution of the Experiment Tree

- `idea001` to `idea006` explore first-order changes around the baseline: fusion, masking, loss weighting, optimization, positional encoding, and augmentation.
- `idea007` to `idea014` mostly refine or combine the strongest wins from earlier rounds: LLRD, depth positional encoding, decoder width, multi-scale features, regularization, and loss variants.
- `idea015` onward shifts to prediction protocol and training dynamics: iterative refinement, alternative output representations, temporal fusion, weight averaging, anatomical priors, and refine-specific optimization.
