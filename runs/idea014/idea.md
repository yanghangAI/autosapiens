# idea014 -- Best-of-Breed Combination: LLRD + Depth PE + Wide Head

**Expected Designs:** 3

## Starting Point

The baseline starting point for this idea is:

`runs/idea008/design003/code/`

That design produced the best completed validation weighted MPJPE at **112.0 mm** (pelvis = 93.7 mm, body = 121.0 mm) using continuous depth PE with sqrt-spaced anchors, built on idea005/design001's row+col+depth decomposition. This idea layers the two strongest orthogonal improvements on top: LLRD from idea004/design002 and the wide head from idea009/design002.

## Concept

The pipeline has now identified three independently strong improvements that target non-overlapping components of the model:

| Improvement | Component Modified | Best Result |
|---|---|---|
| LLRD (gamma=0.90, unfreeze=5) | Optimizer schedule | 112.3 mm body (idea004/d002) |
| Continuous depth PE (sqrt) | Backbone positional encoding | 112.0 mm weighted (idea008/d003) |
| Wide head (hidden=384) | Decoder head | 112.3 mm body (idea009/d002) |

idea011 already tests LLRD + depth PE (2 of 3). This idea is the first to test the **full triple combination** (LLRD + depth PE + wide head) and pairwise combinations not yet explored (depth PE + wide head). Since each improvement modifies a different part of the model, they should combine without architectural conflict.

The wide head (idea009/design002) matched the best body MPJPE at 112.3 mm and showed that expanding the decoder hidden dimension from 256 to 384 is beneficial when combined with LLRD. Adding continuous depth PE should further improve pelvis localization.

## Broader Reflection

### Strong prior results to build on

- **idea004/design002** (val_mpjpe_body = **112.3 mm**) -- LLRD is the most consistent body MPJPE improvement.
- **idea008/design003** (val_mpjpe_weighted = **112.0 mm**, pelvis = **93.7 mm**) -- continuous depth PE is the strongest pelvis/weighted improvement.
- **idea009/design002** (val_mpjpe_body = **112.3 mm**, val_mpjpe_weighted = **130.4 mm**) -- wide head (hidden=384) matched the best body MPJPE and had the lowest val_mpjpe_weighted among idea009 designs, suggesting the wider head captures richer joint representations.
- **idea011** (LLRD + depth PE, still training) -- early results at epoch 10 show design004 at 117.7 mm body, on track to potentially beat either component alone. This validates that LLRD + depth PE can coexist.

### Patterns to avoid

- **idea007** (bucketed depth PE + LLRD) performed poorly (129-135 mm) -- only the continuous interpolated depth PE variant should be used, not the bucketed version.
- **idea009/design001** (6-layer decoder) did not improve over the 4-layer baseline (113.0 vs 112.3 mm). Deeper head is not helpful; wider head is.
- **idea003** (dynamic loss weighting) and **idea002** (attention masking) showed that aggressive structural/training changes underperform. This idea avoids such changes and only combines proven winners.
- **idea006** (augmentation) showed horizontal flip hurts pelvis. No augmentation changes here.

## Design Axes

### Category A -- Exploit & Extend

All three designs in this idea are Category A (combine independently strong improvements).

**Axis A1: Depth PE + Wide Head (no LLRD).**
Apply the wide head architecture from idea009/design002 (hidden_dim=384, 4 layers, 8 heads, input_proj Linear(1024->384), query Embedding(70,384)) to the idea008/design003 codebase (continuous depth PE, sqrt spacing). Keep the original flat optimizer from idea008/design003 (no LLRD). This isolates the interaction between depth PE and head width.

*Derives from:* idea008/design003 + idea009/design002.

**Axis A2: LLRD + Depth PE + Wide Head (triple combination).**
Apply both the LLRD schedule from idea004/design002 (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4) AND the wide head from idea009/design002 (hidden_dim=384) to the idea008/design003 codebase (continuous depth PE, sqrt spacing). The depth PE parameters (row_emb, col_emb, depth_emb) get head-level LR (1e-4). The wide head parameters also get head-level LR. This is the most ambitious combination.

*Derives from:* idea004/design002 + idea008/design003 + idea009/design002.

**Axis A3: LLRD + Depth PE + Wide Head + increased weight decay.**
Same as Axis A2 but with `weight_decay=0.3` (up from the baseline). idea012/design002 showed early promise with weight_decay=0.3, and the triple combination adds more parameters to the head, increasing overfitting risk. Higher weight decay provides a regularization counterbalance.

*Derives from:* idea004/design002 + idea008/design003 + idea009/design002 + idea012/design002.

## Expected Designs

The Designer should generate **3** novel designs:

1. **Depth PE + Wide Head** -- Start from `runs/idea008/design003/code/`. Replace the standard head (hidden=256) with the wide head (hidden_dim=384, num_heads=8, num_layers=4, input_proj Linear(1024->384), query Embedding(70,384), output Linear(384->3)). Keep the original optimizer from idea008/design003 (no LLRD, no progressive unfreezing). Keep continuous depth PE with sqrt spacing unchanged.
2. **LLRD + Depth PE + Wide Head** -- Same as design 1 but add LLRD: gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4. Depth PE params get head-level LR. Progressive unfreezing: freeze backbone for epochs 0-4, unfreeze at epoch 5 with per-layer LR = 1e-4 * 0.90^i. Cosine schedule with warmup_epochs=3.
3. **LLRD + Depth PE + Wide Head + weight_decay=0.3** -- Same as design 2 but increase weight_decay from 0.03 to 0.3 for all optimizer param groups.

## Design Constraints

- The continuous depth PE architecture (row/col/depth decomposition, sqrt anchor spacing, 16 depth anchors) from idea008/design003 must not be modified.
- The wide head follows idea009/design002 specs: hidden_dim=384, num_heads=8 (384/8=48 per head), num_layers=4, FFN_dim=4*384=1536, dropout=0.1. Input projection Linear(1024->384). Query embedding Embedding(70,384). Output Linear(384->3).
- For designs 2-3 (LLRD): follow the exact LLRD pattern from idea004/design002. Per-layer LR = base_lr * gamma^i where i=0 is shallowest block. Phase 1 (epochs 0 to 4): freeze backbone, train head + depth PE. Phase 2 (epoch 5+): unfreeze backbone with per-layer LR groups.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- `grad_clip=1.0`.
- Designs 1: use `weight_decay=0.03`, `lambda_depth=0.1`, `lambda_uv=0.2` (matching idea008/design003).
- Design 2: use `weight_decay=0.03`.
- Design 3: use `weight_decay=0.3`.
- Do not modify `infra.py` or the loss formulation (Smooth L1, beta=0.05).
- Memory check: wide head at 384 adds ~3M params over the 256 head. Combined with depth PE overhead from idea008/design003, total is well within 11GB at batch=4.
