# idea011 — LLRD Schedule Combined with Continuous Depth PE

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea008/design003/code/`

That design produced the best completed validation weighted MPJPE at **112.0 mm** using continuous depth positional encoding with near-emphasized (square-root) anchor spacing, built on top of `idea005/design001`'s row+column+depth decomposition. This idea layers the LLRD optimization schedule from `idea004/design002` on top of that architecture.

## Concept

The two strongest completed results in the pipeline target orthogonal aspects of the model:

- **idea004/design002** (val_mpjpe_body = 112.3 mm, val_mpjpe_weighted = 130.7 mm) achieved the best body joint accuracy by refining backbone optimization via layer-wise learning rate decay (LLRD) with gamma=0.90 and progressive unfreezing at epoch 5.
- **idea008/design003** (val_mpjpe_body = 121.0 mm, val_mpjpe_weighted = 112.0 mm) achieved the best overall weighted score by dramatically improving pelvis localization (93.7 mm) through continuous depth-aware positional encoding with near-emphasized spacing.

These two improvements have **never been combined**. LLRD modifies the optimizer schedule; continuous depth PE modifies the positional encoding fed to the backbone. There is no architectural conflict. The hypothesis is that combining them will yield both the strong body MPJPE from LLRD and the strong pelvis accuracy from depth PE, potentially achieving a new best on both metrics simultaneously.

## Broader Reflection

### Strong prior results to build on

- **idea004/design002** (val_mpjpe_body = **112.3 mm**) is the best body MPJPE. Its LLRD schedule (gamma=0.90, unfreeze_epoch=5) is well-characterized.
- **idea008/design003** (val_mpjpe_weighted = **112.0 mm**, pelvis = **93.7 mm**) is the best weighted score and best pelvis. Its continuous depth PE with sqrt spacing is the strongest positional encoding variant.
- **idea008/design004** (val_mpjpe_weighted = **112.1 mm**) confirms the robustness of the continuous depth PE approach (hybrid two-resolution variant also performs well).

### Patterns to avoid

- **idea007** (depth PE + LLRD combo attempt) performed poorly (129-135 mm) but used a different, weaker depth PE formulation (bucketed, not continuous interpolated). The key lesson is that the depth PE variant matters — only the proven continuous interpolated version should be combined with LLRD.
- **idea002** (kinematic masking) and **idea003** (curriculum loss) showed that aggressive structural changes to attention or training dynamics underperform. This idea avoids such changes.
- **idea006** (augmentation) showed that horizontal flip with joint swapping can harm pelvis localization badly (270 mm pelvis error). No augmentation changes are made here.

## Design Axes

### Category A — Exploit & Extend

**Axis A1: Direct combination of LLRD + continuous depth PE.**
Apply the exact LLRD schedule from idea004/design002 (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4) to the idea008/design003 architecture (continuous depth PE with sqrt spacing, 16 depth anchors). The depth PE parameters (`row_emb`, `col_emb`, `depth_emb`) get the same high LR as the head (1e-4). This is the most conservative combination.

*Derives from:* idea004/design002 + idea008/design003.

**Axis A2: LLRD with reduced gamma + continuous depth PE.**
Same as Axis A1 but with gamma=0.85 (more aggressive layer-wise decay). idea004/design003 showed gamma=0.85 performed slightly worse than 0.90 on its own (113.9 vs 112.3 body MPJPE), but the stronger positional encoding from depth PE may interact differently — a more aggressive LLRD could help the model rely more on the enriched positional information rather than lower backbone layers.

*Derives from:* idea004/design003 + idea008/design003.

**Axis A3: LLRD + continuous depth PE with later unfreezing.**
Same as Axis A1 but with unfreeze_epoch=10 instead of 5. idea004/design004 showed that later unfreezing (epoch 10) with gamma=0.95 achieved 112.8 mm — competitive but slightly worse. With the depth PE providing richer spatial information to the head, the head may benefit from longer exclusive training before backbone fine-tuning begins.

*Derives from:* idea004/design004 + idea008/design003.

**Axis A4: LLRD + continuous depth PE with gated depth residual.**
Combine the LLRD schedule (gamma=0.90, unfreeze_epoch=5) with the depth PE from idea008/design002, which adds a learned residual gate controlling the depth PE contribution. This tests whether giving the model explicit control over depth PE strength is beneficial when combined with a refined optimization schedule.

*Derives from:* idea004/design002 + idea008/design002.

## Expected Designs

The Designer should generate **4** novel designs:

1. **LLRD (gamma=0.90, unfreeze=5) + sqrt-spaced continuous depth PE** — Direct combination of the two best configurations.
2. **LLRD (gamma=0.85, unfreeze=5) + sqrt-spaced continuous depth PE** — More aggressive LLRD decay.
3. **LLRD (gamma=0.90, unfreeze=10) + sqrt-spaced continuous depth PE** — Later backbone unfreezing.
4. **LLRD (gamma=0.90, unfreeze=5) + gated continuous depth PE** — Adding a learned scalar gate on the depth PE residual (from idea008/design002).

## Design Constraints

- Designs 1-3 start from `runs/idea008/design003/code/` and add LLRD scheduling to the optimizer.
- Design 4 starts from `runs/idea008/design002/code/` and adds LLRD scheduling.
- The continuous depth PE architecture (row/col/depth decomposition, sqrt spacing for designs 1-3, gated residual for design 4) must not be modified.
- The LLRD implementation must follow the same pattern as idea004/design002: per-layer LR = `base_lr * gamma^i` where i=0 is the shallowest backbone block, with cosine schedule + warmup_epochs=3.
- During the frozen phase (epochs 0 to unfreeze_epoch-1), only the head and depth PE parameters are trained. Backbone params have `requires_grad=False`.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- `weight_decay=0.03` for backbone, head, and depth PE groups.
- `grad_clip=1.0`.
- `lambda_depth=0.1`, `lambda_uv=0.2`.
- Do not modify `infra.py`, transforms, or the loss computation.
