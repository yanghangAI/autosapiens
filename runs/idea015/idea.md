# idea015 -- Iterative Refinement Decoder on SOTA Triple Combo

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea014/design003/code/`

That design is the current best completed run: **val_mpjpe_body = 106.85 mm**, **val_pelvis = 96.73 mm**, **val_mpjpe_weighted = 103.51 mm**. It combines LLRD (gamma=0.90, unfreeze=5), continuous sqrt-spaced depth PE, a wide head (hidden_dim=384, 4 layers, 8 heads), and weight_decay=0.3. This idea explores a single unexplored architectural axis on top of that new SOTA: **iterative refinement of joint predictions via cascaded decoder passes**.

## Concept

No prior idea has varied the *prediction protocol* of the head. All prior designs run a single forward pass of the TransformerDecoder over 70 joint queries and emit a single 3D joint prediction. In pose estimation (see DETR-style pose heads, PRTR, PoseFormer), **iterative refinement** of query features — re-feeding a coarse prediction back into the query embedding — is a standard and well-motivated way to improve localization without adding width or loss terms.

The hypothesis: given that idea014/design003 already has a strong positional prior (depth PE) and a rich wide head, the residual errors are likely due to ambiguous long-range image-token attention matching. A second decoder pass, where each joint query is conditioned on its own coarse 3D prediction, should be able to *re-attend* to the memory tokens near the predicted location and reduce error. This is orthogonal to LLRD, depth PE, and head width.

## Broader Reflection

### Strong prior results to build on

- **idea014/design003** (val_mpjpe_body = **106.85 mm**, weighted = **103.51 mm**) — new SOTA, triple combination. Starting point.
- **idea014/design002** (val_mpjpe_body = **108.04 mm**) — close second, same triple combination without increased weight decay.
- **idea011/design001** (val_mpjpe_body = **109.93 mm**) — LLRD + continuous depth PE (no wide head) confirms the LLRD+depth PE base is robust.
- **idea008/design003** (val_mpjpe_weighted = **112.0 mm**) — confirms continuous sqrt-spaced depth PE.
- **idea009/design002** (wide head) — confirms hidden_dim=384 is beneficial.

### Patterns to avoid

- **idea007** (bucketed depth PE + LLRD) — stuck near 130+ mm. Do not touch the PE type.
- **idea009/design001** (6-layer decoder) — deeper decoder did not help; iterative refinement is different (same 4 layers, *re-used* with injected prediction), not deeper stack.
- **idea002** (kinematic masking) — hard attention masks hurt. Use soft conditioning only.
- **idea003** (dynamic loss weighting) — aggressive curriculum underperformed. Use only a simple auxiliary supervision on intermediate predictions.
- **idea006/design001** (horizontal flip) — pelvis blew up. Do not touch augmentation here.

### Promising direction

idea014/design003 is the first run to combine all three independent winners. The pelvis error (96.73 mm) is now very close to body error (106.85), meaning the remaining gap is on *body joint localization*, not root localization. Refinement heads historically target exactly this regime: a coarse-to-fine cascade where the second stage only needs to fix small-to-medium errors. This is the right next step on top of the SOTA.

## Design Axes

### Category A -- Exploit & Extend

All 4 designs derive from `runs/idea014/design003/code/` and keep everything (LLRD gamma=0.90, unfreeze=5, sqrt continuous depth PE, wide head 384, weight_decay=0.3, losses, lambda_depth=0.1, lambda_uv=0.2, 20 epochs, warmup=3) **identical** to that run. Only the decoder prediction protocol changes.

**Axis A1: Two-pass iterative refinement with joint-prediction feedback.**
After the first 4-layer decoder pass produces coarse joints `J1 (B,70,3)`, project `J1` through a small MLP `Linear(3->384) -> GELU -> Linear(384->384)` to produce a *refinement delta* `R (B,70,384)`. Add `R` to the original joint queries (`query_embed + R`), re-run the *same* 4-layer decoder over the same memory, and emit a second `J2 (B,70,3)` via a second Linear(384->3) output head. Final prediction = `J2`. Apply the standard Smooth L1 loss on both `J1` and `J2` with weights `0.5` and `1.0` respectively (deep supervision on the coarse stage stabilizes training).

*Derives from:* idea014/design003 + DETR iterative refinement pattern.

**Axis A2: Two-pass refinement with memory-gated attention.**
Same as A1, but instead of adding the refinement delta to queries, use the coarse prediction `J1` to compute a per-joint soft "focus" prior: project the predicted `(u,v)` (from pelvis-relative -> absolute approximation) onto the memory grid (40×24) as a Gaussian centred on the projected location with sigma=2 patches. Use this Gaussian as an *additive bias* (not a hard mask, avoiding NaN rows) to the cross-attention logits of the second decoder pass. The decoder still self-attends and cross-attends normally but is gently biased to attend near its own coarse prediction.

*Derives from:* idea014/design003 + soft-gated cascaded refinement (extends idea005 continuous-PE philosophy).

**Axis A3: Three-pass refinement (progressive).**
Same architecture as A1 but cascade the decoder three times (J1 -> J2 -> J3) with deep supervision losses at weights 0.25, 0.5, 1.0. Final prediction = J3. Tests whether a third pass yields additional gains or overfits given only 20 epochs. The *same* weights are reused across passes (weight sharing) to keep parameters bounded.

*Derives from:* idea014/design003 + multi-stage DETR refinement.

**Axis A4: Two-pass refinement with separate refinement decoder (non-shared weights).**
Same as A1 but the refinement stage uses an *independent* 2-layer TransformerDecoder (hidden=384, 8 heads) rather than re-using the 4-layer decoder. Deep supervision weights 0.5 / 1.0. This separates "coarse" capacity from "refine" capacity so each can specialize. Trade-off: more parameters (+~3M in the extra 2-layer decoder), but memory still fits (batch=4 leaves headroom on 11GB).

*Derives from:* idea014/design003 + two-stage head specialization.

## Expected Designs

The Designer should generate **4** novel designs (count only the new variations; the idea014/design003 baseline is already evaluated and is NOT re-designed):

1. **Two-pass shared-decoder refinement (query injection).**
2. **Two-pass shared-decoder refinement (cross-attention Gaussian bias from J1).**
3. **Three-pass shared-decoder refinement (deep supervision 0.25/0.5/1.0).**
4. **Two-pass two-decoder refinement (independent 2-layer refine decoder).**

## Design Constraints

- All designs start from `runs/idea014/design003/code/` and copy its optimizer, LR schedule, LLRD config (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.
- The continuous sqrt-spaced depth PE (16 anchors) and wide head shape (hidden_dim=384, num_heads=8, num_layers=4, FFN=1536) MUST NOT be modified.
- Deep supervision losses apply to intermediate predictions via the *same* Smooth L1 (beta=0.05, BODY_IDX only) as the final loss. Weights: A1/A2 => `0.5*L(J1) + 1.0*L(J2)`; A3 => `0.25*L(J1) + 0.5*L(J2) + 1.0*L(J3)`; A4 => `0.5*L(J1) + 1.0*L(J2)`.
- For A2: the memory bias is `additive` (shape `(B, 70, 960)` broadcast over heads). Use `memory_mask` argument of `TransformerDecoderLayer.forward` or an equivalent manual path. Clamp to finite values; never produce fully -inf rows.
- For A2: to convert predicted 3D joints into `(u,v)` on the 40x24 grid, use the inverse of the root-relative transform encoded in the data pipeline: approximate as the normalized pelvis offset only (we do NOT reconstruct camera intrinsics). Specifically, compute `uv_norm = pelvis_uv + 0.5 * (joints[:,:,:2] / joints[:,:,2:3].clamp(min=0.1))` and clamp to `[0,1]`, then scale to grid coordinates. Designer must confirm the `pelvis_uv` tensor is available from the data pipeline; if not, fall back to using the center of the grid + joint offsets normalized by the image size (document the chosen fallback in `design.md`).
- For A2: sigma of the Gaussian bias is fixed to 2.0 patches, bias magnitude scaled by a learnable scalar initialized at 0.0 so the first pass is identical to A1/baseline at step 0.
- For A4: the additional 2-layer decoder parameters go into the `head_params` optimizer group (LR=1e-4, weight_decay=0.3). No LLRD on head params.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- Do not modify `infra.py` or the transforms.
- Memory check: shared-decoder refinement (A1/A2/A3) adds only the 3->384 MLP (~150K params) and one extra Linear(384->3) per stage. A4 adds ~3M params. All fit in 11GB at batch=4 with accum=8.
