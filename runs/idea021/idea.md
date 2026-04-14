# idea021 -- Anatomical Priors on the Two-Decoder SOTA

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea015/design004/code/`

That design is the current overall SOTA: **val_mpjpe_body = 102.51 mm**, **val_pelvis = 91.62 mm**, **val_mpjpe_weighted = 98.92 mm**. Architecture: LLRD (gamma=0.90, unfreeze=5), continuous sqrt-spaced depth PE, wide head (hidden_dim=384, 4 layers, 8 heads), weight_decay=0.3, and two-pass iterative refinement with an independent 2-layer refine decoder.

## Concept

idea019 tested anatomical priors (bone-length loss, kinematic attention bias, symmetry loss, joint-group embeddings) on idea015/design001 (shared-decoder, val_weighted=101.94). Results were marginal: best body MPJPE of 105.77 (design002, kinematic bias) and best weighted of 102.22 (design004, joint-group init). However, idea019 was built on the **wrong baseline** -- it used idea015/design001 (shared-decoder) instead of idea015/design004 (two-decoder), which is 3 mm better on weighted MPJPE.

The two-decoder architecture in design004 has a dedicated 2-layer refine decoder that is architecturally better suited to receive anatomical priors: the refinement pass processes already-reasonable coarse predictions, making structural constraints (bone lengths, symmetry) more meaningful than when applied to random initial predictions. This idea re-tests the most promising anatomical priors from idea019 on the correct (stronger) baseline, selecting only the configurations that showed the most promise.

## Broader Reflection

### Strong prior results to build on

- **idea015/design004** (val_weighted=**98.92**, body=**102.51**, pelvis=**91.62**) — true SOTA. Independent 2-layer refine decoder.
- **idea019/design002** (val_body=**105.77**, weighted=**102.85**) — kinematic soft-attention bias was the best body MPJPE in idea019. This was on idea015/design001 (shared decoder, 3 mm worse baseline).
- **idea019/design004** (val_weighted=**102.22**, pelvis=**93.94**) — joint-group query init was the best weighted MPJPE in idea019. Also on the weaker baseline.
- **idea019/design001** (val_body=**106.26**) — bone-length aux loss was neutral on the shared-decoder baseline.
- **idea019/design005** (val_body=**107.62**, pelvis=**104.37**) — combined priors (bone+sym+kinematic) was worse than individual priors. Over-regularization likely.

### Key selection rationale

From idea019's results:
- **Kinematic soft-attention bias** (design002) gave the best body improvement. It modifies the self-attention in the decoder, which is a natural fit for the refine decoder in design004.
- **Joint-group query init** (design004) gave the best weighted improvement and good pelvis (93.94). It adds group structure before the refinement pass, which pairs well with an independent refine decoder.
- **Bone-length loss** (design001) was neutral, but the refine decoder produces more refined predictions which may benefit more from a bone-length constraint.
- **Combined priors** (design005) hurt -- so we will test at most a careful pair combination, not the full triple.

### Patterns to avoid

- Do NOT combine all three priors (idea019/design005 showed this hurts).
- Do NOT use symmetry loss alone (idea019/design003 failed at epoch 6 with 137.56 mm body).
- Keep anatomical prior additions lightweight -- no new decoder layers, no extra model copies.

## Design Axes

### Category A -- Exploit & Extend

**Axis A1: Kinematic soft-attention bias in the refine decoder.**
Apply the kinematic-chain soft self-attention bias from idea019/design002 to the **refine decoder only** (not the coarse decoder). This is a natural architectural fit: the refine decoder's 2-layer self-attention among joint queries is where anatomical structure should be most useful, since the coarse predictions are already available. Implementation: precompute a `(70, 70)` hop-distance matrix from `SMPLX_SKELETON`, convert to additive bias (+1.0 for 1-hop, +0.5 for 2-hop, +0.25 for 3-hop, 0 beyond), scale by a learnable scalar initialized at 0.0, and pass as `tgt_mask` to the `refine_decoder` only. The coarse decoder keeps `tgt_mask=None`.

*Derives from:* idea019/design002 (kinematic bias, best body 105.77) ported to the stronger idea015/design004 baseline.

**Axis A2: Joint-group query injection before the refine decoder.**
Add a learnable group embedding `(4, 384)` that is added to `queries2` before the refine decoder pass. Groups: torso (pelvis, spine, neck, head = joints 0-3), arms (shoulders, elbows, wrists = joints 4-9), legs (hips, knees, ankles = joints 10-15), extremities (remaining joints 16-69). Zero-initialized so training starts identical to baseline. This gives the refine decoder an anatomical grouping prior.

*Derives from:* idea019/design004 (joint-group init, best weighted 102.22) ported to the stronger baseline.

### Category B -- Novel Exploration

**Axis B1: Bone-length loss on the refine decoder output only.**
Add a bone-length auxiliary loss that penalizes deviations from per-batch mean bone lengths. Compute `bone_loss = mean(|pred_bone_len - gt_bone_len|)` over `SMPLX_SKELETON` edges restricted to `BODY_IDX` (joints 0-21). Applied to J2 only (not J1), with `lambda_bone=0.05` (half the weight used in idea019, which was 0.1 and neutral). The two-decoder architecture produces better J2 predictions than the shared-decoder, so the bone constraint operates in a lower-error regime where it should be more effective. Loss = `0.5*L(J1) + 1.0*L(J2) + 0.05*bone_loss(J2)`.

**Axis B2: Kinematic bias + joint-group injection (careful pair combination).**
Combine A1 and A2 only: kinematic soft-attention bias in the refine decoder + joint-group query injection before the refine decoder. This is the strongest pair from idea019 (design002 and design004 were the two best). The combined priors were NOT tested as a pair in idea019 (design005 combined three priors including symmetry loss, which hurt). This pair combination is strictly new. No bone-length loss or symmetry loss is added, avoiding the over-regularization seen in idea019/design005.

## Expected Designs

The Designer should generate **4** novel designs:

1. **Kinematic soft-attention bias in refine decoder only** (Axis A1).
2. **Joint-group query injection before refine decoder** (Axis A2).
3. **Bone-length loss on J2 with lambda=0.05** (Axis B1).
4. **Kinematic bias + joint-group injection combined** (Axis B2).

## Design Constraints

- All designs start from `runs/idea015/design004/code/` and copy its full configuration: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False.
- The two-decoder architecture (4-layer coarse + 2-layer refine with independent weights), continuous sqrt-spaced depth PE (16 anchors), and wide head (hidden_dim=384, num_heads=8) MUST NOT be modified.
- Deep supervision base weights remain `0.5*L(J1) + 1.0*L(J2)` for all designs. Additional loss terms are additive.
- Use `SMPLX_SKELETON` from `infra.py` for bone connectivity. Use `BODY_IDX = list(range(22))` for body joint indexing.
- Soft biases (A1) must be additive, never produce fully -inf rows, and start with a learned scalar initialized at 0.0.
- New learnable parameters (group embeddings, bias scalar) go into the head optimizer group (LR=1e-4, weight_decay=0.3).
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- Do not modify `infra.py` or the transforms.
- For the kinematic bias (A1, B2): the bias is applied ONLY to `self.refine_decoder`, NOT to `self.decoder`. The coarse decoder retains `tgt_mask=None`. Since `nn.TransformerDecoder` does not directly expose `tgt_mask` per-layer, the Designer must implement a manual loop over `self.refine_decoder.layers` to pass the bias as `tgt_mask` to each layer's forward call. Alternatively, pass `tgt_mask` to `self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)` which applies it to all layers.

## Memory Budget Note

All 4 designs add negligible parameters:
- A1: one `(70, 70)` buffer + one scalar parameter (~5K total).
- A2: one `(4, 384)` embedding (~1.5K parameters).
- B1: zero new parameters (loss computation only).
- B2: combination of A1 + A2, ~6.5K new parameters.

All designs fit comfortably within 11 GB at batch=4.
