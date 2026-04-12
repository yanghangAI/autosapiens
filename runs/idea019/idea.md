# idea019 -- Anatomical Structure Priors for Iterative Refinement

**Expected Designs:** 5

## Starting Point

The baseline starting point for this idea is:

`runs/idea015/design001/code/`

That design is the current best completed run: **val_mpjpe_body = 107.85 mm**, **val_pelvis = 89.95 mm**, **val_mpjpe_weighted = 101.94 mm**. It combines LLRD (gamma=0.90, unfreeze=5), continuous sqrt-spaced depth PE, a wide head (hidden_dim=384, 4 layers, 8 heads), weight_decay=0.3, and two-pass iterative refinement with query injection via a small MLP. This idea explores incorporating anatomical structure priors into the iterative refinement process to reduce body joint localization error.

## Concept

Every design so far treats the 70 joint queries as independent, unstructured tokens. The model must learn all anatomical relationships (bone connectivity, symmetry, proportional constraints) purely from data within 20 epochs. Injecting lightweight structural priors -- especially into the refinement pass where coarse predictions are already available -- should help the model correct anatomically implausible configurations. This is orthogonal to all components already in the SOTA stack (LLRD, depth PE, wide head, iterative refinement) and modifies only how the decoder processes joint queries in the second refinement pass.

## Broader Reflection

### Strong prior results to build on

- **idea015/design001** (val_body=**107.85**, pelvis=**89.95**, weighted=**101.94**) -- new SOTA. Two-pass shared-decoder refinement with query injection MLP. The first iterative refinement to work. This is the starting point.
- **idea014/design003** (val_body=**106.85**, pelvis=**96.73**, weighted=**103.51**) -- triple combo baseline that idea015 extends. Slightly better body MPJPE but worse pelvis and weighted.
- **idea013/design003** (val_body=**112.99**, pelvis=**167.29**) -- bone-length auxiliary loss (lambda_bone=0.1) was tested on idea004/design002 baseline but did not significantly improve results there. However, it was tested on a weaker baseline without iterative refinement. The bone-length loss concept is sound but may need the right architecture to shine.
- **idea009/design002** (val_body=**112.34**) -- wide head confirmed useful; already incorporated in the SOTA stack.
- **idea002** (kinematic masking) -- hard attention masks on joint self-attention were tried early and did not help (val_body ~126-128 mm). However, those masks were binary and applied to the original baseline without any of the current improvements. Soft structural biases (not hard masks) are a different approach.

### Patterns to avoid

- **Hard binary attention masks** (idea002) -- hurt performance. Use only soft additive biases or loss-based priors.
- **idea016** (2.5D heatmaps) -- val_body ~210 mm, complete failure. Do not change the output representation.
- **idea017** (temporal fusion) -- still struggling at 150+ mm. Do not touch the input pipeline.
- **idea018** (weight averaging) -- failed. Do not add EMA/SWA.
- **idea015/design002** (Gaussian attention bias on cross-attention from coarse predictions) -- failed at epoch 6 with 130+ mm. Avoid complex cross-attention biases based on predicted spatial locations.
- **idea015/design003** (three-pass refinement) -- failed at epoch 5 with 184+ mm. Do not add a third pass.

### Promising direction

The gap between train and val body MPJPE for idea015/design001 is about 28 mm (79.5 train vs 107.8 val), indicating continued overfitting. The remaining val error (107.85 mm) is dominated by body joints -- not pelvis (89.95 mm is quite good). Anatomical priors could regularize the body joint predictions specifically: bone-length consistency, symmetric left-right constraints, and joint-group structure could all reduce the error on the hardest joints (e.g., wrists, ankles) without adding significant parameters.

## Design Axes

### Category A -- Exploit & Extend

**Axis A1: Bone-length auxiliary loss on the refinement output.**
Add a bone-length consistency loss that penalizes deviations from the mean bone lengths computed from the ground truth in each batch. Use `SMPLX_SKELETON` edges from `infra.py` restricted to `BODY_IDX` joints (0-21). Compute `bone_loss = mean(|pred_bone_len - gt_bone_len|)` over all skeleton edges for each sample. Apply only to the refined prediction `J2` (not `J1`) with weight `lambda_bone=0.1`. The deep supervision weights remain `0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2)`.

*Derives from:* idea013/design003 (bone-length loss concept) + idea015/design001 (iterative refinement SOTA). idea013 tested bone-length loss on a weaker baseline; this tests it on the SOTA with the hypothesis that refinement provides better coarse predictions for the bone constraint to act on.

**Axis A2: Kinematic-chain self-attention bias in the refinement pass.**
During the second decoder pass only, add a soft additive bias to the self-attention (tgt_mask) among joint queries based on kinematic distance in `SMPLX_SKELETON`. Joints that are 1-hop apart get bias +1.0, 2-hop get +0.5, 3-hop get +0.25, beyond 3 hops get 0.0. The bias is a fixed `(22, 22)` tensor (BODY_IDX only; non-body queries get 0 bias) registered as a buffer, scaled by a learnable scalar initialized at 0.0 so the model starts identical to baseline. This encourages nearby joints to share information during refinement without hard masking.

*Derives from:* idea002 (kinematic masking concept, which used hard masks and failed) + idea015/design001 (SOTA refinement). The key difference is (a) soft additive bias not hard mask, (b) applied only in the refinement pass, (c) learnable magnitude starting at zero.

### Category B -- Novel Exploration

**Axis B1: Left-right symmetry loss.**
Add a symmetry regularization term that penalizes asymmetric bone lengths between corresponding left-right joint pairs. Pairs: (L_shoulder/R_shoulder, L_elbow/R_elbow, L_wrist/R_wrist, L_hip/R_hip, L_knee/R_knee, L_ankle/R_ankle) -- 6 pairs total, identified by their indices in `BODY_IDX`. Symmetry loss = `mean(|left_bone_len - right_bone_len|)` over all 6 pairs of matching limb segments. Applied to `J2` only, weight `lambda_sym=0.05`. This is never tried before and provides a strong anatomical prior.

**Axis B2: Joint-group query initialization in the refinement pass.**
Before the second decoder pass, partition the 70 joint queries into 4 groups: torso (pelvis, spine, neck, head), arms (shoulders, elbows, wrists), legs (hips, knees, ankles), and extremities (hands, face). Add a learnable group embedding `(4, 384)` that is added to each query based on its group membership before the refinement pass. This gives the model a coarse spatial prior about which queries belong together anatomically. The group embedding is zero-initialized so training starts identical to baseline.

**Axis B3: Combined anatomical priors (bone-length + symmetry + kinematic bias).**
Combine A1, A2, and B1 into a single design to test whether the anatomical priors are complementary. Loss = `0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2) + 0.05*sym_loss(J2)`. Self-attention kinematic bias is applied in the refinement pass with learned scalar (init=0.0). This tests the maximum-prior configuration.

## Expected Designs

The Designer should generate **5** novel designs:

1. **Bone-length auxiliary loss on refinement output** (Axis A1).
2. **Kinematic-chain soft self-attention bias in refinement pass** (Axis A2).
3. **Left-right symmetry loss** (Axis B1).
4. **Joint-group query initialization in refinement pass** (Axis B2).
5. **Combined anatomical priors** (Axis B3: bone-length + symmetry + kinematic bias).

## Design Constraints

- All designs start from `runs/idea015/design001/code/` and copy its optimizer, LR schedule, LLRD config (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.
- The continuous sqrt-spaced depth PE (16 anchors), wide head (hidden_dim=384, num_heads=8, num_layers=4, FFN=1536), and two-pass iterative refinement with query injection MLP MUST NOT be modified.
- Deep supervision base weights remain `0.5*L(J1) + 1.0*L(J2)` for all designs. Additional loss terms are additive.
- Use `SMPLX_SKELETON` from `infra.py` for bone connectivity. Use `BODY_IDX = list(range(22))` for body joint indexing.
- Soft biases (A2) must be additive, never produce fully -inf rows, and start with a learned scalar initialized at 0.0.
- New learnable parameters (group embeddings, bias scalars) go into the `head_params` optimizer group (LR=1e-4, weight_decay=0.3).
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- Do not modify `infra.py` or the transforms.
- Memory: all designs add negligible parameters (<1M extra). No OOM risk at batch=4.

## Memory Budget Note

All 5 designs add minimal extra parameters:
- A1: zero new parameters (loss computation only).
- A2: one `(22, 22)` buffer + one scalar parameter.
- B1: zero new parameters (loss computation only).
- B2: one `(4, 384)` embedding (~1.5K parameters).
- B3: combination of above, still <10K new parameters total.

All designs fit comfortably within 11GB at batch=4.
