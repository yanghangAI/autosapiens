# idea013 — Joint Prediction Loss Reformulation

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea004/design002/train.py`

That design produced the best completed validation body MPJPE at **112.3 mm** using LLRD (gamma=0.90, unfreeze_epoch=5). The LLRD schedule and model architecture are kept fixed across all designs so that gains are attributable purely to loss function changes.

## Concept

Every experiment in the pipeline uses the same loss function: **Smooth L1 (beta=0.05)** applied independently to each body joint's 3D coordinates. This formulation treats all joints equally, ignores inter-joint relationships, and uses a fixed robustness threshold. No prior idea has varied the loss function itself (idea003 varied loss *weights* between tasks, not the loss *formulation* for pose).

There are well-established alternative loss formulations for 3D pose regression that address known shortcomings of per-joint Smooth L1:

1. **Bone-length consistency losses** penalize violations of anatomical constraints between connected joints, encouraging structurally plausible poses.
2. **Adaptive per-joint weighting** can allocate more gradient signal to harder joints (extremities) vs. easier joints (torso).
3. **Wing loss** (designed for landmark regression) provides stronger gradients for small errors than Smooth L1, which has a linear gradient regime beyond beta.
4. **Velocity/smoothness priors** are not applicable here (single-frame prediction).

These changes modify only the loss computation in the training loop, requiring no architectural changes, no extra memory, and no changes to the evaluation protocol.

## Broader Reflection

### Strong prior results to build on

- **idea004/design002** (val_mpjpe_body = **112.3 mm**) is the best body MPJPE. Its LLRD schedule is the foundation here.
- **idea009/design002** (val_mpjpe_body = **112.3 mm**) matched the best with a wider head. Head architecture matters little — the loss function is a more fundamental axis.
- **idea012/design002** (val_mpjpe_body = **117.7 mm** at epoch 12, still training) shows weight decay helps generalization. Loss reformulation is complementary.

### Patterns to avoid

- **idea003** (curriculum loss weighting) showed that dynamic loss scheduling needs more than 20 epochs to stabilize. All designs here use fixed loss formulations (no epoch-dependent scheduling).
- **idea002** (kinematic masking) showed that hard structural constraints on attention hurt convergence. The bone-length loss here is a soft penalty, not a hard constraint.
- **idea006/design001** (horizontal flip augmentation) showed that changes affecting the coordinate system can cause catastrophic pelvis localization errors. Loss changes here operate on the already-computed root-relative joint coordinates, so they cannot introduce such artifacts.

## Design Axes

### Category A -- Exploit & Extend

**Axis A1: Smooth L1 with reduced beta.**
The baseline uses beta=0.05 m (50 mm), meaning errors below 50 mm are in the quadratic (L2-like) regime and errors above are in the linear (L1-like) regime. Reducing beta to 0.01 m (10 mm) makes the loss more L1-like for a wider range of errors, providing stronger gradients for medium-sized errors (10-50 mm range). This is a direct refinement of the only loss hyperparameter that has never been tuned.

*Derives from:* idea004/design002 (best result), refining the existing Smooth L1 beta parameter.

**Axis A2: Smooth L1 with increased beta.**
Conversely, increasing beta to 0.1 m (100 mm) makes the loss more L2-like for typical error magnitudes. This gives stronger gradients for large errors early in training, potentially accelerating initial convergence. The quadratic regime reduces gradient explosion from outlier frames.

*Derives from:* idea004/design002, exploring the opposite direction of Axis A1 to characterize the beta sensitivity.

### Category B -- Novel Exploration

**Axis B1: Bone-length auxiliary loss.**
Add a soft bone-length consistency penalty: for each bone (edge in SMPLX_SKELETON), compute the predicted bone length as `||pred[joint_i] - pred[joint_j]||` and the target bone length from the ground truth. The auxiliary loss is `L_bone = mean(|pred_bone_len - gt_bone_len|)` over all body skeleton edges. Total loss = `L_smooth_l1 + lambda_bone * L_bone` with `lambda_bone=0.1`. This encourages anatomically consistent predictions without modifying the architecture. The bone list is derived from `SMPLX_SKELETON` in `infra.py`, filtered to body joints only (indices 0-21).

**Axis B2: Hard-joint-weighted loss.**
Compute per-joint MPJPE on the training set for the first epoch, then apply inverse-frequency weighting: joints with higher average error receive proportionally more loss weight. Specifically, after epoch 0, compute `w_j = mean_error_j / mean(mean_error_all)` for each body joint j, then clamp weights to [0.5, 2.0] and normalize to sum to num_body_joints. Apply these fixed weights for epochs 1-19. This is a one-shot reweighting (not dynamic per-epoch) to avoid the instability seen in idea003.

## Expected Designs

The Designer should generate **4** novel designs:

1. **Small-beta Smooth L1 (beta=0.01)** -- Change `beta` from 0.05 to 0.01 in the Smooth L1 loss. Everything else identical to idea004/design002.
2. **Large-beta Smooth L1 (beta=0.1)** -- Change `beta` from 0.05 to 0.1 in the Smooth L1 loss. Everything else identical.
3. **Bone-length auxiliary loss** -- Add `lambda_bone=0.1 * mean(|pred_bone_len - gt_bone_len|)` summed over all body skeleton edges from `SMPLX_SKELETON`. Body edges only (both endpoints in joints 0-21). The bone lengths are computed as L2 norms of predicted and target joint position differences for each edge. Total loss = Smooth L1 (beta=0.05) + lambda_bone * L_bone.
4. **Hard-joint-weighted loss** -- After epoch 0, compute per-joint mean training error, derive fixed weights w_j = clamp(err_j / mean_err, 0.5, 2.0), normalize to sum to 22 (num body joints). Apply these weights element-wise to the per-joint Smooth L1 loss for epochs 1-19. Store weights as a buffer after epoch 0.

## Design Constraints

- Keep the LLRD schedule from idea004/design002 (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4) fixed across all designs.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- `grad_clip=1.0`.
- Do not modify `infra.py`, transforms, or the model architecture.
- For design 3 (bone loss): the bone list must be derived from `SMPLX_SKELETON` in `infra.py`. Filter to edges where both endpoints are body joints (new indices 0-21 after remapping). Pre-compute this edge list once at model initialization.
- For design 4 (hard-joint weighting): the weight computation happens once after epoch 0 completes. During epoch 0, accumulate per-joint L1 errors. At epoch boundary, compute and freeze weights. This requires a small modification to the training loop, not the model.
- All designs continue to evaluate with standard MPJPE (unweighted) for fair comparison.
