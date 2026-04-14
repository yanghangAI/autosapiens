# idea020 -- Refinement-Specific Loss and Gradient Strategy

**Expected Designs:** 5

## Starting Point

The baseline starting point for this idea is:

`runs/idea015/design004/code/`

That design is the current overall SOTA: **val_mpjpe_body = 102.51 mm**, **val_pelvis = 91.62 mm**, **val_mpjpe_weighted = 98.92 mm**. It combines LLRD (gamma=0.90, unfreeze=5), continuous sqrt-spaced depth PE, a wide head (hidden_dim=384, 4 layers, 8 heads), weight_decay=0.3, and **two-pass iterative refinement with an independent 2-layer refine decoder** (separate from the 4-layer coarse decoder). The coarse pass outputs J1 via `joints_out`, then the refinement MLP injects J1 back into the query space, and the independent 2-layer refine decoder produces J2 via `joints_out2`. Deep supervision: `0.5*L(J1) + 1.0*L(J2)`.

## Concept

The two-decoder refinement (idea015/design004) outperforms the shared-decoder approach (idea015/design001) by 3 mm on weighted MPJPE. However, the loss function, deep supervision weighting, and gradient flow between the two passes have never been varied. The coarse decoder and refine decoder currently share the same loss type (Smooth L1 beta=0.05), the same deep supervision ratio (0.5:1.0), and the same optimizer group (head LR=1e-4). There are several promising axes:

1. **Detaching the coarse prediction from the refinement input** (stop gradient on J1 before the refine MLP) — forces the refine decoder to work with a fixed input signal rather than co-adapting with the coarse decoder. This is standard in cascaded detection (e.g., Cascade R-CNN).
2. **Varying the deep supervision weight** — the 0.5:1.0 ratio was chosen without tuning. A lower coarse weight (or zero) may free the coarse decoder to produce more informative features even if its direct pose predictions are less accurate.
3. **Using a different loss on the refinement pass** — L1 loss (no smoothing) on J2 provides stronger gradients for medium-sized errors, which is appropriate for the refinement pass where errors should already be moderate.
4. **Applying higher LR to the refine decoder** — the refine decoder has only 2 layers and is initialized from scratch; it may benefit from a faster learning rate than the coarse decoder.
5. **Residual refinement formulation** — instead of predicting absolute J2, predict a delta from J1 (i.e., `J2 = J1 + joints_out2(out2)`). This structures the refinement task as residual correction, which should be easier to learn for small errors.

All axes are pure training-loop or loss-function changes (plus one minor model forward change for the residual formulation), fitting comfortably within 11 GB with no new modules beyond what idea015/design004 already has.

## Broader Reflection

### Strong prior results to build on

- **idea015/design004** (val_weighted=**98.92**, val_body=**102.51**, pelvis=**91.62**) — two-decoder refinement is the confirmed SOTA. The independent refine decoder is the key innovation that works.
- **idea015/design001** (val_weighted=**101.94**, pelvis=**89.95**) — shared-decoder + query injection. Slightly better pelvis but 3 mm worse overall. The pelvis advantage may come from the simpler single-decoder gradient flow.
- **idea019/design002** (val_body=**105.77**) — kinematic soft-attention bias on idea015/design001. Marginal gain on body MPJPE. Anatomical priors work but are not transformative.
- **idea004/design002** (val_body=**112.32**) — LLRD alone. Shows the backbone fine-tuning schedule is critical and already optimized.

### Patterns to avoid

- **Adding parameters or extra model copies** — OOM risk is real (idea017, idea018 all failed). This idea adds zero new parameters.
- **Three-pass refinement** (idea015/design003) — failed badly; multi-pass beyond 2 does not work.
- **Heatmap representations** (idea016) — dead end.
- **Temporal fusion** (idea017) — does not fit in 11 GB.

### Key insight

The train-val gap for idea015/design004 is ~30 mm (72 train vs 102 val body). The refinement pass has its own train-val gap. By changing how gradients flow through the two-pass system and what the refinement pass optimizes for, we can attack the overfitting problem from the loss/optimization side without adding regularization (which was already tried in idea012 with limited success).

## Design Axes

### Category A -- Exploit & Extend

**Axis A1: Stop-gradient on coarse prediction before refinement MLP.**
Detach J1 before passing it to `self.refine_mlp()`. The forward pass becomes:
```python
R = self.refine_mlp(J1.detach())  # stop gradient from refine decoder flowing back to coarse decoder via J1
queries2 = out1 + R
```
This decouples the two decoders' gradient flows. The coarse decoder optimizes only for its own `0.5*L(J1)` loss. The refine decoder sees a stable input signal. Standard practice in cascaded detection.

*Derives from:* idea015/design004 (refining the two-decoder architecture's gradient flow).

**Axis A2: Reduced coarse supervision weight (0.1 instead of 0.5).**
Change deep supervision from `0.5*L(J1) + 1.0*L(J2)` to `0.1*L(J1) + 1.0*L(J2)`. This relaxes the constraint on the coarse decoder, allowing it to optimize its intermediate representations for the refinement pass rather than for direct pose accuracy. The coarse loss still provides a minimal supervision signal to avoid degenerate features.

*Derives from:* idea015/design004 (tuning the deep supervision hyperparameter, which was never varied).

### Category B -- Novel Exploration

**Axis B1: L1 loss (no smoothing) on the refinement pass only.**
Replace the Smooth L1 loss for J2 with pure L1 loss:
```python
l_pose2 = F.l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
```
The coarse pass retains Smooth L1 (beta=0.05). Rationale: by the refinement stage, errors are moderate (not huge), so the Smooth L1 transition zone (beta=0.05) may smooth out gradients unnecessarily. Pure L1 gives constant-magnitude gradients for all error sizes, which should help the refinement pass make consistent corrections.

**Axis B2: Higher LR for the refine decoder (2x head LR).**
Create a separate optimizer group for `refine_decoder`, `refine_mlp`, and `joints_out2` with LR = 2e-4 (twice the head LR of 1e-4). These modules are initialized from scratch and have fewer layers; they may benefit from faster learning. The coarse decoder retains LR = 1e-4. Implementation: in `_build_optimizer_frozen()` and `_build_optimizer_full()`, split `model.head.parameters()` into coarse-head params and refine-head params.

**Axis B3: Residual refinement formulation.**
Change the model's second-pass output from absolute prediction to residual correction:
```python
# In Pose3DHead.forward():
delta = self.joints_out2(out2)  # (B, 70, 3) — predicted correction
J2 = J1 + delta                 # absolute = coarse + correction
```
Loss is still applied to J2 (absolute prediction). This structural change means `joints_out2` only needs to predict small corrections (typically < 0.05m) rather than full joint positions (~0.1-0.5m). Zero-initialization of `joints_out2` weights is already the default, making J2 = J1 at initialization (smooth warmup). Pelvis outputs should still use `out2[:, 0, :]` features (not J2) for `depth_out` and `uv_out` since those are feature-based, not coordinate-based.

## Expected Designs

The Designer should generate **5** novel designs:

1. **Stop-gradient on coarse J1 before refinement** (Axis A1).
2. **Reduced coarse supervision weight 0.1** (Axis A2).
3. **L1 loss on refinement pass only** (Axis B1).
4. **Higher LR for refine decoder (2x)** (Axis B2).
5. **Residual refinement formulation** (Axis B3).

## Design Constraints

- All designs start from `runs/idea015/design004/code/` and copy its optimizer, LR schedule, LLRD config (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4), weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2.
- The two-decoder architecture (4-layer coarse + 2-layer refine), continuous sqrt-spaced depth PE (16 anchors), wide head (hidden_dim=384, num_heads=8), and LLRD schedule MUST NOT be modified (except where a specific design axis requires a targeted change).
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- Do not modify `infra.py` or the transforms.
- For design B2 (higher refine LR), the Designer must carefully split `model.head.parameters()` into two optimizer groups: one for coarse-head params (decoder, input_proj, joint_queries, joints_out, depth_out, uv_out) at LR=1e-4, and one for refine params (refine_decoder, refine_mlp, joints_out2) at LR=2e-4.
- For design B3 (residual refinement), the change is only in `Pose3DHead.forward()` — replace `J2 = self.joints_out2(out2)` with `J2 = J1 + self.joints_out2(out2)`. No other changes needed.

## Memory Budget Note

All 5 designs add zero new parameters. The only changes are to loss computation, gradient flow, optimizer grouping, or a single line in the forward pass. All fit identically to idea015/design004 in 11 GB at batch=4.
