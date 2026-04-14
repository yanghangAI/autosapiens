# idea020 / design005 — Residual Refinement Formulation (Axis B3)

## Starting Point

`runs/idea015/design004/code/`

## Problem

In idea015/design004, the refine decoder's output head `joints_out2` predicts absolute joint coordinates `J2 ∈ R^(B×70×3)`, where values are in the range ~0.1–0.5 m. The refinement pass must learn to reproduce large absolute coordinate values from scratch, even though J1 is already a reasonable estimate. This makes the optimization landscape harder than it needs to be for a refinement stage.

## Proposed Solution

Change the refinement formulation from absolute prediction to **residual correction**:

```python
delta = self.joints_out2(out2)   # (B, 70, 3) — predicted correction (small values)
J2 = J1 + delta                  # absolute = coarse + correction
```

The loss is still applied to J2 (absolute coordinates). `joints_out2` only needs to predict small corrections (~0.01–0.05 m), which is a much easier task than predicting absolute coordinates. Because `joints_out2` is zero-initialized, at the start of training `J2 = J1` (the refinement passes through the coarse prediction unchanged), providing a smooth warm-start.

Note: `depth_out` and `uv_out` continue to operate on `out2[:, 0, :]` features (not on J2), so those heads are unchanged.

## Change Required

**model.py only** — two-line change in `Pose3DHead.forward()`:

```python
# BEFORE (in idea015/design004):
out2 = self.refine_decoder(queries2, memory)
J2   = self.joints_out2(out2)

# AFTER (design005):
out2  = self.refine_decoder(queries2, memory)
delta = self.joints_out2(out2)    # (B, 70, 3): predicted correction
J2    = J1 + delta                 # residual: absolute = coarse + correction
```

The rest of `Pose3DHead.forward()` is unchanged. `pelvis_token = out2[:, 0, :]` is unaffected.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea020/design005"

# Architecture (unchanged)
arch            = "sapiens_0.3b"
head_hidden     = 384
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
num_depth_bins  = 16
refine_passes         = 2
refine_decoder_layers = 2
refine_loss_weight    = 0.5

# Training (unchanged)
epochs          = 20
lr_backbone     = 1e-4
base_lr_backbone = 1e-4
llrd_gamma      = 0.90
unfreeze_epoch  = 5
lr_head         = 1e-4
lr_depth_pe     = 1e-4
weight_decay    = 0.3
warmup_epochs   = 3
grad_clip       = 1.0

# Loss weights (unchanged)
lambda_depth    = 0.1
lambda_uv       = 0.2
```

## Implementation Notes

- **model.py**: In `Pose3DHead.forward()`, replace `J2 = self.joints_out2(out2)` with:
  ```python
  delta = self.joints_out2(out2)
  J2    = J1 + delta
  ```
  No other model changes needed. `joints_out2` zero-initialization in `_init_weights()` is already the default (see `nn.init.zeros_(m.bias)` and `trunc_normal_(m.weight, std=0.02)`) — at init, the network output is approximately zero mean, so `J2 ≈ J1 + ε ≈ J1`. This is the smooth warm-start property.
- **train.py**: Unchanged. Loss is still `0.5 * l_pose1 + 1.0 * l_pose2` where `l_pose2` is computed on `out["joints"]` (= J2 = J1 + delta).
- **config.py**: Only `output_dir` changes.

## New Parameters

Zero. Identical parameter count to `runs/idea015/design004`. `joints_out2` already exists; its output interpretation changes from absolute to residual.

## Expected Effect

The residual formulation reduces the optimization burden on `joints_out2`: instead of predicting values in the ~0.1–0.5 m range, it predicts corrections in the ~0.01–0.05 m range. This leads to better-conditioned gradients early in training. Additionally, the smooth warm-start (J2 ≈ J1 at initialization) ensures the coarse loss `0.5*L(J1)` and refine loss `1.0*L(J2)` are initially equal, giving a stable training start.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4).
