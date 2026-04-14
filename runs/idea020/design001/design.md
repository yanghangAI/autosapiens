# idea020 / design001 — Stop-Gradient on Coarse J1 Before Refinement (Axis A1)

## Starting Point

`runs/idea015/design004/code/`

## Problem

In the two-pass refinement architecture, the refinement MLP receives `J1` as input and passes it to the independent refine decoder. Because `J1` is a differentiable output of the coarse decoder, gradients from the refinement branch flow back through `J1` into the coarse decoder. This creates a coupling: the coarse decoder co-adapts with the refinement branch to minimize the final J2 loss, which may compromise the coarse decoder's ability to produce a stable, informative feature representation `out1`.

## Proposed Solution

Detach `J1` before passing it to `self.refine_mlp()`. This stops gradient flow from the refinement branch back through the coarse decoder via `J1`. The two decoders now optimize fully independent objectives:
- Coarse decoder: optimized only via `0.5 * L(J1)`.
- Refine decoder: optimized via `1.0 * L(J2)`, receiving a stable, detached `J1` input.

This is standard practice in cascaded detection (e.g., Cascade R-CNN) to avoid feature corruption from co-adaptation.

## Change Required

**model.py only** — one-line change in `Pose3DHead.forward()`:

```python
# BEFORE (in idea015/design004):
R        = self.refine_mlp(J1)         # J1 carries gradients from coarse decoder
queries2 = out1 + R

# AFTER (design001):
R        = self.refine_mlp(J1.detach())  # stop gradient through J1
queries2 = out1 + R
```

Everything else (loss, optimizer, config) is unchanged.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea020/design001"

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
refine_loss_weight    = 0.5   # coarse pass weight

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

- **model.py**: Change `self.refine_mlp(J1)` to `self.refine_mlp(J1.detach())` in `Pose3DHead.forward()`. No other model changes.
- **train.py**: Unchanged. Loss is `0.5 * l_pose1 + 1.0 * l_pose2` as in baseline.
- **config.py**: Only `output_dir` changes.

## New Parameters

Zero. Identical parameter count to `runs/idea015/design004`.

## Expected Effect

The coarse decoder focuses entirely on its own `0.5*L(J1)` objective. The refine decoder receives a stable, fixed input from `J1.detach()` and optimizes its `1.0*L(J2)` objective independently. This should reduce the train-val gap by preventing co-adaptation overfitting between the two branches.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4).
