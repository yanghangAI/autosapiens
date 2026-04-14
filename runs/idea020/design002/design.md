# idea020 / design002 — Reduced Coarse Supervision Weight 0.1 (Axis A2)

## Starting Point

`runs/idea015/design004/code/`

## Problem

The deep supervision ratio `0.5*L(J1) + 1.0*L(J2)` was chosen in idea015/design004 without tuning. The coarse loss weight of 0.5 is relatively high, strongly supervising the coarse decoder to produce accurate joint predictions directly. This may prevent the coarse decoder from developing intermediate feature representations (`out1`) that are maximally useful for the refinement pass — the coarse decoder may trade internal expressiveness for direct prediction accuracy.

## Proposed Solution

Reduce the coarse supervision weight from 0.5 to 0.1:

```
loss = 0.1 * L(J1) + 1.0 * L(J2)
```

The coarse loss is reduced to a minimal regularizer that prevents degenerate feature collapse, while allowing the coarse decoder to optimize its representations primarily for the refinement pass's benefit. The refinement pass remains at weight 1.0 (full supervision).

## Change Required

**train.py only** — one-line change in `train_one_epoch()`:

```python
# BEFORE (in idea015/design004):
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2

# AFTER (design002):
l_pose  = 0.1 * l_pose1 + 1.0 * l_pose2
```

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir` and `refine_loss_weight`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea020/design002"

refine_loss_weight    = 0.1   # CHANGED from 0.5: reduced coarse supervision

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

- **train.py**: Change `l_pose = 0.5 * l_pose1 + 1.0 * l_pose2` to `l_pose = 0.1 * l_pose1 + 1.0 * l_pose2`. No other changes.
- **model.py**: Unchanged.
- **config.py**: Update `output_dir` and `refine_loss_weight = 0.1`.

Note: The Builder should use `args.refine_loss_weight` from config if wired up, or hardcode `0.1` in the loss line. The simplest change is a direct constant in `train.py`.

## New Parameters

Zero. Identical parameter count to `runs/idea015/design004`.

## Expected Effect

The coarse decoder is free to develop richer intermediate features since it only needs to loosely satisfy the 0.1-weighted coarse loss. The refinement pass, receiving better intermediate features `out1` as its query initialization, should produce better J2. May also reduce overfitting since the coarse decoder is not over-specialized for direct prediction.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4).
