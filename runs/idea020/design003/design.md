# idea020 / design003 — L1 Loss on Refinement Pass Only (Axis B1)

## Starting Point

`runs/idea015/design004/code/`

## Problem

Both passes in idea015/design004 use the same Smooth L1 loss (beta=0.05) via `pose_loss()` from `infra.py`. Smooth L1 with beta=0.05 applies smooth quadratic behavior for errors below 5 cm and linear for errors above 5 cm. For the refinement pass (J2), errors should already be moderate (J2 is correcting J1), so the transition region at 5 cm is frequently active, smoothing out gradients at error sizes that the refinement pass needs to handle aggressively. Pure L1 provides constant-magnitude gradients at all error sizes, which may be more effective for a refinement stage working on already-reduced errors.

## Proposed Solution

Replace the Smooth L1 loss for J2 with pure `F.l1_loss`. The coarse pass retains Smooth L1 (beta=0.05) via `pose_loss()`:

```python
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])  # Smooth L1, unchanged
l_pose2 = F.l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])         # Pure L1 for refinement
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
```

## Change Required

**train.py only** — add `import torch.nn.functional as F` (already present) and change the `l_pose2` line:

```python
# BEFORE (in idea015/design004):
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2

# AFTER (design003):
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2 = F.l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
```

Note: `F` is already imported in `train.py` as part of PyTorch. Verify at top of file; if not, add `import torch.nn.functional as F`.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea020/design003"

# Loss weights (unchanged)
refine_loss_weight    = 0.5   # coarse pass weight unchanged
lambda_depth    = 0.1
lambda_uv       = 0.2

# Architecture and training all unchanged
arch            = "sapiens_0.3b"
head_hidden     = 384
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
num_depth_bins  = 16
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
```

## Implementation Notes

- **train.py**: Replace `l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])` with `l_pose2 = F.l1_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])`. Add `import torch.nn.functional as F` at top if not present (check baseline).
- **model.py**: Unchanged.
- **config.py**: Only `output_dir` changes.

## New Parameters

Zero. Identical parameter count to `runs/idea015/design004`.

## Expected Effect

The refinement pass receives constant-magnitude gradients at all error scales, which should make the refine decoder more aggressive in correcting medium-sized errors (10–50 mm range). The coarse pass retains Smooth L1, which is more stable for the wider range of coarse errors early in training. The pure L1 loss values will be on the same order of magnitude as Smooth L1 since both compute mean absolute-value-like quantities.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4).
