# idea021 / design003 — Bone-Length Loss on J2 with lambda=0.05 (Axis B1)

## Starting Point

`runs/idea015/design004/code/`

## Problem

In idea019/design001, bone-length auxiliary loss was neutral when applied on the shared-decoder (idea015/design001) SOTA baseline. However, the shared decoder produced coarser predictions than the two-decoder refinement baseline. The refine decoder in idea015/design004 produces more refined predictions (J2), meaning the per-prediction bone-length deviations are smaller and closer to anatomically realistic values. A bone-length constraint in this lower-error regime should be more effective — it acts as a soft regularizer that prevents the refine pass from distorting inter-joint distances during correction.

## Proposed Solution

Add a bone-length auxiliary loss on J2 only (not J1), with `lambda_bone = 0.05` (half the weight tested in idea019 where 0.1 was neutral). The loss penalizes the mean absolute deviation between predicted and ground-truth bone lengths for body joints only (BODY_IDX, joints 0-21).

```
total_loss = 0.5 * L(J1) + 1.0 * L(J2) + 0.05 * bone_loss(J2)
```

## Mathematical Formulation

```python
# Bone-length loss helper (add to train.py):
def bone_length_loss(pred_joints, gt_joints, edges):
    """
    Args:
        pred_joints: (B, N_joints, 3) — predicted joint positions (body only)
        gt_joints:   (B, N_joints, 3) — GT joint positions (body only)
        edges:       list of (i, j) pairs — bone connectivity within body joints
    Returns:
        scalar loss: mean |pred_bone_len - gt_bone_len|
    """
    losses = []
    for (i, j) in edges:
        pred_len = torch.norm(pred_joints[:, i] - pred_joints[:, j], dim=-1)  # (B,)
        gt_len   = torch.norm(gt_joints[:, i]   - gt_joints[:, j],   dim=-1)  # (B,)
        losses.append(torch.abs(pred_len - gt_len))
    return torch.stack(losses, dim=1).mean()  # mean over (B, num_edges)
```

The `edges` are SMPLX_SKELETON edges restricted to `a < 22` and `b < 22` (body joints only), using the re-indexed joint ordering from `infra.py`.

## Changes Required

**train.py**:
1. Import `SMPLX_SKELETON` from `infra` (add to existing imports):
   ```python
   from infra import (... SMPLX_SKELETON, ...)
   ```
2. Add `bone_length_loss()` helper function (see above).
3. Compute body-only edges once at the top of `main()` (before the epoch loop):
   ```python
   BODY_EDGES = [(a, b) for (a, b) in SMPLX_SKELETON if a < 22 and b < 22]
   ```
4. In `train_one_epoch()`, add the bone loss computation after the existing pose losses:
   ```python
   l_bone = bone_length_loss(
       out["joints"][:, BODY_IDX], joints[:, BODY_IDX], BODY_EDGES
   )
   l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_bone * l_bone
   ```
   Note: `BODY_IDX` is a slice object `slice(0, 22)` from infra. When indexing `BODY_EDGES`, joint indices `a, b` are already in body-relative space (0-21) since we filtered `a < 22 and b < 22`. The indexing `out["joints"][:, BODY_IDX]` → (B, 22, 3), so BODY_EDGES indices reference positions 0-21 of this sub-tensor correctly.

**model.py**: Unchanged.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir` and add `lambda_bone`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea021/design003"

lambda_bone = 0.05   # ADDED: bone-length auxiliary loss weight on J2

# All other fields unchanged from idea015/design004
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
lambda_depth    = 0.1
lambda_uv       = 0.2
```

## Implementation Notes

- The Builder must verify that `SMPLX_SKELETON` is importable from `infra.py` and that its indices are in the re-indexed space (not the original SMPLX ordering). Check `infra.py` for `_ORIG_TO_NEW` remapping — `SMPLX_SKELETON` should already be in the remapped index space.
- `BODY_IDX = slice(0, 22)` from infra. When creating `BODY_EDGES`, filter `(a, b)` with `a < 22 and b < 22`. The resulting indices directly index into `pred_joints[:, BODY_IDX]` (shape (B, 22, 3)).
- The bone loss is applied **only during training**, not during validation (`validate()` in infra.py is unchanged).

## New Parameters

Zero. Pure training-loop change.

## Expected Effect

The bone-length loss provides a soft anatomical regularizer specifically on the refined predictions J2. Unlike the coarser J1 predictions, J2 is in a regime where small deviations from correct bone lengths matter more. The reduced weight (0.05 vs. 0.1 in idea019) avoids over-constraining the refinement pass. This may particularly improve the body MPJPE metric by preventing small but systematic bone-length distortions introduced by the refinement decoder.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4). The bone loss computation is a lightweight O(num_edges) operation.
