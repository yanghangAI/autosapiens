# idea020 / design004 — Higher LR for Refine Decoder (2x Head LR, Axis B2)

## Starting Point

`runs/idea015/design004/code/`

## Problem

In idea015/design004, all head parameters (`model.head.parameters()`) are placed in a single optimizer group with `lr_head = 1e-4`. This includes both the coarse 4-layer decoder (which has a learned warm-start from early training epochs) and the refine branch (refine_decoder, refine_mlp, joints_out2), which is randomly initialized from scratch. A randomly initialized 2-layer module may benefit from a higher learning rate early in training to converge faster toward its optimization target.

## Proposed Solution

Split `model.head.parameters()` into two separate optimizer groups:
1. **Coarse-head group** (LR = 1e-4): `input_proj`, `joint_queries`, `decoder`, `joints_out`, `depth_out`, `uv_out`.
2. **Refine-head group** (LR = 2e-4): `refine_decoder`, `refine_mlp`, `joints_out2`.

The refine-head group uses 2× the head learning rate. All other aspects (LLRD schedule, weight decay, gradient clip) are unchanged.

## Change Required

**train.py only** — modify `_build_optimizer_frozen()` and `_build_optimizer_full()` to split the head group:

```python
def _coarse_head_params():
    """Parameters of the coarse pass (not refine branch)."""
    head = model.head
    return (
        list(head.input_proj.parameters()) +
        list(head.joint_queries.parameters()) +
        list(head.decoder.parameters()) +
        list(head.joints_out.parameters()) +
        list(head.depth_out.parameters()) +
        list(head.uv_out.parameters())
    )

def _refine_head_params():
    """Parameters of the refine branch only."""
    head = model.head
    return (
        list(head.refine_decoder.parameters()) +
        list(head.refine_mlp.parameters()) +
        list(head.joints_out2.parameters())
    )

LR_REFINE = args.lr_head * 2.0   # 2e-4
```

In both `_build_optimizer_frozen()` and `_build_optimizer_full()`, replace:
```python
groups.append({"params": list(model.head.parameters()), "lr": args.lr_head})
```
with:
```python
groups.append({"params": _coarse_head_params(), "lr": args.lr_head})
groups.append({"params": _refine_head_params(), "lr": LR_REFINE})
```

LR schedule reporting: the frozen-phase LR groups shift by one (head group was group 13, now groups 13+14). Update the LR reporting logic accordingly:

```python
# Frozen phase: groups 0..11=blocks 12-23, group 12=depth_pe, group 13=coarse_head, group 14=refine_head
# Full phase: groups 0=embed, 1-24=blocks 0-23, group 25=depth_pe, group 26=coarse_head, group 27=refine_head
if epoch < UNFREEZE_EPOCH:
    lr_bb = optimizer.param_groups[11]["lr"]   # block 23
    lr_hd = optimizer.param_groups[13]["lr"]   # coarse head
else:
    lr_bb = optimizer.param_groups[24]["lr"]   # block 23
    lr_hd = optimizer.param_groups[26]["lr"]   # coarse head
```

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir` and add `lr_refine_head`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea020/design004"

lr_refine_head = 2e-4   # ADDED: refine decoder LR (2x lr_head)

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

- **train.py**: Add `_coarse_head_params()` and `_refine_head_params()` helper functions. Update both `_build_optimizer_frozen()` and `_build_optimizer_full()` to use two head groups. Update LR group index references in the epoch-reporting block.
- **model.py**: Unchanged.
- **config.py**: Add `lr_refine_head = 2e-4`. Update `output_dir`.

## New Parameters

Zero new parameters. Identical parameter count to `runs/idea015/design004`. Only the optimizer grouping changes.

## Expected Effect

The refine branch (randomly initialized) receives 2× the gradient step size of the coarse decoder, allowing it to converge faster toward its refinement objective in the first few epochs. Since the refine decoder is 2 layers (vs. 4 layers for coarse), it has fewer parameters to learn and higher LR may be well-tolerated. This should particularly improve early-epoch refinement quality.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4). An additional optimizer state tensor is added for the separate refine group, but this is negligible (~few MB).
