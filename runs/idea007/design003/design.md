# Design 003 — Strong LLRD with Earlier Unfreeze (`gamma=0.90`, `unfreeze_epoch=3`)

## Starting Point

`runs/idea005/design001/`

## Problem

The stronger `gamma=0.90` schedule may be beneficial, but waiting until epoch 5 to
unfreeze the whole backbone could be too conservative once the model already has explicit
depth-aware positional encoding. This design tests earlier full-backbone adaptation while
retaining strong shallow-layer protection during the warmup window.

## Proposed Solution

Keep the architecture from `runs/idea005/design001` unchanged and reuse the strong LLRD
schedule from Design 002, but move the full-backbone unfreeze earlier.

### LLRD Schedule

- `num_blocks = 24`
- `gamma = 0.90`
- deepest block base LR: `lr_backbone = 1e-5`
- head LR: `lr_head = 1e-4`
- depth-bucket PE LR: `lr_depth_pe = 1e-4`
- unfreeze epoch: `3`

Per-block LR:

```python
lr_i = lr_backbone * gamma ** (23 - i)
```

Embedding group LR:

```python
lr_embed = lr_backbone * gamma ** 24
```

### Freeze / Unfreeze Policy

- Epochs `0-2`: freeze `vit.patch_embed`, frozen zeroed `vit.pos_embed`, and blocks `0-11`
- Epochs `3-19`: unfreeze the full backbone and rebuild the optimizer

### Optimizer Groups

Epochs `0-2`:
- depth-bucket PE group
- blocks `12-23` as separate groups
- head group

Epochs `3-19`:
- embedding group: `patch_embed` only
- blocks `0-23` as separate groups
- depth-bucket PE group
- head group

### Important Builder Notes

1. Keep `DepthBucketPE` and the modified backbone forward exactly as in
   `runs/idea005/design001`.
2. The earlier unfreeze must happen before that epoch's LR scaling is applied.
3. Keep all random seeds hardcoded to `2026`.

## Files to Modify

- `code/train.py`
- `code/config.py`

`code/model.py` and `code/transforms.py` remain unchanged from the starting point.

## Config Fields

```python
output_dir       = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea007/design003"
arch             = "sapiens_0.3b"
img_h            = IMG_H
img_w            = IMG_W
head_hidden      = 256
head_num_heads   = 8
head_num_layers  = 4
head_dropout     = 0.1
drop_path        = 0.1
epochs           = 20
lr_backbone      = 1e-5
lr_head          = 1e-4
lr_depth_pe      = 1e-4
weight_decay     = 0.03
warmup_epochs    = 3
grad_clip        = 1.0
lambda_depth     = 0.1
lambda_uv        = 0.2
num_depth_bins   = 16
gamma            = 0.90
unfreeze_epoch   = 3
```
