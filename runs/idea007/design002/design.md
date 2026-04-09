# Design 002 — Strong LLRD on Depth-Bucket PE (`gamma=0.90`, `unfreeze_epoch=5`)

## Starting Point

`runs/idea005/design001/`

## Problem

The best completed architecture already injects depth-aware positional information, but
it may still overwrite shallow pretrained RGB structure too aggressively during early
fine-tuning. `idea004/design002` suggested stronger decay can help preserve those early
features.

## Proposed Solution

Use the same architecture as `runs/idea005/design001`, but adopt the stronger LLRD
schedule from `idea004/design002`.

### LLRD Schedule

- `num_blocks = 24`
- `gamma = 0.90`
- deepest block base LR: `lr_backbone = 1e-5`
- head LR: `lr_head = 1e-4`
- depth-bucket PE LR: `lr_depth_pe = 1e-4`
- unfreeze epoch: `5`

Per-block LR:

```python
lr_i = lr_backbone * gamma ** (23 - i)
```

Embedding group LR:

```python
lr_embed = lr_backbone * gamma ** 24
```

### Freeze / Unfreeze Policy

- Epochs `0-4`: freeze `vit.patch_embed`, frozen zeroed `vit.pos_embed`, and blocks `0-11`
- Epochs `5-19`: unfreeze the full backbone and rebuild the optimizer

### Optimizer Groups

Epochs `0-4`:
- depth-bucket PE group
- blocks `12-23` as separate groups
- head group

Epochs `5-19`:
- embedding group: `patch_embed` only
- blocks `0-23` as separate groups
- depth-bucket PE group
- head group

### Important Builder Notes

1. Keep the custom `DepthBucketPE` architecture unchanged.
2. Do **not** place `depth_bucket_pe` parameters in the backbone block groups.
3. Do **not** include the frozen zero-buffer `vit.pos_embed` in optimizer groups.
4. Rebuild the optimizer exactly once at `epoch == unfreeze_epoch` before LR scaling.

## Files to Modify

- `code/train.py`
- `code/config.py`

`code/model.py` and `code/transforms.py` remain unchanged from the starting point.

## Config Fields

```python
output_dir       = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea007/design002"
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
unfreeze_epoch   = 5
```
