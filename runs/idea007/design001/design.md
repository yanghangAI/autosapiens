# Design 001 — Gentle LLRD on Depth-Bucket PE (`gamma=0.95`, `unfreeze_epoch=5`)

## Starting Point

`runs/idea005/design001/`

## Problem

`runs/idea005/design001` introduced depth-bucket positional embeddings and achieved the
best completed result so far, but it still fine-tunes the entire ViT backbone with a
single backbone learning rate. `idea004` showed that conservative layer-wise adaptation
can improve stability by protecting shallow pretrained blocks early in training.

## Proposed Solution

Keep the depth-bucket positional embedding architecture unchanged from
`runs/idea005/design001`, but replace the optimizer schedule with layer-wise learning-rate
decay (LLRD) and progressive unfreezing.

### LLRD Schedule

- `num_blocks = 24`
- `gamma = 0.95`
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

1. Keep the custom `DepthBucketPE` module and custom backbone forward from
   `runs/idea005/design001` unchanged.
2. Do **not** include the frozen zero-buffer `vit.pos_embed` in optimizer groups.
3. Depth-bucket PE parameters stay in their own high-LR group (`1e-4`) for all epochs.
4. Apply the existing linear-warmup + cosine-decay scale to every group's `initial_lr`.
5. Preserve deterministic seed `2026` and the existing no-heavy-augmentation setup.

## Files to Modify

- `code/train.py`
- `code/config.py`

`code/model.py` and `code/transforms.py` should remain unchanged from the starting point.

## Config Fields

```python
output_dir       = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea007/design001"
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
gamma            = 0.95
unfreeze_epoch   = 5
```
