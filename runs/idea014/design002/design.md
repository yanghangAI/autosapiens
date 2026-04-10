# Design 002 — LLRD + Depth PE + Wide Head (Triple Combination)

## Starting Point

`runs/idea008/design003/`

This design starts from idea008/design003 (val_mpjpe_weighted = 112.0 mm, pelvis = 93.7 mm), which implements continuous depth PE with sqrt-spaced anchors. Two additional proven improvements are layered on top: LLRD from idea004/design002 and the wide head from idea009/design002.

## Problem

The pipeline has identified three independently strong improvements targeting non-overlapping model components:

| Improvement | Component | Best Result |
|---|---|---|
| LLRD (gamma=0.90, unfreeze=5) | Optimizer schedule | 112.3 mm body (idea004/d002) |
| Continuous depth PE (sqrt) | Backbone positional encoding | 112.0 mm weighted (idea008/d003) |
| Wide head (hidden=384) | Decoder head | 112.3 mm body (idea009/d002) |

idea011 tested LLRD + depth PE (2 of 3). This design is the first full triple combination.

## Proposed Solution

Apply both the LLRD schedule and the wide head to the idea008/design003 codebase:

1. **Wide head**: `head_hidden=384`, `head_num_heads=8`, `head_num_layers=4`, `dim_feedforward=1536`. `input_proj Linear(1024->384)`, `joint_queries Embedding(70,384)`, output linears updated.
2. **LLRD**: `gamma=0.90`, `base_lr_backbone=1e-4`, `unfreeze_epoch=5`. Per-layer LR = `1e-4 * 0.90^(23-i)` for block `i` (0=shallowest, 23=deepest).
3. **Continuous depth PE**: sqrt-spaced anchors, 16 depth bins, interpolation -- all unchanged from idea008/design003.

### LLRD Schedule Detail

**Phase 1 (epochs 0-4): Frozen lower backbone**
- Blocks 0-11 + patch/pos embeddings: frozen (`requires_grad=False`), omitted from optimizer.
- Blocks 12-23: 12 param groups, `lr_i = 1e-4 * 0.90^(23-i)`.
- Depth PE params (`row_emb`, `col_emb`, `depth_emb`): 1 group, `lr = 1e-4`.
- Head params: 1 group, `lr = 1e-4`.
- Total param groups: 14 (12 backbone blocks + depth_pe + head).

**Phase 2 (epoch 5+): All unfrozen**
- Patch+pos embedding: 1 group, `lr = 1e-4 * 0.90^24 ~ 8.014e-6`.
- Blocks 0-23: 24 groups, `lr_i = 1e-4 * 0.90^(23-i)`.
- Depth PE params: 1 group, `lr = 1e-4`.
- Head params: 1 group, `lr = 1e-4`.
- Total param groups: 27 (1 embed + 24 blocks + depth_pe + head).

At unfreeze, the optimizer is rebuilt from scratch with all groups. `initial_lr` is set per group, then the current cosine scale is applied.

### LR for Depth PE Params

The depth PE parameters (`row_emb`, `col_emb`, `depth_emb`) are new/task-specific modules (not pretrained). They get head-level LR (`1e-4`) in their own param group, same as in idea008/design003. They are NOT subject to LLRD decay.

## LR Schedule

Linear warmup for 3 epochs, cosine decay thereafter. Applied multiplicatively to all `initial_lr` values.

## Exact File-Level Edit Plan

### `code/config.py`

- `head_hidden = 384` (from 256).
- `lr_backbone = 1e-4` (from 1e-5; this is now the LLRD base rate for the deepest block).
- Add `gamma = 0.90`.
- Add `unfreeze_epoch = 5`.
- `output_dir` updated.

### `code/model.py`

No changes. `Pose3DHead` parameterizes all sizes through `hidden_dim`. Continuous depth PE is unchanged.

### `code/train.py`

Must implement the LLRD optimizer logic:

1. **Collect per-block param groups**: iterate `model.backbone.vit.layers` (24 blocks), assign `lr_i = base_lr * gamma^(23-i)`.
2. **Separate depth PE params** (`row_emb`, `col_emb`, `depth_emb`) into their own group at `lr = 1e-4`.
3. **Freeze blocks 0-11 + embeddings** at startup.
4. **Unfreeze + rebuild optimizer** at epoch 5.
5. **Cosine schedule** with warmup applied to all groups via `initial_lr`.

This follows the exact pattern from idea004/design002's train.py.

### `code/transforms.py`

No changes.

### `code/infra.py`

No changes.

## `config.py` Fields

Set these fields explicitly:

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea014/design002"
arch            = "sapiens_0.3b"
img_h           = IMG_H
img_w           = IMG_W
head_hidden     = 384      # widened from 256
head_num_heads  = 8
head_num_layers = 4
head_dropout    = 0.1
drop_path       = 0.1
epochs          = 20
lr_backbone     = 1e-4     # LLRD base rate for deepest block
lr_head         = 1e-4
lr_depth_pe     = 1e-4
gamma           = 0.90     # LLRD decay factor
unfreeze_epoch  = 5        # progressive unfreezing epoch
weight_decay    = 0.03
warmup_epochs   = 3
num_depth_bins  = 16
grad_clip       = 1.0
lambda_depth    = 0.1
lambda_uv       = 0.2
```

## Builder Implementation Notes

1. Start from `runs/idea008/design003/` using `setup-design`.
2. In `config.py`: set `head_hidden=384`, `lr_backbone=1e-4`, add `gamma=0.90` and `unfreeze_epoch=5`, update `output_dir`.
3. In `train.py`: port the LLRD optimizer logic from `runs/idea004/design002/code/train.py`. The key additions are: per-block param groups with decayed LR, freezing blocks 0-11 at startup, optimizer rebuild at epoch 5, and ensuring depth PE params are in their own group at head-level LR.
4. In `model.py`: no changes needed. The wide head propagates from `head_hidden=384` in config.
5. Do not modify the continuous depth PE logic (sqrt spacing, interpolation, anchors).
6. Do not modify `infra.py` or the loss formulation.
7. Verify that after optimizer rebuild at epoch 5, depth PE params still get `lr=1e-4` (not subject to LLRD decay).

## Parameter Count

Same as design001: ~9.88M head params + depth PE overhead. ~308M total. Well within VRAM at batch=4.

## Expected Outcome

This is the most promising design. LLRD protects pretrained backbone features, continuous depth PE improves pelvis localization, and the wide head increases decoder capacity. Expecting val_mpjpe_weighted below 112.0 mm and val_mpjpe_body below 112.3 mm, beating both idea008/design003 and idea004/design002.
