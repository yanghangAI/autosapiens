# Design 001 — 6-Layer Decoder Head

## Starting Point

`runs/idea004/design002`

This design builds directly on idea004/design002 (val_mpjpe_body = 112.3 mm), the current best result, which uses the LLRD schedule (gamma=0.90, unfreeze_epoch=5). The head architecture is the only thing changed.

## Problem

The baseline `Pose3DHead` uses 4 transformer decoder layers. In DETR-family models, additional cross-attention layers allow the joint queries to iteratively refine their positions against the backbone feature map. The baseline has never been tested beyond 4 layers. There is headroom to add 2 more decoder layers cheaply since the head is only ~5.8M parameters — two extra layers add roughly 2.9M parameters, well within the VRAM budget.

## Proposed Change

Increase `head_num_layers` from 4 to 6. Everything else is identical to idea004/design002:
- `input_proj`: Linear(1024 → 256) — unchanged
- `joint_queries`: Embedding(70, 256) — unchanged
- Decoder hidden dim: 256, num_heads: 8 — unchanged
- No change to LLRD schedule or any other hyperparameter

**Rationale:** Six decoder layers allow three additional rounds of cross-attention and self-attention over the backbone features. In DETR the performance gap between 4 and 6 layers is consistently positive (≈1–2 MPJPE improvement) without overfitting within 20 epochs at this scale. The change is surgically minimal — one integer config change — so any improvement or degradation is directly attributable to decoder depth.

## Parameter Count Estimate

Each `TransformerDecoderLayer` (d_model=256, dim_feedforward=1024) adds approximately:
- Self-attention: 4 × 256² = 262 144 params
- Cross-attention: 4 × 256² = 262 144 params
- FFN: 2 × 256 × 1024 + biases ≈ 524 288 + 3 072 = 527 360 params
- Layer norms: 4 × 2 × 256 = 2 048 params

Total per layer ≈ 1.054M. Two extra layers ≈ +2.1M params. New head total ≈ 7.9M. Safe within VRAM.

## Configuration (`config.py` changes)

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design001"

# Model — only change
head_hidden     = 256
head_num_heads  = 8
head_num_layers = 6      # was 4 in baseline

# Schedule — identical to idea004/design002
lr_backbone     = 1e-4
lr_head         = 1e-4
gamma           = 0.90
unfreeze_epoch  = 5
warmup_epochs   = 3
epochs          = 20
weight_decay    = 0.03
grad_clip       = 1.0
lambda_depth    = 0.1
lambda_uv       = 0.2
head_dropout    = 0.1
drop_path       = 0.1
```

## Implementation Notes

- `model.py` `Pose3DHead.__init__` receives `num_layers=6` via the config; no code change needed.
- `train.py` LLRD optimizer logic is unchanged. The head param group still bundles all head parameters under a single lr_head group — adding 2 decoder layers means those extra layer params are automatically included.
- No change to `infra.py`, `transforms.py`, or backbone.

## Expected Outcome

Expecting val_mpjpe_body to improve modestly (1–3 mm) over the 112.3 mm baseline. If it degrades, that signals diminishing returns or mild overfitting from the added capacity, which would be informative for ruling out deeper decoders.
