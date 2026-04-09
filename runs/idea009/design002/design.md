# Design 002 — Wide Head (hidden_dim=384)

## Starting Point

`runs/idea004/design002`

This design builds directly on idea004/design002 (val_mpjpe_body = 112.3 mm), the current best result, which uses the LLRD schedule (gamma=0.90, unfreeze_epoch=5). Only the head width is changed — the number of decoder layers, the LLRD schedule, and all other hyperparameters remain identical to the baseline.

## Problem

The baseline `Pose3DHead` uses `hidden_dim=256` for all internal representations: the input projection, joint query embeddings, and the transformer decoder's d_model. With a 293M-parameter backbone producing 1024-dim feature tokens, the 256-dim bottleneck may limit how much discriminative information the joint queries can extract per cross-attention step. Widening to 384 gives each query token 50% more capacity per dimension, potentially improving localization accuracy without adding decoder depth.

## Proposed Change

Increase `head_hidden` from 256 to 384. This affects three components inside `Pose3DHead`:

1. **`input_proj`**: `Linear(1024 → 384)` — maps backbone features into the wider head space.
2. **`joint_queries`**: `Embedding(70, 384)` — wider learnable query embeddings.
3. **Decoder d_model**: 384, `dim_feedforward = 384 × 4 = 1536` — wider FFN sublayers in all 4 decoder layers.
4. **Output heads**: `joints_out Linear(384 → 3)`, `depth_out Linear(384 → 1)`, `uv_out Linear(384 → 2)` — unchanged structure, updated in_features.

`head_num_layers` stays at 4 and `head_num_heads` stays at 8.

**Divisibility check:** 384 / 8 = 48 — valid; each attention head has 48-dim keys/queries/values.

**Rationale:** Widening the hidden dimension is a direct way to add representational capacity to the head without increasing depth or introducing new inductive biases. The backbone's 1024-dim features are projected to only 256 dims in the baseline, which is an aggressive compression. Relaxing that to 384 lets the joint queries retain more of the backbone signal through cross-attention. The change is fully captured in a single config field (`head_hidden`) and requires no structural code changes — `Pose3DHead` already parameterizes all layer sizes through `hidden_dim`.

## Parameter Count Estimate

Baseline head (hidden_dim=256, 4 layers):
- `input_proj`: 1024 × 256 + 256 = 262 400
- `joint_queries`: 70 × 256 = 17 920
- Per decoder layer (d_model=256, ffn=1024):
  - Self-attn QKV + out: 4 × 256² = 262 144
  - Cross-attn QKV + out: 4 × 256² = 262 144
  - FFN: 2 × 256 × 1024 + biases ≈ 524 544
  - LayerNorms: 4 × 2 × 256 = 2 048
  - Per-layer total ≈ 1.051M
- 4 layers ≈ 4.203M
- Output heads: (256×3 + 3) + (256×1 + 1) + (256×2 + 2) = 1 539
- **Baseline head total ≈ 5.48M params**

Wide head (hidden_dim=384, 4 layers):
- `input_proj`: 1024 × 384 + 384 = 393 600
- `joint_queries`: 70 × 384 = 26 880
- Per decoder layer (d_model=384, ffn=1536):
  - Self-attn QKV + out: 4 × 384² = 589 824
  - Cross-attn QKV + out: 4 × 384² = 589 824
  - FFN: 2 × 384 × 1536 + biases ≈ 1 179 648 + 3 456 ≈ 1 183 104
  - LayerNorms: 4 × 2 × 384 = 3 072
  - Per-layer total ≈ 2.366M
- 4 layers ≈ 9.462M
- Output heads: (384×3 + 3) + (384×1 + 1) + (384×2 + 2) = 2 307
- **Wide head total ≈ 9.88M params**

Delta: +4.4M parameters over baseline. Total model: ~303M params. Well within VRAM budget.

## Configuration (`config.py` changes)

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design002"

# Model — only change vs baseline
head_hidden     = 384      # was 256; widens input_proj, joint_queries, decoder d_model
head_num_heads  = 8        # unchanged; 384 / 8 = 48 — valid
head_num_layers = 4        # unchanged

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

- `Pose3DHead.__init__` already uses `hidden_dim` as the single width parameter for `input_proj`, `joint_queries`, the decoder `d_model`, and all output linears — no structural code change is needed. Setting `head_hidden=384` in `config.py` propagates automatically.
- `SapiensPose3D.__init__` passes `head_hidden` through to `Pose3DHead` — no change needed there either.
- `train.py` LLRD optimizer logic is unchanged. All head parameters are still collected under the single `lr_head` param group; the wider layers are automatically included.
- No changes to `infra.py`, `transforms.py`, or backbone.
- Builder should verify that `config.py` uses `head_hidden = 384` and that `output_dir` points to `runs/idea009/design002`.

## Expected Outcome

Expecting val_mpjpe_body to improve by 1–4 mm over the 112.3 mm baseline. Wider heads have shown consistent gains in transformer regression tasks when the backbone is significantly larger than the head (as is the case here: 293M backbone vs. ~5.5M head). If the result is comparable to or worse than Design 1 (6-layer decoder), it suggests that depth rather than width is the more useful capacity axis for this task.
