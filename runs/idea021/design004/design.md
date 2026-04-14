# idea021 / design004 — Kinematic Bias + Joint-Group Injection Combined (Axis B2)

## Starting Point

`runs/idea015/design004/code/`

## Problem

idea019/design005 (combined bone+sym+kinematic) was worse than individual priors — suggesting over-regularization. However, that design combined three distinct priors including symmetry loss, which failed badly (idea019/design003). The two best individual priors from idea019 were kinematic soft-attention bias (design002, body=105.77) and joint-group query injection (design004, weighted=102.22). These two were never tested together as a pair — design005 in idea019 added symmetry loss on top, which was the harmful component.

This design tests the pair (kinematic bias + joint-group injection) on the stronger two-decoder baseline, deliberately omitting symmetry loss and bone loss.

## Proposed Solution

Combine design001 (Axis A1) and design002 (Axis A2) of this idea:
1. **Kinematic soft-attention bias** (`kin_bias_scale * kin_bias`) passed as `tgt_mask` to `self.refine_decoder` only.
2. **Joint-group query injection** (`group_emb(joint_group_ids)`) added to `queries2` before the refine decoder.

Both modifications target the same location (the refine decoder pass) and should be complementary: the group embedding provides a query-level anatomical grouping prior, while the kinematic bias provides an attention-level structural connectivity prior.

## Mathematical Formulation

```python
# In Pose3DHead.forward():

# Pass 1: coarse (unchanged)
out1 = self.decoder(queries, memory)  # tgt_mask=None
J1   = self.joints_out(out1)

# Refinement query construction (unchanged)
R        = self.refine_mlp(J1)
queries2 = out1 + R

# [NEW A2] Add group embedding to queries2
group_bias = self.group_emb(self.joint_group_ids)  # (70, hidden_dim)
queries2   = queries2 + group_bias.unsqueeze(0)    # (B, 70, hidden_dim)

# [NEW A1] Kinematic bias matrix
bias_matrix = self.kin_bias_scale * self.kin_bias  # (70, 70)

# Pass 2: refine with both priors
out2 = self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)
J2   = self.joints_out2(out2)
```

## Changes Required

**model.py** — combine changes from design001 and design002:

1. Import `SMPLX_SKELETON` from `infra`.
2. Add `_compute_kin_bias()` helper function (same as design001).
3. In `Pose3DHead.__init__()`:
   - Add: `self.register_buffer("kin_bias", _compute_kin_bias(SMPLX_SKELETON, num_joints))`
   - Add: `self.kin_bias_scale = nn.Parameter(torch.zeros(1))`
   - Add: `self.group_emb = nn.Embedding(4, hidden_dim)` with `nn.init.zeros_(self.group_emb.weight)`
   - Add: register `joint_group_ids` buffer (70,) with group assignments:
     ```python
     joint_group_ids = torch.zeros(num_joints, dtype=torch.long)
     joint_group_ids[4:10]  = 1  # arms
     joint_group_ids[10:16] = 2  # legs
     joint_group_ids[22:]   = 3  # extremities
     self.register_buffer("joint_group_ids", joint_group_ids)
     ```
4. In `Pose3DHead.forward()`:
   - After `queries2 = out1 + R`, insert:
     ```python
     group_bias = self.group_emb(self.joint_group_ids)
     queries2   = queries2 + group_bias.unsqueeze(0)
     bias_matrix = self.kin_bias_scale * self.kin_bias
     ```
   - Change refine decoder call to: `out2 = self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)`

**train.py**: Unchanged.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea021/design004"

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

## New Parameters

- `kin_bias_scale`: 1 scalar parameter (zero-init)
- `group_emb`: 4 × 384 = 1,536 parameters (zero-init)
- `kin_bias` buffer: (70, 70) floats — not a parameter

Total new trainable: **1,537 parameters**, all zero-initialized.

Both belong to `model.head.parameters()` automatically → head optimizer group (LR=1e-4, WD=0.3).

## Expected Effect

The joint-group embedding and kinematic attention bias are complementary anatomical priors targeting different aspects of the refine decoder:
- Group embedding: gives each query a structural identity prior before decoding.
- Kinematic bias: biases self-attention toward anatomically connected joints during decoding.

Both are zero-initialized (exact baseline at training start). The combination was never tested in idea019 in isolation from symmetry loss — this is a novel pair experiment. If the two priors are additive, this should be the best single-model result among the idea021 designs.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4). ~1,537 extra parameters and one (70,70) buffer are negligible.
