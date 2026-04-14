# idea021 / design002 — Joint-Group Query Injection Before Refine Decoder (Axis A2)

## Starting Point

`runs/idea015/design004/code/`

## Problem

In idea015/design004, the queries fed into the refine decoder (`queries2 = out1 + R`) carry no explicit anatomical group structure. The refine decoder treats all 70 joint queries symmetrically in terms of their starting conditions, even though joints naturally cluster into groups with similar movement patterns (torso, arms, legs, extremities). Adding a learnable group embedding before the refine decoder pass can provide the decoder with explicit anatomical grouping context, biasing it to attend to structurally similar joints.

## Proposed Solution

Define 4 anatomical groups and assign each of the 70 joints to one group:
- **Group 0 — Torso**: pelvis, spine, neck, head (joints 0–3, approximately)
- **Group 1 — Arms**: shoulders, elbows, wrists (joints ~4–9)
- **Group 2 — Legs**: hips, knees, ankles (joints ~10–15)
- **Group 3 — Extremities**: fingers, toes, face joints (joints 16–69)

Create a learnable `nn.Embedding(4, hidden_dim)` zero-initialized. Before the refine decoder pass, add the group embedding to `queries2`:

```python
queries2 = queries2 + self.group_emb(self.joint_group_ids)  # (B, 70, hidden_dim)
```

Zero initialization means training starts identically to baseline.

## Group Assignment

Based on the SMPLX joint convention used in this codebase (indices 0-21 are body joints, 22+ are hand/face):

```python
# joint_group_ids: (70,) long tensor, registered as buffer
# Group 0: torso/spine — joints 0,1,2,3 (pelvis, L/R hip, spine1 or similar)
# Group 1: arms — joints ~4-9 (collar, shoulder, elbow, wrist)
# Group 2: legs — joints ~10-15 (hip, knee, ankle, foot)
# Group 3: extremities — joints 16-21 (hands/feet endpoints) + 22-69 (face, fingers)
```

The Builder should define `joint_group_ids` as a fixed buffer based on the actual SMPLX joint ordering used in `infra.py` (the `_ORIG_TO_NEW` remapping). A reasonable default assignment covering the standard SMPLX body joints (0-21):
- Joints 0-3: group 0 (pelvis, spine)
- Joints 4-9: group 1 (arms)
- Joints 10-15: group 2 (legs)
- Joints 16-21: group 0 (head/neck/face base — torso group)
- Joints 22-69: group 3 (extremities: hands, face, toes)

## Mathematical Formulation

```python
# In Pose3DHead.__init__():
self.group_emb = nn.Embedding(4, hidden_dim)
nn.init.zeros_(self.group_emb.weight)   # zero-init: no effect at training start

# joint_group_ids: (70,) int64 tensor (fixed, not learned)
joint_group_ids = torch.zeros(num_joints, dtype=torch.long)
joint_group_ids[4:10]  = 1  # arms
joint_group_ids[10:16] = 2  # legs
joint_group_ids[22:]   = 3  # extremities
self.register_buffer("joint_group_ids", joint_group_ids)

# In Pose3DHead.forward():
group_bias = self.group_emb(self.joint_group_ids)   # (70, hidden_dim)
queries2   = queries2 + group_bias.unsqueeze(0)     # (B, 70, hidden_dim)
out2       = self.refine_decoder(queries2, memory)
```

## Changes Required

**model.py**:
1. In `Pose3DHead.__init__()`:
   - Add: `self.group_emb = nn.Embedding(4, hidden_dim)` with `nn.init.zeros_(self.group_emb.weight)`
   - Add: register `joint_group_ids` buffer (70,) with group assignments above.
2. In `Pose3DHead.forward()`:
   - Before `out2 = self.refine_decoder(queries2, memory)`, insert:
     ```python
     group_bias = self.group_emb(self.joint_group_ids)
     queries2   = queries2 + group_bias.unsqueeze(0)
     ```

**train.py**: Unchanged.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea021/design002"

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

`group_emb`: `4 × 384 = 1,536` parameters. Zero-initialized, so no effect at training start.

`group_emb` belongs to `model.head.parameters()` automatically — goes into head optimizer group (LR=1e-4, WD=0.3).

## Expected Effect

The group embedding gives the refine decoder a structural prior about which joints belong to the same anatomical segment. Since the refine decoder is relatively small (2 layers) and already produces good predictions, this lightweight structural prior should help it produce more internally consistent refined predictions. This is a direct port of idea019/design004 (best weighted=102.22) to the stronger two-decoder baseline.

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4). 1,536 extra parameters are negligible.
