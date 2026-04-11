# design002 — Cross-Frame Memory Attention (2-frame, both trainable, gradient checkpointing)

## Starting Point

`runs/idea014/design003/code/`

## Summary

True temporal cross-attention fusion. The dataloader yields two frames: the past frame at `t-5` and the centre frame at `t`. The backbone runs once per frame inside the training step, with `torch.utils.checkpoint.checkpoint` applied to BOTH backbone forwards to halve gradient memory. The two backbone outputs are projected through `input_proj` and concatenated along the spatial token axis: `memory = cat([proj(feat_prev), proj(feat_t)], dim=1)` — shape `(B, 1920, 384)`. The decoder cross-attends to this doubled memory. Only centre-frame joints are supervised. LLRD is applied identically to both backbone instances (they share weights).

## Problem

Channel stacking (design001) limits temporal expressiveness because the backbone must discover temporal structure from raw pixel channels. Cross-attention in the decoder head allows the model to explicitly query both time steps for each joint, learning which spatial locations in each frame are diagnostic. This is analogous to how VideoPose3D and PoseFormer exploit temporal context.

## Proposed Solution

### Two-frame dataloader

Same as design001: fetch `past_frame_idx = max(0, frame_idx - 1)`, load `rgb_prev` and `depth_prev`, apply the **same crop bbox**. Expose as `sample["rgb_prev"]` and `sample["depth_prev"]` (already tensor-ised by an augmented `ToTensor`).

### Shared-weight backbone with gradient checkpointing

The model has a **single** `SapiensBackboneRGBD` instance (in_channels=4, as in the baseline). In `forward`, it is called twice:

```python
import torch.utils.checkpoint as ckpt

def forward(self, x_t, x_prev):
    # Both calls are gradient-checkpointed to save activation memory
    feat_t    = ckpt.checkpoint(self.backbone, x_t,    use_reentrant=False)
    feat_prev = ckpt.checkpoint(self.backbone, x_prev, use_reentrant=False)
    return self.head(feat_t, feat_prev)
```

`SapiensBackboneRGBD._run_vit_manual` is already amenable to checkpointing because it takes a single input tensor and returns a single output tensor.

### Head: concatenated memory

In `Pose3DHead.forward`, accept an optional second feature map:

```python
def forward(self, feat_t, feat_prev=None):
    B = feat_t.size(0)
    mem_t = self.input_proj(feat_t.flatten(2).transpose(1, 2))   # (B, 960, 384)
    if feat_prev is not None:
        mem_prev = self.input_proj(feat_prev.flatten(2).transpose(1, 2))  # (B, 960, 384)
        memory = torch.cat([mem_prev, mem_t], dim=1)               # (B, 1920, 384)
    else:
        memory = mem_t                                              # single-frame fallback
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
    out = self.decoder(queries, memory)
    ...
```

The `TransformerDecoder` handles variable-length memory natively — no architectural change to the decoder itself.

Note: `input_proj` is shared for both frames (weight tying). This is intentional: the same linear projection maps both time steps into the 384-d space.

### LLRD / optimizer

The backbone is a single module — LLRD groups are built identically to idea014/design003. The two forward passes share all parameters, so gradient accumulation is correct. No changes to `_build_optimizer_frozen()` or `_build_optimizer_full()`.

### Training loop change

```python
rgb_prev  = batch["rgb_prev"].to(device, non_blocking=True)
depth_prev = batch["depth_prev"].to(device, non_blocking=True)
x_t    = torch.cat([rgb, depth], dim=1)       # (B, 4, H, W)
x_prev = torch.cat([rgb_prev, depth_prev], dim=1)  # (B, 4, H, W)

out = model(x_t, x_prev)
# loss identical to baseline (centre-frame targets only)
```

During validation (`infra.validate`), the model is called in single-frame mode: `model(x_t, None)` or equivalently the model can wrap the head to accept only `feat_t` when `feat_prev` is None. This keeps validation comparable to the single-frame baseline.

**Important:** If OOM occurs during the dry run (first 10 steps), the Orchestrator should mark this design `Infeasible`.

## config.py Fields

```python
# Temporal context
temporal_mode     = "cross_attn_both_trainable"  # informational
use_grad_ckpt     = True  # apply torch.utils.checkpoint to both backbone forwards

# Everything else UNCHANGED from idea014/design003:
in_channels      = 4      # single backbone, 4-channel (centre frame); prev frame fed separately
arch             = "sapiens_0.3b"
head_hidden      = 384
head_num_heads   = 8
head_num_layers  = 4
head_dropout     = 0.1
drop_path        = 0.1
epochs           = 20
batch_size       = 4
lr_backbone      = 1e-4
base_lr_backbone = 1e-4
llrd_gamma       = 0.90
unfreeze_epoch   = 5
lr_head          = 1e-4
lr_depth_pe      = 1e-4
weight_decay     = 0.3
warmup_epochs    = 3
grad_clip        = 1.0
accum_steps      = 8
amp              = False
lambda_depth     = 0.1
lambda_uv        = 0.2
num_depth_bins   = 16
```

## Memory Estimate

Two gradient-checkpointed backbone passes: activation memory is approximately halved per pass compared to a normal forward. Estimated peak memory: ~9-10 GB at batch=4, accum=8. The Builder MUST verify with a 1-step dry run (`max_batches=1`) before full training. If peak exceeds 10.5 GB, reduce `batch_size` to 2 and increase `accum_steps` to 16 to preserve effective batch size.

## Expected Outcome

Moderate-to-large improvement. True cross-attention over both time steps should substantially reduce depth ambiguity. The decoder can learn to attend to optical-flow-like spatial correspondences between `t-5` and `t`. Target: body MPJPE -3 to -8 mm, pelvis -3 to -8 mm vs. baseline.
