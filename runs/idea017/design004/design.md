# design004 — Three-Frame Symmetric Temporal Fusion (t-5, t, t+5; past/future frozen, centre trainable)

## Starting Point

`runs/idea014/design003/code/`

## Summary

The most expressive temporal design. The dataloader yields three frames: `t-5`, `t`, and `t+5` (clamped at sequence boundaries). The past-frame and future-frame backbone passes run inside `torch.no_grad()` (frozen context); only the centre-frame pass participates in gradient updates and LLRD. The three backbone outputs are projected through the shared `input_proj` and concatenated along the spatial token axis: `memory = cat([proj(feat_prev), proj(feat_t), proj(feat_next)], dim=1)` — shape `(B, 2880, 384)`. This symmetric window is known to be strictly stronger than an asymmetric causal window for mid-sequence frames (cf. VideoPose3D ablations). Only centre-frame joints are supervised.

## Problem

Designs 001-003 use only a past frame (asymmetric window). For sequences shot at 25 fps with FRAME_STRIDE=5, the "future" frame at `t+5` (200 ms ahead) carries near-identical scene context and provides the symmetric temporal prior that reduces velocity/acceleration ambiguity. The symmetric window was the key factor in VideoPose3D's improvement over causal variants.

## Proposed Solution

### Dataloader: three-frame fetch

```python
past_idx   = max(0, frame_idx - 1)          # t-5 in dataset-index space
future_idx = min(n_frames - 1, frame_idx + 1)  # t+5 in dataset-index space
```

Load `rgb_prev`, `depth_prev`, `rgb_next`, `depth_next` using the **same crop bbox** as the centre frame. Expose all four as tensors in the sample dict after applying the same manual cropping+normalization as in designs 002-003.

### Model forward pass

```python
def forward(self, x_t, x_prev=None, x_next=None):
    # Centre-frame: full gradient (LLRD)
    feat_t = self.backbone(x_t)

    feats_context = []
    if x_prev is not None:
        with torch.no_grad():
            feat_prev = self.backbone(x_prev).detach()
        feats_context.append(feat_prev)

    if x_next is not None:
        with torch.no_grad():
            feat_next = self.backbone(x_next).detach()
        feats_context.append(feat_next)

    return self.head(feat_t, feats_context)
```

Explicit `.detach()` after `no_grad()` blocks ensures no gradient leakage. Intermediate activations from the frozen passes are freed immediately after detach (no references are kept).

### Head: concatenated three-frame memory

```python
def forward(self, feat_t, feats_context=None):
    B = feat_t.size(0)
    mem_t = self.input_proj(feat_t.flatten(2).transpose(1, 2))   # (B, 960, 384)

    mems = [mem_t]
    if feats_context:
        for f in feats_context:
            mems.append(self.input_proj(f.flatten(2).transpose(1, 2)))  # (B, 960, 384) each
    # Ordering: [prev, t, next] for symmetry — context frames bracket the centre
    if len(mems) == 3:
        memory = torch.cat([mems[1], mems[0], mems[2]], dim=1)  # [prev_mem, t_mem, next_mem]
    elif len(mems) == 2:
        memory = torch.cat(mems, dim=1)
    else:
        memory = mems[0]

    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
    out = self.decoder(queries, memory)
    pelvis_token = out[:, 0, :]
    return {
        "joints":       self.joints_out(out),
        "pelvis_depth": self.depth_out(pelvis_token),
        "pelvis_uv":    self.uv_out(pelvis_token),
    }
```

Memory ordering `[prev, t, next]` places the centre frame in the middle of the key-value sequence, which makes it trivially accessible to the cross-attention's positional proximity bias.

### LLRD / optimizer

Identical to idea014/design003. Only the centre-frame backbone path (single module) participates in LLRD. No optimizer changes.

### Training loop

```python
rgb_prev  = batch["rgb_prev"].to(device, non_blocking=True)
depth_prev = batch["depth_prev"].to(device, non_blocking=True)
rgb_next  = batch["rgb_next"].to(device, non_blocking=True)
depth_next = batch["depth_next"].to(device, non_blocking=True)

x_t    = torch.cat([rgb,      depth],      dim=1)
x_prev = torch.cat([rgb_prev, depth_prev], dim=1)
x_next = torch.cat([rgb_next, depth_next], dim=1)

with torch.amp.autocast("cuda", enabled=args.amp):
    out    = model(x_t, x_prev, x_next)
    l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
    l_dep  = pose_loss(out["pelvis_depth"], gt_pd)
    l_uv   = pose_loss(out["pelvis_uv"],    gt_uv)
    loss   = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

During validation: `model(x_t, None, None)` (single-frame fallback).

### Sequence boundary handling

- `past_frame_idx = max(0, frame_idx - 1)`: at the first frame of a sequence, past = current (delta = 0, no temporal info, but the model should still function).
- `future_frame_idx = min(n_frames - 1, frame_idx + 1)`: at the last frame, future = current.
- Clamping is done in dataset-index space (each index step = FRAME_STRIDE=5 raw frames).

## config.py Fields

```python
# Temporal context
temporal_mode     = "three_frame_symmetric"  # informational

# Everything else UNCHANGED from idea014/design003:
in_channels      = 4
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

One trainable backbone pass + two `no_grad` backbone passes (activations freed after `.detach()`). Activation memory footprint from the frozen passes is transient. Peak memory is dominated by the single trainable pass and the doubled decoder cross-attention (memory length 2880 vs. 960). Estimated: ~8-9 GB at batch=4, accum=8. Should fit within the 11 GB budget. Builder should verify with a 1-step dry run. If OOM, reduce batch_size to 2 and increase accum_steps to 16.

## Expected Outcome

The most expressive design — the symmetric window should provide the largest pelvis depth reduction. Target: body MPJPE -4 to -10 mm, pelvis -5 to -10 mm vs. baseline. The future-frame context is particularly valuable for pelvis depth: given the camera-space motion trajectory `(t-5, t, t+5)`, the depth of pelvis at `t` is constrained by smooth motion from both sides.
