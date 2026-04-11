# design003 — Cross-Frame Memory Attention (2-frame, past frozen no_grad, centre trainable)

## Starting Point

`runs/idea014/design003/code/`

## Summary

Same two-frame cross-attention architecture as design002, but the **past-frame (t-5) backbone forward runs inside `torch.no_grad()` and is not gradient-checkpointed**. Only the centre-frame backbone is trained (LLRD applies to it only). This halves gradient memory vs. design002 by eliminating the backward pass through the past-frame path, while still supplying rich temporal features to the decoder cross-attention. The hypothesis is that frozen temporal context (i.e., a "read-only memory" from the past frame) is sufficient and may even act as regularization.

## Problem

Design002 trains both backbone passes, which risks OOM on the 11 GB M40. Freezing the past-frame pass reduces gradient memory by ~40% while retaining the same decoder expressiveness. Many temporal models in the literature (e.g., semi-online architectures) show that a frozen "context encoder" can still provide useful temporal cues.

## Proposed Solution

### Dataloader

Identical to design002: fetch `past_frame_idx = max(0, frame_idx - 1)`, load `rgb_prev` + `depth_prev` using the same crop bbox, expose as tensors in `sample["rgb_prev"]` and `sample["depth_prev"]`.

### Model forward pass

```python
def forward(self, x_t, x_prev=None):
    # Centre-frame: full gradient (with LLRD)
    feat_t = self.backbone(x_t)           # (B, embed_dim, H_tok, W_tok)

    if x_prev is not None:
        # Past-frame: frozen inference, no gradient
        with torch.no_grad():
            feat_prev = self.backbone(x_prev)
        feat_prev = feat_prev.detach()    # explicitly detach to be safe
    else:
        feat_prev = None

    return self.head(feat_t, feat_prev)
```

No `torch.utils.checkpoint` on either path: the past-frame pass has no gradient, so no activation memory is retained for it. The centre-frame pass runs normally (no checkpointing needed since only one trainable pass).

### Head: concatenated memory

Identical to design002:

```python
def forward(self, feat_t, feat_prev=None):
    B = feat_t.size(0)
    mem_t = self.input_proj(feat_t.flatten(2).transpose(1, 2))    # (B, 960, 384)
    if feat_prev is not None:
        mem_prev = self.input_proj(feat_prev.flatten(2).transpose(1, 2))  # (B, 960, 384)
        memory = torch.cat([mem_prev, mem_t], dim=1)                # (B, 1920, 384)
    else:
        memory = mem_t
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
    out = self.decoder(queries, memory)
    pelvis_token = out[:, 0, :]
    return {
        "joints":       self.joints_out(out),
        "pelvis_depth": self.depth_out(pelvis_token),
        "pelvis_uv":    self.uv_out(pelvis_token),
    }
```

### LLRD / optimizer

LLRD applies only to the single shared backbone (centre-frame path). The optimizer structure is identical to idea014/design003 — no changes to `_build_optimizer_frozen()` or `_build_optimizer_full()`.

### Training loop

```python
rgb_prev  = batch["rgb_prev"].to(device, non_blocking=True)
depth_prev = batch["depth_prev"].to(device, non_blocking=True)
x_t    = torch.cat([rgb, depth], dim=1)
x_prev = torch.cat([rgb_prev, depth_prev], dim=1)

with torch.amp.autocast("cuda", enabled=args.amp):
    out    = model(x_t, x_prev)
    l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
    l_dep  = pose_loss(out["pelvis_depth"], gt_pd)
    l_uv   = pose_loss(out["pelvis_uv"],    gt_uv)
    loss   = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

During validation: `model(x_t, None)` (single-frame fallback in the head).

## config.py Fields

```python
# Temporal context
temporal_mode     = "cross_attn_past_frozen"  # informational

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

One trainable backbone pass (no checkpointing) + one `no_grad` past-frame pass (activations immediately freed after `.detach()`). Estimated peak memory: ~7-8 GB at batch=4, accum=8. Comfortably within the 11 GB budget.

## Expected Outcome

Similar or slightly lower improvement vs. design002 (since past-frame features are frozen). The advantage is that training is more stable (no gradient explosion through two coupled passes) and memory is comfortable. Target: body MPJPE -3 to -6 mm, pelvis -3 to -6 mm vs. baseline.
