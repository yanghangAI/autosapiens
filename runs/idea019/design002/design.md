# design002 — Kinematic-Chain Soft Self-Attention Bias in Refinement Pass (Axis A2)

## Starting Point

`runs/idea015/design001/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE, two-pass shared-decoder refinement with query injection MLP.

## Problem

The current SOTA (idea015/design001) treats all 70 joint queries as unstructured tokens in both decoder passes. During the refinement pass (pass 2), the model must re-discover kinematic relationships from scratch using only learned weights, despite having coarse predictions available. Injecting a soft additive self-attention bias based on kinematic distance — specifically in the second pass only — should help adjacent joints (e.g., elbow + wrist) share information preferentially during refinement, reducing localization error on distal joints without using hard masks (which failed in idea002).

## Proposed Solution

**Soft additive kinematic self-attention bias in the second decoder pass only, with a learnable scalar magnitude initialized to 0.0.**

The approach:
1. Precompute a fixed `(70, 70)` kinematic distance matrix at init using BFS over `SMPLX_SKELETON`. Body joints (0-21) get distance-based bias values; all non-body queries (indices 22-69) get 0 bias for both their row and column.
2. Convert distances to bias values: hop-1 → +1.0, hop-2 → +0.5, hop-3 → +0.25, beyond 3 hops or non-body → 0.0.
3. Register this as a buffer `kin_bias (70, 70)`.
4. A learnable scalar `kin_bias_scale` (init=0.0) scales the entire bias, so training starts identical to the baseline.
5. During the second decoder pass, manually loop over the `TransformerDecoder` layers and inject the additive bias into the self-attention as `attn_mask`. Because `nn.MultiheadAttention` sums `attn_mask` into the logits before softmax, passing `kin_bias_scale * kin_bias` (shape `(70, 70)`, broadcast over batch and heads) applies the soft bias.

**Critical invariant:** No row of `kin_bias` is all-finite-negative, so no joint is fully cut off from attending to all others. The bias only boosts nearby pairs, it never sets anything to -inf.

### Architecture Changes (model.py — Pose3DHead)

```python
import torch
import torch.nn as nn
import collections

# In Pose3DHead.__init__, after building the decoder:

# Precompute kinematic hop distance matrix
def _build_kin_bias(num_joints, smplx_skeleton, body_n=22,
                    hop_weights=(1.0, 0.5, 0.25), max_hops=3):
    """BFS from each body joint to compute hop distances; returns (num_joints, num_joints) bias tensor."""
    # Build adjacency list for body joints only
    adj = collections.defaultdict(set)
    for (a, b) in smplx_skeleton:
        if a < body_n and b < body_n:
            adj[a].add(b)
            adj[b].add(a)
    bias = torch.zeros(num_joints, num_joints)
    for src in range(body_n):
        dist = {src: 0}
        queue = collections.deque([src])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= max_hops:
                continue
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = d + 1
                    queue.append(nb)
        for tgt, d in dist.items():
            if d > 0 and d <= max_hops:
                bias[src, tgt] = hop_weights[d - 1]
    return bias

# In __init__:
from infra import SMPLX_SKELETON
kin_bias = _build_kin_bias(num_joints, SMPLX_SKELETON)
self.register_buffer("kin_bias", kin_bias)  # (70, 70) fixed buffer
self.kin_bias_scale = nn.Parameter(torch.zeros(1))  # learnable scalar, init=0.0
```

```python
# In forward(), replace the second decoder pass:

# Pass 2: refined prediction (shared decoder weights) with kinematic bias on self-attention
bias = self.kin_bias_scale * self.kin_bias  # (70, 70)
out2 = queries2  # (B, 70, hidden_dim)
for layer in self.decoder.layers:
    # layer is a TransformerDecoderLayer (norm_first=True, batch_first=True)
    # self-attention: tgt attends tgt with attn_mask=bias
    out2 = layer(out2, memory, tgt_mask=bias)
if self.decoder.norm is not None:
    out2 = self.decoder.norm(out2)
J2 = self.joints_out2(out2)  # (B, 70, 3)
```

Note: `nn.TransformerDecoderLayer.forward` accepts `tgt_mask` as an additive float bias (shape broadcastable to `(B*heads, 70, 70)` or `(70, 70)`) when the attention module's `is_causal=False`. This is standard PyTorch behavior.

### Loss (train.py)

Unchanged from idea015/design001:
```python
l_pose1 = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2 = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose  = 0.5 * l_pose1 + 1.0 * l_pose2
l_dep   = pose_loss(out["pelvis_depth"], gt_pd)
l_uv    = pose_loss(out["pelvis_uv"],    gt_uv)
loss    = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

### Optimizer Group

`kin_bias_scale` is a parameter on `model.head`, so it is automatically included in the `head_params` optimizer group (LR=1e-4, weight_decay=0.3). No changes to train.py optimizer setup.

### config.py Changes

Add the following informational fields:
```python
kin_bias_max_hops = 3        # maximum BFS hops for kinematic bias
kin_bias_scale_init = 0.0    # init value for learnable bias scalar
```

## Memory Estimate

- `kin_bias` buffer: 70×70 floats = ~20 KB (negligible).
- `kin_bias_scale`: 1 scalar parameter.
- Total new learnable params: 1. No OOM risk.

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea015/design001/code/ |
| New module | kin_bias (70,70) buffer + kin_bias_scale (1 param, init=0) |
| Bias application | Additive to self-attention tgt_mask in decoder pass 2 only |
| Bias values | hop-1: +1.0, hop-2: +0.5, hop-3: +0.25, else: 0.0 |
| Non-body queries | 0 bias (rows 22-69 remain zero) |
| Loss formula | 0.5*L(J1) + 1.0*L(J2) (unchanged) |
| Extra params | 1 scalar |
| All other HPs | Identical to idea015/design001 |
