# design005 — Combined Anatomical Priors: Bone-Length Loss + Symmetry Loss + Kinematic Bias (Axis B3)

## Starting Point

`runs/idea015/design001/code/`

All hyperparameters from that run are preserved exactly: LLRD (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4), lr_head=1e-4, lr_depth_pe=1e-4, weight_decay=0.3, warmup_epochs=3, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, epochs=20, amp=False, batch_size=4, accum_steps=8, head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1, drop_path=0.1, num_depth_bins=16, sqrt-spaced continuous depth PE, two-pass shared-decoder refinement with query injection MLP.

## Problem

Designs 1-3 each test a single anatomical prior in isolation. It is plausible that bone-length consistency, left-right symmetry, and kinematic-chain attention are complementary: the loss-based priors (A1, B1) regularize the 3D output geometry, while the attention bias (A2) shapes how the decoder gathers information during refinement. Testing all three together in a single design identifies whether they are additive or conflicting, and provides the maximum-prior configuration for benchmarking.

## Proposed Solution

**All three anatomical priors combined: bone-length loss (lambda_bone=0.1) + symmetry loss (lambda_sym=0.05) + kinematic self-attention bias (learnable scalar, init=0.0) in the refinement decoder pass.**

### Architecture Changes (model.py — Pose3DHead)

Combine the changes from design001 (no model.py changes for bone/sym loss) and design002 (kin_bias buffer + kin_bias_scale scalar + manual pass-2 decoder loop):

```python
import collections

def _build_kin_bias(num_joints, smplx_skeleton, body_n=22,
                    hop_weights=(1.0, 0.5, 0.25), max_hops=3):
    adj = collections.defaultdict(set)
    for (a, b) in smplx_skeleton:
        if a < body_n and b < body_n:
            adj[a].add(b)
            adj[b].add(a)
    bias = torch.zeros(num_joints, num_joints)
    import collections as _col
    for src in range(body_n):
        dist = {src: 0}
        queue = _col.deque([src])
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

# In __init__ (after decoder):
from infra import SMPLX_SKELETON
kin_bias = _build_kin_bias(num_joints, SMPLX_SKELETON)
self.register_buffer("kin_bias", kin_bias)       # (70, 70)
self.kin_bias_scale = nn.Parameter(torch.zeros(1))  # learnable scalar, init=0.0
```

```python
# In forward():

# Pass 1 (unchanged)
out1 = self.decoder(queries, memory)
J1   = self.joints_out(out1)

# Refinement injection
R        = self.refine_mlp(J1)
queries2 = out1 + R

# Pass 2 with kinematic self-attention bias
bias = self.kin_bias_scale * self.kin_bias   # (70, 70)
out2 = queries2
for layer in self.decoder.layers:
    out2 = layer(out2, memory, tgt_mask=bias)
if self.decoder.norm is not None:
    out2 = self.decoder.norm(out2)
J2 = self.joints_out2(out2)

pelvis_token = out2[:, 0, :]
return {
    "joints":        J2,
    "joints_coarse": J1,
    "pelvis_depth":  self.depth_out(pelvis_token),
    "pelvis_uv":     self.uv_out(pelvis_token),
}
```

### Loss Composition (train.py)

Combines bone-length loss (design001) + symmetry loss (design003):

```python
from infra import SMPLX_SKELETON, BODY_IDX

BODY_EDGES = [(a, b) for (a, b) in SMPLX_SKELETON if a < 22 and b < 22]

SYM_PAIRS = [
    (13, 16, 14, 17),   # upper arm
    (16, 18, 17, 19),   # forearm
    (18, 20, 19, 21),   # hand root
    (1,  4,  2,  5),    # thigh
    (4,  7,  5,  8),    # shin
    (7,  10, 8,  11),   # foot
]

def bone_length_loss(pred_joints, gt_joints, edges):
    total = 0.0
    for (a, b) in edges:
        pred_len = (pred_joints[:, a, :] - pred_joints[:, b, :]).norm(dim=-1)
        gt_len   = (gt_joints[:, a, :]   - gt_joints[:, b, :]  ).norm(dim=-1)
        total += (pred_len - gt_len).abs().mean()
    return total / max(len(edges), 1)

def symmetry_loss(pred_joints, sym_pairs):
    total = 0.0
    for (la, lb, ra, rb) in sym_pairs:
        left_len  = (pred_joints[:, la, :] - pred_joints[:, lb, :]).norm(dim=-1)
        right_len = (pred_joints[:, ra, :] - pred_joints[:, rb, :]).norm(dim=-1)
        total += (left_len - right_len).abs().mean()
    return total / max(len(sym_pairs), 1)

# In train_one_epoch:
l_pose1   = pose_loss(out["joints_coarse"][:, BODY_IDX], joints[:, BODY_IDX])
l_pose2   = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
l_bone    = bone_length_loss(out["joints"], joints, BODY_EDGES)
l_sym     = symmetry_loss(out["joints"], SYM_PAIRS)
l_pose    = 0.5 * l_pose1 + 1.0 * l_pose2 + args.lambda_bone * l_bone + args.lambda_sym * l_sym
l_dep     = pose_loss(out["pelvis_depth"], gt_pd)
l_uv      = pose_loss(out["pelvis_uv"],    gt_uv)
loss      = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps
```

### Optimizer Group

`kin_bias_scale` is a parameter on `model.head`, automatically included in the `head_params` optimizer group (LR=1e-4, weight_decay=0.3).

### config.py Changes

```python
lambda_bone       = 0.1    # bone-length auxiliary loss weight
lambda_sym        = 0.05   # symmetry auxiliary loss weight
kin_bias_max_hops = 3      # max BFS hops for kinematic attention bias
kin_bias_scale_init = 0.0  # learnable bias scalar init
```

## Memory Estimate

- `kin_bias` buffer: 70×70 floats (~20 KB).
- `kin_bias_scale`: 1 parameter.
- Total new learnable params: 1. No OOM risk.

## Summary

| Field | Value |
|---|---|
| Starting point | runs/idea015/design001/code/ |
| model.py changes | kin_bias buffer (70,70) + kin_bias_scale (1 param, init=0) + manual pass-2 decoder loop |
| New losses | bone_length_loss(J2, lambda_bone=0.1) + symmetry_loss(J2, lambda_sym=0.05) |
| Loss formula | 0.5*L(J1) + 1.0*L(J2) + 0.1*bone_loss(J2) + 0.05*sym_loss(J2) + lambda_depth*L_dep + lambda_uv*L_uv |
| Kinematic bias | Additive self-attn bias in pass 2, hop-1:+1.0, hop-2:+0.5, hop-3:+0.25 |
| Extra params | 1 scalar |
| All other HPs | Identical to idea015/design001 |
