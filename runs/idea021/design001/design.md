# idea021 / design001 — Kinematic Soft-Attention Bias in Refine Decoder Only (Axis A1)

## Starting Point

`runs/idea015/design004/code/`

## Problem

The independent 2-layer refine decoder in idea015/design004 treats all 70 joint queries as independent entities during attention. Anatomically, joints connected by a bone are structurally correlated — the wrist depends on the elbow, which depends on the shoulder. Injecting this structure as a soft additive bias in the refine decoder's self-attention should make the decoder more aware of kinematic dependencies when producing refined predictions.

## Proposed Solution

Precompute a `(70, 70)` hop-distance matrix from `SMPLX_SKELETON` using BFS and convert to an additive attention bias:
- 1-hop neighbors: +1.0
- 2-hop neighbors: +0.5
- 3-hop neighbors: +0.25
- Beyond 3 hops: 0.0

Scale by a single learnable scalar `kin_bias_scale` initialized to 0.0, so at training start the bias is zero (identical to baseline). Pass the scaled bias as `tgt_mask` to **only** `self.refine_decoder`. The coarse decoder retains `tgt_mask=None`.

## Mathematical Formulation

```python
# Pre-training setup (in __init__ or as a buffer):
kin_bias = compute_kin_bias(SMPLX_SKELETON)  # (70, 70) float buffer

# In Pose3DHead.__init__():
self.register_buffer("kin_bias", kin_bias)
self.kin_bias_scale = nn.Parameter(torch.zeros(1))

# In Pose3DHead.forward():
bias_matrix = self.kin_bias_scale * self.kin_bias  # (70, 70)
# Pass to refine_decoder only:
out2 = self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)
# Coarse decoder unchanged:
out1 = self.decoder(queries, memory)  # tgt_mask=None
```

Note: `nn.TransformerDecoder` accepts `tgt_mask` and passes it to all layers as additive self-attention bias (PyTorch convention). Since `batch_first=True`, the mask is `(tgt_len, tgt_len)` = `(70, 70)`. The value is added directly to attention logits before softmax (additive mask semantics in PyTorch when the mask is a float tensor, not a boolean tensor). Since all values are non-negative or zero (no -inf), no attention positions are fully blocked.

## Helper Function (add to model.py)

```python
def _compute_kin_bias(smplx_skeleton, num_joints: int = 70) -> torch.Tensor:
    """BFS hop-distance → additive attention bias matrix."""
    import collections
    adj = [[] for _ in range(num_joints)]
    for a, b in smplx_skeleton:
        if a < num_joints and b < num_joints:
            adj[a].append(b)
            adj[b].append(a)
    
    bias = torch.zeros(num_joints, num_joints)
    hop_weights = {1: 1.0, 2: 0.5, 3: 0.25}
    
    for src in range(num_joints):
        visited = {src: 0}
        queue = collections.deque([(src, 0)])
        while queue:
            node, hop = queue.popleft()
            if hop >= 3:
                continue
            for nbr in adj[node]:
                if nbr not in visited:
                    visited[nbr] = hop + 1
                    queue.append((nbr, hop + 1))
        for dst, hop in visited.items():
            if hop > 0 and hop <= 3:
                bias[src, dst] = hop_weights[hop]
    return bias
```

## Changes Required

**model.py**:
1. Import `SMPLX_SKELETON` from `infra` (add to existing infra imports).
2. Add `_compute_kin_bias()` helper function.
3. In `Pose3DHead.__init__()`:
   - Add: `self.register_buffer("kin_bias", _compute_kin_bias(SMPLX_SKELETON, num_joints))`
   - Add: `self.kin_bias_scale = nn.Parameter(torch.zeros(1))`
4. In `Pose3DHead.forward()`:
   - Compute bias: `bias_matrix = self.kin_bias_scale * self.kin_bias`
   - Pass to refine decoder: `out2 = self.refine_decoder(queries2, memory, tgt_mask=bias_matrix)`
   - Coarse decoder unchanged: `out1 = self.decoder(queries, memory)`

**train.py**: Unchanged.

## Configuration (config.py fields)

All values identical to `runs/idea015/design004/code/config.py` except `output_dir`:

```python
output_dir  = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea021/design001"

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

1 scalar (`kin_bias_scale`) + 1 buffer (`kin_bias`, shape `(70,70)`, not a parameter). Total trainable: 1 new parameter.

`kin_bias_scale` belongs to `model.head.parameters()` — it goes into the head optimizer group (LR=1e-4, WD=0.3) automatically.

## Expected Effect

The kinematic bias in the refine decoder should make nearby joints' self-attention more prominent, resulting in more anatomically coherent refined predictions. The zero initialization of `kin_bias_scale` ensures exact baseline behavior at training start. This is a direct port of the best idea019 result (design002, body=105.77) to the stronger two-decoder baseline (body=102.51).

## Memory Estimate

Identical to `runs/idea015/design004` (~11 GB at batch=4). Buffer and one scalar are negligible.
