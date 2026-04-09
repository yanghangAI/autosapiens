# Design 004 — Per-Layer Input Feature Gate

## Starting Point

`runs/idea004/design002`

This design builds directly on idea004/design002 (val_mpjpe_body = 112.3 mm), the current best result, which uses the LLRD schedule (gamma=0.90, unfreeze_epoch=5). The head architecture (4 layers, hidden_dim=256), the LLRD schedule, and all other hyperparameters remain identical to the baseline. Only the gating mechanism on the cross-attention memory input is added.

## Problem

The baseline `Pose3DHead` computes a single projected memory tensor:

```
memory = input_proj(backbone_features)   # (B, H*W, 256)
```

and feeds it identically into all 4 decoder layers. Each `TransformerDecoderLayer` performs cross-attention between the evolving joint queries and this same frozen `memory` representation. The backbone features are depth-channel-augmented (RGBD input), introducing a distribution shift compared to the RGB-only pretrain. A scalar gate per layer allows each cross-attention stage to independently learn how strongly to rely on the projected features — with all gates initialized to 1.0 so behavior at epoch 0 exactly matches the baseline. As training proceeds, gates can decay below 1 to suppress noisy backbone signals early in decoding, or amplify them in later layers once queries have refined.

## Proposed Change

Add a `nn.ParameterList` of 4 scalar gates (one per decoder layer) to `Pose3DHead`. Before each decoder layer's cross-attention step, the projected `memory` tensor is element-wise scaled by `sigmoid(gate_i)`. Because `sigmoid(0) = 0.5` and we want gate-open behavior at init, gates are initialized to a large positive value such that `sigmoid(gate_i) ≈ 1.0`. Specifically, initializing each gate to `4.6` gives `sigmoid(4.6) ≈ 0.99`, effectively open at the start.

### Why sigmoid rather than a raw scalar?

A raw (unbounded) scalar could go negative, collapsing the cross-attention input. `sigmoid` bounds the gate in `(0, 1)`, guaranteeing the gate is always a non-negative scaling factor. This prevents degenerate solutions while still allowing the gate to approach 0 (suppress) or 1 (pass through).

### Architecture change

The current `Pose3DHead.forward` delegates entirely to `nn.TransformerDecoder`, which does not expose per-layer memory hooks. To implement per-layer gating, the forward pass must call each `TransformerDecoderLayer` manually rather than calling `self.decoder(queries, memory)` as a single unit. The stored decoder layers are accessible via `self.decoder.layers`.

#### Forward pass (new logic):

```python
def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
    B = feat.size(0)
    memory = self.input_proj(feat.flatten(2).transpose(1, 2))   # (B, S, hidden_dim)
    out = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, J, hidden_dim)
    for layer, gate in zip(self.decoder.layers, self.gates):
        gated_memory = torch.sigmoid(gate) * memory             # broadcast over (B, S, hidden_dim)
        out = layer(out, gated_memory)                           # standard decoder layer call
    if self.decoder.norm is not None:
        out = self.decoder.norm(out)
    pelvis_token = out[:, 0, :]
    return {
        "joints":       self.joints_out(out),
        "pelvis_depth": self.depth_out(pelvis_token),
        "pelvis_uv":    self.uv_out(pelvis_token),
    }
```

#### `__init__` addition (inside `Pose3DHead.__init__`, after the existing decoder construction):

```python
# Per-layer input feature gates: sigmoid(gate) ∈ (0,1), init ≈ 1.0
self.gates = nn.ParameterList([
    nn.Parameter(torch.tensor(4.6))   # sigmoid(4.6) ≈ 0.99
    for _ in range(num_layers)
])
```

#### `_init_weights` — no change needed for gates (already initialized in `__init__` via `torch.tensor(4.6)`).

### Implementation scope

Only `model.py` changes:
1. Add `self.gates` `ParameterList` in `Pose3DHead.__init__`.
2. Replace the single `self.decoder(queries, memory)` call in `forward` with the explicit per-layer loop above.
3. No changes to `train.py`, `infra.py`, `transforms.py`, or `config.py` beyond `output_dir`.

The `gates` parameters are inside `model.head`, so they are automatically included in the `lr_head` param group of the LLRD optimizer without any modification to `train.py`.

## Parameter Count Estimate

Baseline head: ~5.48M params.

Each gate is 1 scalar. 4 gates = 4 additional parameters (negligible).

New head total: ~5.48M + 4 ≈ 5.48M params (effectively unchanged).

No VRAM impact.

## Configuration (`config.py` changes)

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design004"

# Model — identical to baseline; gating implemented purely in model.py
head_hidden     = 256
head_num_heads  = 8
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

No new config fields are needed — the gate initialization value (4.6) is hardcoded in `model.py` and is not a tunable hyperparameter exposed to `config.py`.

## Verification Steps for Builder

1. After model construction, confirm:
   ```python
   print([float(torch.sigmoid(g).item()) for g in model.head.gates])
   # Expected: [~0.99, ~0.99, ~0.99, ~0.99]
   ```
2. Confirm that the optimizer's head param group includes the gate parameters:
   ```python
   head_params = list(model.head.parameters())
   gate_params = list(model.head.gates.parameters())
   # All gate_params should appear in head_params
   ```
3. A quick 1-batch forward pass should produce identical outputs to the baseline at random init, since `sigmoid(4.6) ≈ 0.99` is not exactly 1.0 — a very small discrepancy is expected and acceptable.
4. Confirm `output_dir` in `config.py` points to `runs/idea009/design004`.

## Expected Outcome

Expecting val_mpjpe_body to match or modestly improve on the 112.3 mm baseline (0–2 mm improvement). The gate mechanism's primary value is regularization under distributional shift from the depth channel — the network can learn to suppress backbone features at certain decoder layers if they are noisy with respect to joint queries that have not yet converged. If the gates converge near 0.99 (nearly open), the mechanism has no effect and results match the baseline. If some gates drop significantly (e.g., gate 0 settles near 0.5), that indicates depth-augmented features are less useful in early decoder layers and the gating is doing real work. In either case, the experiment is informative.
