# Code Review — idea009/design004 (Per-Layer Input Feature Gate)

**Design_ID:** idea009/design004
**Reviewed:** 2026-04-09
**Verdict: APPROVED**

---

## Overview

Design004 implements Axis B2: four learned scalar gates (one per TransformerDecoder layer) that
sigmoid-scale the projected backbone feature tensor (`memory`) before each cross-attention step.
Only `model.py` is modified; `train.py`, `config.py`, `transforms.py`, and `infra.py` are
otherwise identical to the baseline (idea004/design002 / idea009 starting point).

---

## Checklist

### 1. `model.py` — Gate Initialization

```python
self.gates = nn.ParameterList([
    nn.Parameter(torch.tensor(4.6))   # sigmoid(4.6) ≈ 0.99
    for _ in range(num_layers)
])
```

Matches design spec exactly. `num_layers` is passed from `config.head_num_layers = 4` through
`SapiensPose3D → Pose3DHead(num_layers=head_num_layers)`. Four gates created, each initialized to
4.6 so `sigmoid(4.6) ≈ 0.99` — effectively open at epoch 0. Correct.

### 2. `model.py` — `_init_weights`

The `_init_weights` method is unchanged: `trunc_normal_` for `joint_queries.weight`, zero-init
bias for regression heads. Gates are NOT re-initialized inside `_init_weights` (they retain
`torch.tensor(4.6)` set in `__init__`). This is exactly correct per design spec.

### 3. `model.py` — `forward` Per-Layer Loop

```python
for layer, gate in zip(self.decoder.layers, self.gates):
    gated_memory = torch.sigmoid(gate) * memory   # broadcast over (B, S, hidden_dim)
    out = layer(out, gated_memory)
if self.decoder.norm is not None:
    out = self.decoder.norm(out)
```

Matches the design pseudocode exactly. Key correctness points:
- `torch.sigmoid(gate)` returns a 0-dim scalar tensor; broadcasting over `(B, S, hidden_dim)` memory is correct in PyTorch.
- `layer(out, gated_memory)` is the standard `nn.TransformerDecoderLayer(tgt, memory)` call — correct signature.
- The `decoder.norm is not None` guard is correct; since the `nn.TransformerDecoder` is constructed without an explicit `norm=` argument, `self.decoder.norm` is `None` — the guard fires false and is harmlessly skipped, preserving baseline behavior.
- The single `self.decoder(queries, memory)` call has been fully replaced; no residual call exists.

### 4. `config.py` — All Fields Present and Correct

| Field | Config Value | Design Spec | Match |
|-------|-------------|-------------|-------|
| output_dir | `runs/idea009/design004` | `runs/idea009/design004` | OK |
| head_hidden | 256 | 256 | OK |
| head_num_heads | 8 | 8 | OK |
| head_num_layers | 4 | 4 | OK |
| head_dropout | 0.1 | 0.1 | OK |
| drop_path | 0.1 | 0.1 | OK |
| lr_backbone | 1e-4 | 1e-4 | OK |
| lr_head | 1e-4 | 1e-4 | OK |
| gamma | 0.90 | 0.90 | OK |
| unfreeze_epoch | 5 | 5 | OK |
| warmup_epochs | 3 | 3 | OK |
| epochs | 20 | 20 | OK |
| weight_decay | 0.03 | 0.03 | OK |
| grad_clip | 1.0 | 1.0 | OK |
| lambda_depth | 0.1 | 0.1 | OK |
| lambda_uv | 0.2 | 0.2 | OK |

All 16 fields match. Gate init value (4.6) is correctly kept internal to `model.py` and not
exposed as a config field, per design instruction.

### 5. `train.py` — No Changes Required; Optimizer Compatibility

Both `_build_optimizer_frozen` and `_build_optimizer_full` group head parameters via
`list(model.head.parameters())`. Since `self.gates` is an `nn.ParameterList` inside
`Pose3DHead`, all 4 gate scalars are automatically included in the `lr_head` param group with no
optimizer code change — exactly as the design specified.

No hyperparameters are hardcoded in `train.py` that should be in `config.py`. The LLRD
schedule, freeze/unfreeze logic, cosine warmup, gradient clip, and loss weights all read from
`args` (the `_Cfg` instance).

### 6. Smoke Test Results

- Completed 2 epochs without error.
- GPU memory after batch 1: allocated=1.76GB, reserved=4.22GB (well within 11GB VRAM budget).
- Param count: 308.8M — consistent with baseline + 4 negligible gate scalars.
- `Param groups: 13 (expect 13)` confirmed at init.
- Training and validation losses decreased monotonically over 2 epochs.

### 7. Minor Notes (Non-Blocking)

- `train.py` docstring still reads "idea004/design002" — stale cosmetic comment inherited from
  baseline. Harmless; does not affect execution.

---

## Decision

**APPROVED.** The implementation faithfully matches the design specification in all respects:
gate initialization, forward loop semantics, sigmoid broadcasting, norm guard, optimizer
grouping, and config completeness. The smoke test passed cleanly within VRAM budget. No bugs
found.
