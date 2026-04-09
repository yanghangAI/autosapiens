# Design 005 — Output LayerNorm Before Final Regression

## Starting Point

`runs/idea004/design002`

This design builds directly on idea004/design002 (val_mpjpe_body = 112.3 mm), the current best result, which uses the LLRD schedule (gamma=0.90, unfreeze_epoch=5). The head architecture (4 layers, hidden_dim=256), the LLRD schedule, and all other hyperparameters remain identical to the baseline. Only a single `LayerNorm(256)` module is inserted immediately before the three output projection layers.

## Problem

The baseline `Pose3DHead.forward` passes the transformer decoder's output tensor `out` (shape `B × J × 256`) directly into the final linear heads (`joints_out`, `depth_out`, `uv_out`) without any normalization. The final decoder layer (with `norm_first=True` / pre-norm) applies layer normalization *before* its self-attention and cross-attention sub-layers, but the residual outputs that emerge from the decoder's final layer are not re-normalized before regression. This means the activation scale entering the linear regressors can vary across training steps, particularly in the early warm-up epochs when the head's weights change rapidly. Unstable input scales to the output linear layers translate to unstable gradient magnitudes for those layers and can delay convergence.

## Proposed Change

Add a single `nn.LayerNorm(hidden_dim)` module named `self.output_norm` to `Pose3DHead`. Apply it to `out` immediately after the decoder and before extracting `pelvis_token` and running the three output projections. The `LayerNorm` weight (`gamma`) is initialized to 1 and bias (`beta`) to 0 by PyTorch's default, so at epoch 0 the transformation is identity — the baseline behavior is exactly reproduced before any training updates `output_norm`'s parameters.

### Architecture change (minimal)

**`__init__` addition** (one line, after `self.decoder` is constructed):

```python
self.output_norm = nn.LayerNorm(hidden_dim)
```

**`forward` change** (one line added after `out = self.decoder(queries, memory)`):

```python
out = self.decoder(queries, memory)      # (B, num_joints, hidden_dim)
out = self.output_norm(out)              # NEW: normalize before regression
pelvis_token = out[:, 0, :]
return {
    "joints":       self.joints_out(out),
    "pelvis_depth": self.depth_out(pelvis_token),
    "pelvis_uv":    self.uv_out(pelvis_token),
}
```

No other code changes are required. The `output_norm` parameters (`weight` and `bias`, each of size 256) are inside `model.head`, so they are automatically included in the `lr_head` param group of the LLRD optimizer without any modification to `train.py`.

### Why this is a sound modification

- **Pre-norm vs. post-norm positioning:** The decoder layers use `norm_first=True`, meaning each sub-layer normalizes its input. However, `nn.TransformerDecoder` with `norm_first=True` does *not* apply a final norm to the decoder output unless an explicit `norm` argument is passed to its constructor. No such norm is passed in the baseline, so the output of the final decoder layer is an unnormalized residual stream. Adding `output_norm` fills this gap, matching the design of standard pre-norm transformers that include a final LN before the task head.
- **Regression stability:** Final-layer normalization pins the input distribution to the output linear layers to zero mean and unit variance per feature, which has been shown across multiple transformer regression works (e.g., DINOv2 linear probing, ViTPose output heads) to improve convergence rate and final accuracy.
- **Trivial cost:** `LayerNorm(256)` adds 512 parameters (weight + bias, both vectors of length 256). Computation is negligible relative to the backbone. No VRAM impact.

### Relationship to existing designs

- **Design 001** (6-layer decoder) increases head depth.
- **Design 002** (wide head, hidden=384) increases head width.
- **Design 003** (sinusoidal query init) changes query initialization.
- **Design 004** (per-layer gate) adds input-side control.
- **Design 005** (this design) adds output-side normalization. It is strictly orthogonal to all prior designs and acts as the sanity-check baseline for output normalization. If even this minimal change improves performance, it confirms that the unnormalized decoder output was a bottleneck. If results are flat, output normalization is not the limiting factor at this scale.

## Parameter Count Estimate

Baseline head: ~5.48M params.

`LayerNorm(256)`: 256 (weight) + 256 (bias) = **512 parameters**.

New head total: ~5.48M + 0.0005M ≈ **5.48M params** (effectively unchanged).

No VRAM impact.

## Configuration (`config.py` changes)

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design005"

# Model — identical to baseline; output_norm implemented purely in model.py
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

No new config fields are required — `output_norm` requires no hyperparameters; its `hidden_dim` argument is derived from the existing `head_hidden` config field.

## Implementation Notes for Builder

1. In `model.py`, inside `Pose3DHead.__init__`, add after the `self.decoder` line:
   ```python
   self.output_norm = nn.LayerNorm(hidden_dim)
   ```
2. In `Pose3DHead.forward`, add one line immediately after `out = self.decoder(queries, memory)`:
   ```python
   out = self.output_norm(out)
   ```
3. No change to `_init_weights` is needed (PyTorch initializes `LayerNorm` with weight=1, bias=0 by default).
4. No changes to `train.py`, `infra.py`, `transforms.py`, or `config.py` beyond `output_dir`.
5. Verify the optimizer includes `output_norm` parameters in the head param group (it will automatically, since `output_norm` is an attribute of `model.head`).

## Verification Steps for Builder

1. After model construction, confirm `output_norm` is part of head parameters:
   ```python
   head_param_names = [n for n, _ in model.head.named_parameters()]
   assert any("output_norm" in n for n in head_param_names)
   ```
2. Confirm the norm is identity at init:
   ```python
   x = torch.randn(2, 70, 256)
   y = model.head.output_norm(x)
   # y should be approximately layer-normalized x (not the same as x, but gamma=1 beta=0)
   assert y.shape == x.shape
   ```
3. Confirm `output_dir` in `config.py` points to `runs/idea009/design005`.

## Expected Outcome

Expecting val_mpjpe_body to be at or slightly better than the 112.3 mm baseline (0–2 mm improvement). This is intentionally the simplest possible head change — it serves as the lower bound for head refinement experiments. A positive result validates that the missing final LN was hurting convergence; a neutral result narrows future search to the other design axes.
