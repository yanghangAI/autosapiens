# Review: idea009/design003 — Sine-Cosine Joint Query Initialization

**Reviewer:** Reviewer Agent
**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Summary

Design 003 targets Axis B1 from idea009: replacing the random `trunc_normal_(std=0.02)` initialization of the 70 × 256 joint query embedding with standard Vaswani et al. (2017) sinusoidal positional encodings. The weights remain fully trainable after initialization. All other hyperparameters and architecture components are identical to idea004/design002.

---

## Evaluation

### 1. Fidelity to Idea Spec

The idea.md (Axis B1) calls for:
- Sine-cosine encodings derived from joint index 0..69.
- `nn.Embedding(70, 256)` weights remain learnable post-init.
- All other components fixed to idea004/design002.

The design faithfully implements this: the `_sinusoidal_init` helper computes the standard PE formula and copies it into the embedding weights via `torch.no_grad()`. No other components are altered. **Pass.**

### 2. Mathematical Correctness

The PE formula matches the canonical Vaswani et al. form:

```
PE[j, 2k]   = sin(j / 10000^(2k / d_model))
PE[j, 2k+1] = cos(j / 10000^(2k / d_model))
```

The implementation uses the numerically stable log-space form for the division term:

```python
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
```

This is equivalent and avoids large intermediate values. With d_model=256 (even), `arange(0, 256, 2)` has 128 elements, correctly producing 128 sin columns and 128 cos columns to fill the 256-dimensional embedding. **Pass.**

The sanity check suggested (`weight[0, :4] ≈ [0.0, 1.0, 0.0, 1.0]`) is correct: at j=0, sin(0)=0 and cos(0)=1 for all frequencies, so the first four values should be [sin(0), cos(0), sin(0·...),cos(0·...)] = [0, 1, 0, 1]. **Pass.**

### 3. Parameter Count

The change is initialization-only; tensor shapes are unchanged. Parameter count remains identical to the baseline (~5.48M head, ~298M total). **Pass.**

### 4. VRAM / 20-Epoch Proxy Constraint

No additional parameters or activations are introduced. VRAM usage is identical to the baseline (well within 11GB on 1080ti). Training speed is unaffected. **Pass.**

### 5. Configuration Completeness

All required config fields are explicitly specified in the design:

| Field | Value |
|---|---|
| output_dir | runs/idea009/design003 |
| head_hidden | 256 |
| head_num_heads | 8 |
| head_num_layers | 4 |
| lr_backbone | 1e-4 |
| lr_head | 1e-4 |
| gamma | 0.90 |
| unfreeze_epoch | 5 |
| warmup_epochs | 3 |
| epochs | 20 |
| weight_decay | 0.03 |
| grad_clip | 1.0 |
| lambda_depth | 0.1 |
| lambda_uv | 0.2 |
| head_dropout | 0.1 |
| drop_path | 0.1 |

No ambiguities for the Builder. **Pass.**

### 6. Implementation Instructions

The design is explicit and correct:
- Change is isolated to `Pose3DHead._init_weights` in `model.py` only.
- `_sinusoidal_init` must be called after `nn.Embedding` construction.
- `torch.no_grad()` context is correctly used to prevent autograd graph pollution at init time.
- The design correctly notes that `nn.Embedding` does not call `reset_parameters()` after `__init__`, so the sinusoidal values will not be overwritten.
- No other files (`train.py`, `infra.py`, `transforms.py`) require modification.

**Pass.**

### 7. Risks / Concerns

None significant. The initialization magnitude is in [-1, 1], which is larger than the `trunc_normal_(std=0.02)` baseline but will be absorbed during the warmup epochs. This is a common and well-understood initialization strategy. The risk of harming convergence is low; the design's expected outcome (0–2 mm improvement or neutral result) is realistic.

---

## Conclusion

The design is mathematically sound, completely specified, isolated to a single file, and within all resource constraints. The Builder has unambiguous instructions with no guesswork required.

**VERDICT: APPROVED**
