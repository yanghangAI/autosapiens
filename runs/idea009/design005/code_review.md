# Code Review — idea009/design005

**Design_ID:** idea009/design005
**Date:** 2026-04-09
**Reviewer:** Reviewer Agent
**Verdict:** APPROVED

---

## Summary

Design 005 adds a single `nn.LayerNorm(hidden_dim)` (`self.output_norm`) to `Pose3DHead`, inserted between the transformer decoder output and the three final regression heads (`joints_out`, `depth_out`, `uv_out`). All hyperparameters and the LLRD schedule are identical to `idea004/design002`. Only `model.py` and `output_dir` in `config.py` are changed.

---

## Design-vs-Implementation Verification

### config.py

All fields match the design spec exactly:

| Field | Design spec | Actual |
|---|---|---|
| `output_dir` | `runs/idea009/design005` | `runs/idea009/design005` |
| `head_hidden` | 256 | 256 |
| `head_num_heads` | 8 | 8 |
| `head_num_layers` | 4 | 4 |
| `head_dropout` | 0.1 | 0.1 |
| `drop_path` | 0.1 | 0.1 |
| `lr_backbone` | 1e-4 | 1e-4 |
| `lr_head` | 1e-4 | 1e-4 |
| `gamma` | 0.90 | 0.90 |
| `unfreeze_epoch` | 5 | 5 |
| `warmup_epochs` | 3 | 3 |
| `epochs` | 20 | 20 |
| `weight_decay` | 0.03 | 0.03 |
| `grad_clip` | 1.0 | 1.0 |
| `lambda_depth` | 0.1 | 0.1 |
| `lambda_uv` | 0.2 | 0.2 |

No experiment-specific values are hardcoded in `train.py`. All correct.

### model.py — `Pose3DHead.__init__`

The `self.output_norm = nn.LayerNorm(hidden_dim)` line is added after `self.decoder` and before `self.joints_out` — exactly as specified. PyTorch default initialization (weight=1, bias=0) is preserved; the design correctly notes no change to `_init_weights` is required (and none was made).

### model.py — `Pose3DHead.forward`

```python
out = self.decoder(queries, memory)          # (B, num_joints, hidden_dim)
out = self.output_norm(out)                  # normalize before regression
pelvis_token = out[:, 0, :]                  # (B, hidden_dim) — pelvis query
```

The placement is exactly as specified: immediately after the decoder, before `pelvis_token` extraction and all three output projections. Correct.

### Optimizer coverage

`output_norm` is an `nn.Module` attribute of `model.head`. `_build_optimizer_frozen` and `_build_optimizer_full` both use `list(model.head.parameters())` which enumerates all head submodules recursively — `output_norm.weight` and `output_norm.bias` are automatically included in the `lr_head` param group. No optimizer code change was needed or made. Correct.

### train.py

No changes beyond the stale docstring reference to "idea004/design002" (cosmetic, carried over from baseline — not a bug). All LLRD logic, warmup schedule, gradient clipping, and loss computation are intact and unchanged.

---

## Correctness Checks

- **Positioning:** `nn.TransformerDecoder` with `norm_first=True` and no explicit `norm` argument does not apply a final layer norm. `output_norm` correctly fills this gap, matching canonical pre-norm transformer design.
- **Shape:** Input `out` is `(B, num_joints, hidden_dim)`. `LayerNorm(hidden_dim)` normalizes over the last dimension — shape is preserved unchanged. Correct.
- **Parameter count:** 512 new parameters (256 weight + 256 bias). Negligible. Total ~308.8M (confirmed by smoke test).
- **VRAM:** GPU allocated 1.76 GB / reserved 4.22 GB after batch 1 — well within the 11 GB 1080ti limit.

---

## Smoke Test

2-epoch test (`slurm_test_55368331.out`) passed cleanly:

- Model loaded 293/293 backbone tensors; head (83 tensors, including `output_norm`) randomly initialized.
- 13 param groups as expected (frozen phase).
- Loss decreased each epoch. Training and val metrics stable.
- No errors or warnings.

---

## Issues

None. The implementation is a minimal, correct, and fully faithful transcription of the design.

---

## Verdict

**APPROVED**

`model.py` and `config.py` match the design spec exactly. No hardcoded hyperparameters in `train.py`. `output_norm` is correctly placed and auto-included in the optimizer. Smoke test passed. Ready for full 20-epoch run.
