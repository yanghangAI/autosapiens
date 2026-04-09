# Review — idea009/design004 (Per-Layer Input Feature Gate)

**Verdict: APPROVED**

**Reviewed:** 2026-04-09

---

## Summary

Design004 implements Axis B2 from idea009/idea.md: a shared `Linear(1024→256)` input projection (unchanged from baseline) augmented with 4 learned scalar gates (one per decoder layer), each sigmoid-activated, controlling how much of the projected backbone feature tensor passes into each cross-attention step.

---

## Evaluation

### 1. Alignment with idea.md Axis B2
The design exactly matches the specified mechanism: shared projection, one sigmoid gate per decoder layer, gates initialized near 1.0 so epoch-0 behavior is approximately baseline. The idea.md says "initialized to 1.0" — the design correctly uses sigmoid(4.6) ≈ 0.99 because sigmoid cannot reach exactly 1.0, and explicitly notes this small discrepancy. This is mathematically honest and acceptable.

### 2. Gate Initialization
Each gate is `nn.Parameter(torch.tensor(4.6))`. sigmoid(4.6) = 0.9899... ≈ 0.99. Negligible deviation from the open-gate baseline at epoch 0. Correct.

### 3. Sigmoid Bounding Justification
The design correctly notes that a raw (unbounded) scalar could go negative and collapse cross-attention inputs. Sigmoid bounds gates to (0,1), guaranteeing non-negative scaling. Sound reasoning.

### 4. Forward Pass Architecture
The design correctly replaces the single `self.decoder(queries, memory)` call with an explicit loop over `self.decoder.layers`. The final `decoder.norm` is correctly preserved with a `None` guard. Broadcasting of the 0-dimensional scalar sigmoid output over `(B, S, hidden_dim)` memory tensor is correct in PyTorch. No shape issues.

### 5. LLRD Optimizer Compatibility
The `nn.ParameterList` of gates lives inside `Pose3DHead`, so all 4 gate parameters are automatically included in the `lr_head` param group. No change to `train.py` optimizer construction needed. Correct.

### 6. Parameter Count and VRAM
4 scalar parameters added to a ~5.48M parameter head — completely negligible. No VRAM impact. Fits well within the 1080ti 11GB budget. Correct.

### 7. Config Completeness
All 16 hyperparameter fields are specified explicitly: `output_dir`, `head_hidden`, `head_num_heads`, `head_num_layers`, `lr_backbone`, `lr_head`, `gamma`, `unfreeze_epoch`, `warmup_epochs`, `epochs`, `weight_decay`, `grad_clip`, `lambda_depth`, `lambda_uv`, `head_dropout`, `drop_path`. The gate init value (4.6) is implementation-internal and correctly not exposed as a config field.

### 8. Constraint Adherence
- 20 epochs: confirmed.
- batch=4, ACCUM_STEPS=8: unchanged (infra.py untouched).
- LLRD schedule (gamma=0.90, unfreeze_epoch=5, base_lr=1e-4): preserved.
- RGBD input with baseline depth normalization: unchanged.
- Only `model.py` modified (init + forward). No changes to `train.py`, `infra.py`, `transforms.py`.

### 9. Builder Instructions
Verification steps are concrete and testable:
1. Gate sigmoid values at init (~0.99 for all 4) — easy to confirm.
2. Gate parameters appear in head param group — easy to confirm.
3. 1-batch forward pass produces near-identical output to baseline (small discrepancy from 0.99 vs 1.0 expected and acceptable).
4. output_dir verification.

### 10. No Issues Found
The design is complete, self-consistent, mathematically sound, and well-scoped. The incremental nature (4 scalar parameters, init-matched to baseline) makes this a clean experiment with interpretable outcomes regardless of result direction.

---

## Decision

**APPROVED.** The design is complete, feasible, and correctly specified. The Builder can implement directly from this design.
