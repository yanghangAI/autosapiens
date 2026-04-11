# Review: idea016/design004 — 2D Heatmap + Scalar Depth + Auxiliary Gaussian MSE Supervision

**Design_ID:** idea016/design004  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Same as design001 (2D heatmap + scalar depth at native 40×24) plus an auxiliary MSE loss on the predicted softmax heatmap against a Gaussian target centred at GT `(u,v)` with sigma=2.0 grid cells. Weight `lambda_hm=0.1`. The head returns `"heatmap_soft"` in addition to joints and pelvis outputs. The `make_gaussian_targets` helper generates normalized Gaussian targets on-the-fly in train.py.

---

## Evaluation

### 1. Fidelity to idea.md Axis B4

- **Same as B1 + Gaussian MSE:** Correct. Architecture identical to design001 except head returns heatmap softmax.
- **Gaussian sigma=2.0 grid cells:** Correct.
- **lambda_hm=0.1:** Correct.
- **MSE on body joints only:** `pred_hm = out["heatmap_soft"][:, BODY_IDX]` → `(B, 22, H*W)`. Correct (22 body joints, matching BODY_IDX).
- **Normalization of Gaussian:** Sum to 1 (divided by sum + 1e-8). This matches the softmax output range, making MSE meaningful.

### 2. Gaussian Target Generation

```python
gauss = exp(-((grid_u - mu_u)^2 + (grid_v - mu_v)^2) / (2 * sigma^2))
gauss = gauss / (gauss.sum(-1, keepdim=True) + 1e-8)
```

- `mu_u = gt_uv[:,0] * (hm_w-1)`, `mu_v = gt_uv[:,1] * (hm_h-1)` — converts normalized `[0,1]` UV to grid coordinates. Correct.
- `grid_u`, `grid_v` are `[0, hm_w-1]` and `[0, hm_h-1]` integer coordinates. Correct.
- Normalization ensures Gaussian sums to 1 — matches softmax output (which also sums to 1). MSE comparison is well-defined.

**Note:** The Gaussian target is generated at training time on GPU. For `(B=4, 70, 960)` this is a trivial operation with negligible cost.

### 3. Loss Computation

```python
loss = (l_pose + lambda_depth*l_dep + lambda_uv*l_uv + lambda_hm*l_hm) / accum_steps
```

- `l_pose = l_xy + l_z` (same as design001)
- `l_hm = F.mse_loss(pred_hm, gt_hm)` where both are `(B, 22, H*W)`
- Total loss: consistent with idea.md constraint (lambda_depth=0.1, lambda_uv=0.2, lambda_hm=0.1 all unchanged from spec)

The design also correctly requires `gt_uv_joints` to be computed before calling `make_gaussian_targets` — this is the same computation as the XY loss target, so no double work.

### 4. Architecture Feasibility

- Additional return: `"heatmap_soft"` key in output dict. Negligible overhead (heatmap_soft is already computed in the forward).
- Gaussian targets: `(4, 70, 960)` = 1.07M floats = 4.3 MB. Trivial.
- MSE on `(4, 22, 960)`: trivial.

### 5. Hyperparameter Completeness

New config fields: `heatmap_h=40`, `heatmap_w=24`, `lambda_hm=0.1`, `hm_sigma=2.0`, `lambda_z_joint=1.0`. All required hyperparameters inherited. Complete.

### 6. Builder Instructions

`make_gaussian_targets` helper is fully specified with code. Loss computation block is complete. `l_hm` logging in iter_logger specified. All 4 required file changes enumerated clearly.

### 7. Constraint Adherence

- lambda_hm=0.1: matches idea.md spec.
- sigma=2.0: matches spec.
- BODY_IDX applied to heatmap MSE: matches spec ("summed over body joints").
- infra.py, transforms: unchanged.
- BATCH_SIZE=4, epochs=20: fixed.

---

## Issues Found

None. Design004 is a clean, well-specified extension of design001. The Gaussian target generation is mathematically sound and the MSE loss formulation is consistent with the softmax output.

---

## Verdict: APPROVED
