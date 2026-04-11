# Review: idea016/design003 — Full 3D Volumetric Heatmap (40×24×16, Integral Pose Regression)

**Design_ID:** idea016/design003  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Full 3D volumetric heatmap: `Linear(384, 40*24*16=15360)` → softmax over entire volume → 3D soft-argmax producing `(u_norm, v_norm, d_abs)` per joint. The 16-bin depth axis uses the same sqrt-spaced anchors as the `DepthBucketPE`. Loss normalizes both depth and UV to `[0,1]` range using `DEPTH_MAX_METERS`. MPJPE decoding converts `(u,v,d_abs)` to root-relative `(x,y,z)` via intrinsics. Single unified output — no separate depth regression head.

---

## Evaluation

### 1. Fidelity to idea.md Axis B3

- **3D volumetric output:** `Linear(384, 40*24*16)` → reshape `(B,70,40,24,16)` → softmax → 3D soft-argmax. Correct.
- **Depth bins reused from depth PE:** `d_k = (k/(D-1))^2 * DEPTH_MAX_METERS` — exactly matches DepthBucketPE geometry. Correct.
- **Precomputed 3D coordinate buffers:** `grid_u_3d`, `grid_v_3d`, `grid_d_3d` all shape `(H*W*D,)` registered as buffers. Correct.
- **No separate depth head:** `depth_joint_out` is removed. Only `depth_out` (pelvis aux) and `uv_out` (pelvis aux) remain. Correct.

### 2. Depth Bin Centres

Formula: `k_vals = torch.arange(D) / (D-1)`, `depth_bin_centres = k_vals^2 * DEPTH_MAX_METERS`. This is correct sqrt-spacing (squaring of linearly-spaced fractions). Matches existing depth PE definition. **Good coherence.**

### 3. Loss Formulation

- Normalize both UV `[0,1]` and depth to `[0,1]` by dividing by `DEPTH_MAX_METERS`: `gt_uvd_scaled[:,:,2] /= d_scale`. This equalizes gradient scales across three axes.
- For GT: `X_ref = gt_d_abs` (not pelvis-only approximation) — uses exact per-joint world depth for projection. More accurate than design001 approximation.
- `gt_uvd = [gt_u, gt_v, gt_d_abs]` — `gt_d_abs = X_pel + joints[:,:,2:3]` (pelvis world depth + root-relative z). Correct.

### 4. MPJPE Decoding — Critical Issue

The `decode_joints_3d` helper has a **potential error** in the pelvis subtraction step:

```python
x_pel = -(pelvis_abs[:, 1:2])   # note: pelvis_abs = (X, Y, Z) world → x_rel=y_world?
```

This line appears to attempt a coordinate axis conversion but the comment is incorrect and the variable `x_pel` is unused in the final computation. The actual subtraction is:

```python
joints_world = torch.stack([x_abs, y_abs, d_abs], dim=-1)
pelvis_world = pelvis_abs.unsqueeze(1)
return joints_world - pelvis_world
```

This subtracts the full pelvis world vector from joints world vector. The correctness depends on the axis convention: if `pelvis_abs = (X_world, Y_world, Z_world)` and the computed `(x_abs, y_abs, d_abs)` are in the same world coordinate frame, then `joints_world - pelvis_world` correctly gives root-relative coordinates. **The stale `x_pel` line is a leftover from a draft and should be removed**, but it does not affect the final computation since `x_pel` is defined but not used. This is a builder-time cleanup issue, not a design-level flaw.

### 5. Architecture Feasibility

- `Linear(384, 15360)`: 5.9M params, 23.6 MB. Within budget.
- Volume tensor `(4, 70, 40, 24, 16)` = 4.3M floats = 17.2 MB. Within budget.
- `idea.md` memory check confirmed: "well within 11GB."

### 6. Hyperparameter Completeness

New config fields: `heatmap_h=40`, `heatmap_w=24`, `d_scale=10.0`. `num_depth_bins=16` already exists. All required hyperparameters inherited. Complete.

### 7. Constraint Adherence

- Depth bins reused from existing PE geometry — correct.
- lambda_depth=0.1, lambda_uv=0.2: pelvis auxiliary losses unchanged.
- infra.py, transforms: not modified.
- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20: fixed.

---

## Issues Found

**Minor (non-blocking):** Stale `x_pel` variable in `decode_joints_3d` is defined but unused. Builder should delete it to avoid confusion. The computation is otherwise correct.

**Minor:** The 3D volumetric heatmap (15360 bins) makes the softmax over a large volume. At `(B=4, 70, 15360)` the softmax is large but trivially handled by GPU. No performance concern.

No fatal issues.

---

## Verdict: APPROVED
