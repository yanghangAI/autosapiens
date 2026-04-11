# Review: idea016/design001 — 2D Heatmap + Scalar Depth (40×24 native grid)

**Design_ID:** idea016/design001  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Replaces the final `Linear(384,3)` regression head with a two-branch output: (1) a 2D heatmap branch `Linear(384, 40*24)` → softmax → soft-argmax → normalized (u,v) coordinates, and (2) a scalar depth branch `Linear(384,1)` per joint. GT joints are converted to normalized UV space in train.py for the XY loss. Depth loss is in metres directly. MPJPE metric is computed by decoding UV+Z back to root-relative metres via a `decode_joints_heatmap` helper. All other hyperparameters unchanged.

---

## Evaluation

### 1. Fidelity to idea.md Axis B1

- **2D heatmap branch:** `Linear(384, 40*24)` → reshape `(B,70,40,24)` → softmax → soft-argmax → (u,v) in `[0,1]`. Correct.
- **Depth branch:** `Linear(384,1)` per joint, root-relative metres. Correct.
- **Coordinate buffers precomputed:** `grid_u`, `grid_v` registered as buffers, not recomputed each forward. Correct.
- **Zero-init heatmap bias:** `nn.init.zeros_(self.heatmap_out.bias)`. Correct.

### 2. Metric Conversion (Critical Constraint)

The design investigates `CropPerson` and the data pipeline. It provides a specific formula using camera intrinsics `K` and `gt_pelvis_abs` to convert GT joints to normalized UV coordinates for the XY loss:

```
X_ref = X_pel + joints[:,:,2:3]  # per-joint world depth
u_px = K[:,0:1,0:1] * (-joints[:,:,0:1]) / X_ref + K[:,0:1,2:3]
```

This is geometrically sound: it uses exact per-joint depth for the projection (not just pelvis depth as a proxy), which avoids the chicken-and-egg problem for the GT side. The GT side is computed directly from GT Z — no circularity.

**Key concern:** The design requires `K` (camera intrinsics) and `gt_pelvis_abs` to be available in the training loop. These must be present in `batch` from the dataloader. The design asserts they are available (`batch["pelvis_abs"]` and camera matrix). This should be verified during implementation, but it is consistent with the data pipeline architecture that has been used in prior ideas.

- **decode_joints_heatmap helper:** Fully specified. Converts predicted `(u_norm, v_norm, z_rel)` → root-relative `(x,y,z)` using intrinsics. Used for MPJPE only (not for loss). Correct separation.

### 3. Hyperparameter Completeness

All required hyperparameters inherited. New fields: `heatmap_h=40`, `heatmap_w=24`, `lambda_z_joint=1.0`. Loss weights: `l_pose = l_xy + l_z` (equal weight 1.0 each, plus existing `lambda_depth=0.1`, `lambda_uv=0.2`). Complete.

### 4. Mathematical Correctness

- Soft-argmax: `pred_u = (hm_soft * grid_u).sum(-1)` — correct. Buffers are shape `(H*W,)`, broadcast over `(B,70,H*W)`.
- Loss in UV space vs. metric space: The XY loss is in normalized UV `[0,1]` while the depth loss is in metres. These are on different scales. Using `l_pose = l_xy + l_z` gives equal weight to both, but UV errors (e.g., 0.01 normalized = small pixel shift) are numerically smaller than depth errors in metres. This scale mismatch may cause the optimizer to implicitly weight depth more. **However**, this is an experimental choice the idea.md sanctions (the metric conversion constraint allows a proxy). The design is self-consistent and explicit about the loss space. Not a fatal flaw.

### 5. Architecture Feasibility

- New params: `Linear(384,960)` = 369K. Replaces `Linear(384,3)` = 1.15K. Net +368K. Negligible.
- Heatmap tensor `(4, 70, 960)` ≈ 1.1 MB. Within budget.
- No change to decoder, backbone, or LLRD.

### 6. Changes Required

Clearly specified: model.py, config.py, train.py changes enumerated. transforms.py unchanged. Builder instructions are actionable.

### 7. Constraint Adherence

- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup=3: fixed.
- Wide head trunk (384, 8 heads, 4 layers): unchanged.
- lambda_depth=0.1, lambda_uv=0.2: unchanged.
- infra.py, transforms: not modified.

---

## Issues Found

**Minor:** The XY and Z loss components are in different units (normalized UV vs. metres), creating an implicit scale imbalance. The design implicitly assigns equal weight (`l_xy + l_z` with both weight 1.0) — acceptable as an experimental starting point. `lambda_z_joint=1.0` in config.py is informational but not actually used in the loss formula to reweight Z relative to XY. Builder should implement as described (equal weights) and observe empirically.

**Minor:** `depth_joint_out` and `depth_out` are separate modules (one per-joint, one for pelvis aux). This is intentional and correctly described.

No fatal issues.

---

## Verdict: APPROVED
