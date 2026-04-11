# Review: idea016/design002 — 2D Heatmap + Scalar Depth (80×48 upsampled grid)

**Design_ID:** idea016/design002  
**Date:** 2026-04-11  
**Verdict:** APPROVED

---

## Summary of Design

Same as design001 but the 40×24 heatmap logits are bilinearly upsampled to 80×48 before softmax + soft-argmax. The output head still produces `Linear(384, 40*24=960)` logits; upsampling happens in the forward pass via `F.interpolate`. Coordinate buffers are for the 80×48 grid (3840 bins). Loss and metric conversion identical to design001.

---

## Evaluation

### 1. Fidelity to idea.md Axis B2

- **Upsampled to 80×48:** Correctly specified. Bilinear interpolation of logit map before softmax — this is the correct approach (upsample logits, not probabilities).
- **Output linear:** Still `Linear(384, 40*24)` — correct. Upsampling is in the forward, not the weight count.
- **Coordinate buffers:** Registered for 80×48 resolution (`up_h=80, up_w=48`). Correct.
- **Depth branch:** Unchanged from design001. Correct.

### 2. Metric Conversion

Identical to design001. The GT UV target computation and `decode_joints_heatmap` helper are the same. The upsampling is internal to the head (different grid resolution for argmax, same normalized `[0,1]` output). Correct.

### 3. Mathematical Correctness

- `F.interpolate` on `(B*70, 1, 40, 24) → (B*70, 1, 80, 48)` with bilinear: correct reshaping pattern.
- After upsample: `hm_up` shape `(B, 70, 3840)` → softmax → soft-argmax with 3840-element grid buffers. Correct.
- `align_corners=False`: appropriate for non-pixel-center interpolation. Consistent with PyTorch defaults.

### 4. Architecture Feasibility

- Upsampled tensor `(4, 70, 80, 48)` = 1.07M floats = 4.3 MB. Negligible.
- `F.interpolate` on `(280, 1, 40, 24)` is trivially fast.
- No increase in trainable parameters vs. design001 (same `Linear(384, 960)`).
- Memory overhead is minimal.

### 5. Hyperparameter Completeness

All required fields inherited. New fields: `heatmap_h=40`, `heatmap_w=24`, `upsample_factor=2`, `lambda_z_joint=1.0`. Complete.

### 6. Loss / Config

`output_dir` correctly updated to `runs/idea016/design002`. Changes Required section clear and actionable.

### 7. Constraint Adherence

All constraints from idea.md satisfied. infra.py, transforms unchanged. Decoder, backbone, LLRD unchanged.

---

## Issues Found

None. Design002 is a clean extension of design001 with a single well-motivated change (4× finer soft-argmax grid). The rationale (sub-pixel accuracy improvement) is valid.

---

## Verdict: APPROVED
